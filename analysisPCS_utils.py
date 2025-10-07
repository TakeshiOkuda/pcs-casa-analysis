"""
PCS analysis utilities: only for personal use
    - 2022-07-15: TOkuda - created
    - 2025-06-xx: TOkuda - updated
"""
import os, sys
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import datetime as dt
import itertools

import logging
logger = logging.getLogger(__name__)

# for analysisUtils
path2au = os.environ.get('PATH_TO_AU', '/users/tokuda/local/scripts/analysis_scripts')
if not os.path.isdir(path2au):
    logger.error(f"analysisUtils directory not found: {path2au}")
    sys.exit(1)
sys.path.append(path2au)
try:
    import analysisUtils as au
    import tmUtils as tu
except Exception as e:
    logger.error(f"Failed to import analysisUtils from {path2au}: {e}")
    sys.exit(1)

import pandas as pd
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# for casatasks and casatools
path2casa = os.environ.get('PATH_TO_CASA', '/users/tokuda/casa-latest/lib/py/lib/python3.6/site-packages')
if not os.path.isdir(path2casa):
    logger.error(f"CASA python path not found: {path2casa}")
    sys.exit(1)
sys.path.append(path2casa)
try:
    from casatools import msmetadata as msmdtool
    from casatools import table as tbtool
    from casatools import quanta as qa
    from casatasks.private import simutil  # for simutil.simutil().itrf2loc(x,y,z,cx,cy,cz)
except Exception as e:
    logger.error(f"Failed to import CASA modules from {path2casa}: {e}")
    sys.exit(1)


SIMUTIL = simutil.simutil()
PCS_GEODETIC = np.array([-67.755761, -23.069533, 5346.0 + 42.126])


@dataclass
class AntennaDebugInfo:
    """Detailed coordinate breakdown for debugging antenna locations."""

    pad_itrf: np.ndarray
    ant_vector: np.ndarray
    corr_itrf: np.ndarray
    ant_itrf: np.ndarray
    ant_enu_manual: np.ndarray


@dataclass
class AntennaGeometry:
    """Geometric information describing a single antenna."""

    id: int
    name: str
    pad_enu: np.ndarray
    ant_enu: np.ndarray
    relative_to_pcs: np.ndarray
    azimuth: float
    elevation: float
    distance: float
    horizontal_distance: float
    axis_angle: float
    debug: Optional[AntennaDebugInfo] = None


@dataclass
class PCSGeometry:
    """Collection of geometry information shared across PCS utilities."""

    asdm: str
    cofa: np.ndarray
    cofa_lat: float
    cofa_long: float
    pcs_itrf: np.ndarray
    pcs_enu: np.ndarray
    cofa_distance: float
    cofa_horizontal_distance: float
    cofa_azimuth: float
    cofa_elevation: float
    antennas: List[AntennaGeometry] = field(default_factory=list)


def _ensure_vector(x, y, z):
    """Convert CASA tool return values to a 3-element numpy array."""
    return np.array([float(np.atleast_1d(x)[0]),
                     float(np.atleast_1d(y)[0]),
                     float(np.atleast_1d(z)[0])])


def _prepare_geometry(vis, debug=False):
    """Return shared geometry information used by PCS helper functions."""
    asdm = vis.split('.ms')[0]

    cofa_x, cofa_y, cofa_z, cofa_lat, cofa_long = au.getCOFAForASDM(asdm)
    cofa = np.array([cofa_x, cofa_y, cofa_z], dtype=float)

    lon_rad = np.deg2rad(PCS_GEODETIC[0])
    lat_rad = np.deg2rad(PCS_GEODETIC[1])
    pcs_itrf = _ensure_vector(*SIMUTIL.long2xyz(lon_rad, lat_rad, PCS_GEODETIC[2], datum='WGS84'))
    pcs_enu = _ensure_vector(*SIMUTIL.itrf2loc(*pcs_itrf, *cofa))

    cofa_horizontal = np.linalg.norm(pcs_enu[:2])
    cofa_d = np.linalg.norm(pcs_enu)
    cofa_az = np.degrees(np.arctan2(pcs_enu[1], pcs_enu[0]))
    cofa_el = np.degrees(np.arctan2(-pcs_enu[2], cofa_horizontal))

    antlist = au.getAntennaNames(vis)
    pad_locs = au.getPadLOCsFromASDM(asdm)
    dict_ant_enu = au.readAntennaPositionFromASDM(asdm)

    antennas: List[AntennaGeometry] = []
    for ant in antlist:
        antid = au.getAntennaIndex(vis, ant)
        pad_enu = pad_locs[ant]
        ant_offset = dict_ant_enu[ant]['position']
        ant_enu = pad_enu + ant_offset
        ant_pcs = np.array(ant_enu - pcs_enu)

        ant_d = np.linalg.norm(ant_pcs)
        ant_l = np.linalg.norm(ant_pcs[:2])
        ant_az = np.degrees(np.arctan2(ant_pcs[1], ant_pcs[0]))
        ant_el = np.degrees(np.arctan2(ant_pcs[2], ant_l)) if ant_l else 0.0

        axis_vec = -pcs_enu
        axis_norm = np.linalg.norm(axis_vec) * ant_d
        if axis_norm:
            cos_theta = np.clip(np.dot(axis_vec, ant_pcs) / axis_norm, -1.0, 1.0)
            az_rel = np.degrees(np.arccos(cos_theta))
            axis_xy = np.array([axis_vec[0], axis_vec[1], 0.0])
            ant_xy = np.array([ant_pcs[0], ant_pcs[1], 0.0])
            cross_z = np.cross(axis_xy, ant_xy)[2]
            if cross_z < 0.0:
                az_rel = -az_rel
        else:
            az_rel = 0.0

        debug_info: Optional[AntennaDebugInfo] = None
        if debug:
            pad_itrf = au.getAntennaPadXYZ(vis, antennaId=antid)
            corr_itrf = au.computeITRFCorrection(pad_itrf, ant_offset)
            ant_itrf = pad_itrf + corr_itrf
            ant_enu_manual = _ensure_vector(*SIMUTIL.itrf2loc(*ant_itrf, *cofa))
            debug_info = AntennaDebugInfo(
                pad_itrf=pad_itrf,
                ant_vector=ant_offset,
                corr_itrf=corr_itrf,
                ant_itrf=ant_itrf,
                ant_enu_manual=ant_enu_manual,
            )

        antennas.append(
            AntennaGeometry(
                id=antid,
                name=ant,
                pad_enu=pad_enu,
                ant_enu=ant_enu,
                relative_to_pcs=ant_pcs,
                azimuth=ant_az,
                elevation=ant_el,
                distance=ant_d,
                horizontal_distance=ant_l,
                axis_angle=az_rel,
                debug=debug_info,
            )
        )

    return PCSGeometry(
        asdm=asdm,
        cofa=cofa,
        cofa_lat=cofa_lat,
        cofa_long=cofa_long,
        pcs_itrf=pcs_itrf,
        pcs_enu=pcs_enu,
        cofa_distance=cofa_d,
        cofa_horizontal_distance=cofa_horizontal,
        cofa_azimuth=cofa_az,
        cofa_elevation=cofa_el,
        antennas=antennas,
    )


#------------------------------------------------------------------------------
PolRX = {'3':  [ -10.00,  +80.00],
         #'4':  [-170.00,  -80.00],
         #'5':  [ -45.00,  +45.00],
         #'6':  [-135.00,  -45.00],
         '6':  [ 45.00,  -45.00],
         '7':  [ -53.55,  +36.45],
         '8':  [   0.00,  +90.00],
         '9':  [-180.00,  -90.00],
         '10': [ +90.00, +180.00]
         }
#------------------------------------------------------------------------------
PolPCS = {'3': +40.0,
          '6':  +0.0,
          '7':  +0.0
          }
#------------------------------------------------------------------------------
"""
"Coordinates of the ALMA Antenna Pads at the AOS in the ITRF (2000) Frame"
Doc.#: SCID-20.02.04.00-0007-B-REP
"""
def _configure_axis(ax, xlim, ylim, xlabel, ylabel):
    """Apply common styling to antenna position plots."""

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='both', which='major')
    ax.grid(visible=True, linestyle='dashed')


def _plot_antennas(ax_configs: Iterable[dict], antennas: Iterable[AntennaGeometry], pcs_enu: np.ndarray):
    """Render antenna and PCS positions across multiple axes."""

    configs = list(ax_configs)
    antenna_list = list(antennas)

    for ant in antenna_list:
        x, y = ant.ant_enu[:2]
        for config in configs:
            ax = config['ax']
            ax.plot(x, y, '.', color='r')
            if config.get('label_antennas'):
                ax.text(x, y, ant.name)

    for config in configs:
        ax = config['ax']
        ax.plot(pcs_enu[0], pcs_enu[1], '*', color='r', markersize=12)
        if config.get('label_pcs'):
            ax.text(pcs_enu[0], pcs_enu[1], 'PCS')
        ax.plot([pcs_enu[0], 0], [pcs_enu[1], 0], linestyle='dotted', color='b')
        _configure_axis(ax, config['xlim'], config['ylim'], config['xlabel'], config['ylabel'])


def _print_geometry_summary(geometry: PCSGeometry):
    """Print a textual summary of the PCS and CofA geometry."""

    print("COFA (ITRF)                :", geometry.cofa)
    print("COFA Latitude              :", geometry.cofa_lat)
    print("COFA Longitude             :", geometry.cofa_long)
    print("PCS position (ITRF)        :", geometry.pcs_itrf)
    print("PCS position (ENU)         :", geometry.pcs_enu)
    print("Az & El to CofA            : [%.3f,%.3f]" % (geometry.cofa_azimuth, geometry.cofa_elevation))
    print("Distance to CofA (D, l)    : [%.3f,%.3f]" % (geometry.cofa_distance, geometry.cofa_horizontal_distance))


def _print_antenna_details(antenna: AntennaGeometry, debug: bool = False):
    """Emit human-readable information about an antenna."""

    print(f"--- {antenna.name} (antennaId={antenna.id}) ---")
    print("Pad position (ENU)         :", antenna.pad_enu)
    print("Antenna position (ENU)     :", antenna.ant_enu)

    if debug and antenna.debug:
        info = antenna.debug
        print("--- coordinates manually calculated")
        print("Pad position (ITRF)        :", info.pad_itrf)
        print("Antenna vector (ENU)       :", info.ant_vector)
        print("ITRF correction (ITRF)     :", info.corr_itrf)
        print("Antenna position (ITRF)    :", info.ant_itrf)
        print("Antenna position (ENU)     :", info.ant_enu_manual)

    print("Az & El to antenna         : [%.3f,%.3f]" % (antenna.azimuth, antenna.elevation))
    print("Distance to antenna (D, l) : [%.3f,%.3f]" % (antenna.distance, antenna.horizontal_distance))
    print("Az w/ respect to the z-axis: %.3f [deg]" % antenna.axis_angle)


def showPCSPosition(vis, debug=False):

    geometry = _prepare_geometry(vis, debug=debug)
    stem = Path(vis).stem
    repdir = Path(f'report_{stem}')
    prefix = stem.split('_')[-1]
    repdir.mkdir(parents=True, exist_ok=True)

    _print_geometry_summary(geometry)

    plt.ioff()
    fig, axs = plt.subplots(1, 2, dpi=100)

    ax_configs = (
        {
            'ax': axs[0],
            'xlim': (-1000, 1000),
            'ylim': (-6000, 0),
            'xlabel': "X [m]",
            'ylabel': "Y [m]",
            'label_antennas': False,
            'label_pcs': True,
        },
        {
            'ax': axs[1],
            'xlim': (-500, 500),
            'ylim': (-2000, -400),
            'xlabel': "X [m]",
            'ylabel': "Y [m]",
            'label_antennas': True,
            'label_pcs': False,
        },
    )

    csv_path = repdir / f"{prefix}_antenna.csv"
    header = ['AntID', 'Antenna', 'Az', 'El', 'Distance1', 'Distance2', 'Alpha']
    list_alpha: List[float] = []

    with csv_path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(header)

        for antenna in geometry.antennas:
            _print_antenna_details(antenna, debug=debug)

            list_alpha.append(antenna.axis_angle)
            writer.writerow([
                antenna.id,
                antenna.name,
                antenna.azimuth,
                antenna.elevation,
                antenna.distance,
                antenna.horizontal_distance,
                antenna.axis_angle,
            ])

    _plot_antennas(ax_configs, geometry.antennas, geometry.pcs_enu)

    fig.savefig(repdir / f"{prefix}_ArrayConfig.png")
    fig.clf()
    plt.close()

    return list_alpha


def getDirectionToAntenna(vis, debug=False):
    geometry = _prepare_geometry(vis, debug=debug)

    _print_geometry_summary(geometry)

    list_coordinates = []
    for antenna in geometry.antennas:
        _print_antenna_details(antenna, debug=debug)

        list_coordinates.append([
            antenna.id,
            antenna.name,
            antenna.azimuth,
            antenna.elevation,
            antenna.distance,
            antenna.horizontal_distance,
            antenna.axis_angle,
        ])

    return list_coordinates

def getALMADashboardInfo():
    #  0-24: DV01-DV25
    # 25-49: DA41-DA65
    # 50-61: CM01-CM12
    # 62-65: PM01-PM04
    try:
        url_dashboard='https://asa.alma.cl/dashboard2-backend/service/api/antennas'
        data = pd.read_json(url_dashboard)
    except:
        data = None
    return data

def checkShadowingPCS():

    pad_file = Path(path2au) / 'AOS_Pads_XYZ_ENU.txt'

    #PCS ITRF: [2224412.407041001, -5438748.522481113, -2485917.0815638653]
    pcs_enu = np.array([-85.32698112, -5170.29232152, 329.22074719])
    offset = np.array([-39.500318459, -720.398306146, -0.242313891])

    pad_vectors = {}

    with open(pad_file, 'r') as f:
        for line in f:
            if '#' in line:
                continue
            tokens = line.split()
            if len(tokens) < 5:
                continue

            pad_id = tokens[4]
            pad_position = np.array(list(map(float, tokens[:3]))) + offset
            vec = pad_position - pcs_enu
            pad_vectors[pad_id] = (
                vec,
                np.linalg.norm(vec),
                float(tokens[3])
            )

    ant_list =[ "DA41", "DA42", "DA43", "DA44", "DA45", "DA46",
                "DA47", "DA48", "DA49", "DA50", "DA51", "DA52",
                "DA53", "DA54", "DA55", "DA56", "DA57", "DA58",
                "DA59", "DA60", "DA61", "DA62", "DA63", "DA64",
                "DA65",
                "DV01", "DV02", "DV03", "DV04", "DV05", "DV06",
                "DV07", "DV08", "DV09", "DV10", "DV11", "DV12",
                "DV13", "DV14", "DV15", "DV16", "DV17", "DV18",
                "DV19", "DV20", "DV21", "DV22", "DV23", "DV24",
                "DV25",
                "CM01", "CM02", "CM03", "CM04", "CM05", "CM06",
                "CM07", "CM08", "CM09", "CM10", "CM11", "CM12",
                "PM01", "PM02", "PM03", "PM04"
                ]

    ALMAInfo = getALMADashboardInfo()
    if ALMAInfo is None:
        return

    pad_by_ant = {}
    for ant in ant_list:
        tmp = ALMAInfo[ALMAInfo['name']==ant]['pad'].values
        if len(tmp) == 0:
            continue
        pad_by_ant[ant] = tmp[0]

    entries = [
        (ant, pad, *pad_vectors[pad])
        for ant, pad in pad_by_ant.items()
        if pad in pad_vectors
    ]

    if not entries:
        return

    ants, pads, vectors, distances, diameters = zip(*entries)
    vectors = np.vstack(vectors)
    distances = np.array(distances)
    diameters = np.array(diameters)
    unit_vectors = vectors / distances[:, None]

    for idx, (ant, pad) in enumerate(zip(ants, pads)):
        dist0 = distances[idx]
        diam0 = diameters[idx]
        cos_theta = np.clip(unit_vectors @ unit_vectors[idx], -1.0, 1.0)
        sin_theta = np.sqrt(1.0 - np.square(cos_theta))
        baselines = dist0 * sin_theta
        diameter_limits = 0.5 * (diam0 + diameters)
        mask = (distances <= dist0) & (np.arange(len(ants)) != idx)
        if not np.any(baselines[mask] < diameter_limits[mask]):
            print(ant, pad, " no blocking")

    return

"""
Functions for PCS analyses
"""
#------------------------------------------------------------------------------
def CurvePCS(angleWG2, angleWG1, angleRX, A, B):
    powerPCS = A*np.power(np.cos((angleWG2-angleWG1)/180.0*np.pi),2.0)
    return powerPCS*np.power(np.cos((angleRX-angleWG2)/180.0*np.pi),2.0)+B

#------------------------------------------------------------------------------
def fitCurvePCS(angleWG2, power, power_rms, angleWG1, angleRX):
    
    p0 = [angleWG1, angleRX, np.max(power), 0.0] 
    try:
        popt, pcov = curve_fit(CurvePCS, angleWG2, power, sigma=power_rms, p0=p0)
    except:
        popt = [0,0,0,0]
        pcov = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    return popt,pcov

#------------------------------------------------------------------------------
# New curve for fitting - 2023-07-23
def OperationMatrixPolarizer(theta, phi, alpha):
    f = 1.0/(np.power(np.cos(phi-alpha),2.0)+np.power(np.tan(theta),2.0))
    a = f * np.power(np.cos(phi-alpha),2.0)
    b = f * np.cos(phi-alpha) * np.tan(theta)
    c = b
    d = f * np.power(np.tan(theta),2.0)
    return np.array([[a,b],[c,d]])

def coeffPolarizer(theta, phi, alpha):
    f = 1.0/(np.power(np.cos(phi-alpha),2.0)+np.power(np.tan(theta),2.0))
    a = f * np.power(np.cos(phi-alpha),2.0)
    b = f * np.cos(phi-alpha) * np.tan(theta)
    c = b
    d = f * np.power(np.tan(theta),2.0)
    return a,b,c,d

def calcMatrixProduct(a,b,c,d,x,y):
    x1 = a*x + b*y
    y1 = c*x + d*y
    return x1, y1

def responsePCS(theta_WG2, phi_WG2, theta_RX, theta_WG1, phi_WG1, alpha, P0, Poffset):

    # Step 0: Wiregrid #1
    Ex = np.sqrt(P0)
    Ey = 0.0
    #
    a,b,c,d = coeffPolarizer(np.deg2rad(theta_WG1), np.deg2rad(phi_WG1), np.deg2rad(alpha))
    Ex, Ey  = calcMatrixProduct(a,b,c,d,Ex,Ey)

    # Step 1: Wiregrid #2
    a,b,c,d = coeffPolarizer(np.deg2rad(theta_WG2), np.deg2rad(phi_WG2), np.deg2rad(alpha))
    Ex, Ey  = calcMatrixProduct(a,b,c,d,Ex,Ey)

    # Step 2: Receiver
    a,b,c,d = coeffPolarizer(np.deg2rad(theta_RX), 0.0, 0.0)
    Ex, Ey  = calcMatrixProduct(a,b,c,d,Ex,Ey)

    return Ex*Ex + Ey*Ey + Poffset

def fittingPCS(theta_WG2, power, power_rms, phi_WG2, theta_RX, theta_WG1, phi_WG1, alpha):

    p0 = [phi_WG2, theta_RX, theta_WG1, phi_WG1, alpha, np.max(power), np.min(power)]

    # phi_WG2, theta_RX, theta_WG1, phi_WG1, alpha, P0, Poffset
    bound_min = (phi_WG2-20.0, theta_RX-20.0, theta_WG1-20.0, phi_WG1-20.0, alpha-10.0, -np.inf, -np.inf)
    bound_max = (phi_WG2+20.0, theta_RX+20.0, theta_WG1+20.0, phi_WG1+20.0, alpha+10.0, np.inf, np.inf)
    bounds = (bound_min, bound_max)

    try:
        popt, pcov = curve_fit(responsePCS, theta_WG2, power, sigma=power_rms, p0=p0, bounds=bounds)
    except:
        popt = [0,0,0,0,0,0,0]
        pcov = [popt,popt,popt,popt,popt,popt,popt]

    return popt,pcov

#------------------------------------------------------------------------------
def getSpectralAutoData_hack(inputms,iant, dd, scanum):
    """
    Used by Coherence.py, reduc_cutscans.py and reduc_oof.py
    hacked by TO
    """
    mytb = tbtool()
    mytb.open(inputms)
    specTb = mytb.query('ANTENNA1==%d && ANTENNA2==%d && DATA_DESC_ID==%d && SCAN_NUMBER == %d' % (iant, iant ,dd, scanum))
    if 'FLOAT_DATA' in specTb.colnames():
        data_query= 'FLOAT_DATA'
    else:
        data_query='DATA'
    specData = specTb.getcol(data_query)
    specTime = specTb.getcol('TIME')
    specFlag = specTb.getcol('FLAG')
    specState= specTb.getcol('STATE_ID')
    specTb.close()
    mytb.close()

    time = []
    for ii in range(len(specTime)):
        timesi = str(specTime[ii])+'s'
        timesi = qa.time(timesi, prec=10, form='fits') # changed prec from 9 to 10 for 2-kHz SQLD data
        time.append(tu.get_datetime_from_isodatetime(timesi[0]))

    target_dtype = np.complex128 if np.iscomplexobj(specData) else np.float64
    specData = np.asarray(specData, dtype=target_dtype)
    mask = np.asarray(specFlag, dtype=bool)

    cleaned_data = specData.copy()
    cleaned_data[mask] = np.nan

    flagData = np.full(specData.shape, np.nan, dtype=specData.dtype)
    np.copyto(flagData, specData, where=mask)

    return [time, cleaned_data, flagData, specState]

#------------------------------------------------------------------------------
def getSpectralData_hack(inputms, iant1, iant2, dd, scanum):
    """
    hacked by TO
    """
    mytb = tbtool()
    mytb.open(inputms)
    specTb = mytb.query('ANTENNA1==%d && ANTENNA2==%d && DATA_DESC_ID==%d && SCAN_NUMBER == %d' % (iant1, iant2 ,dd, scanum))
    if 'FLOAT_DATA' in specTb.colnames():
        data_query= 'FLOAT_DATA'
    else:
        data_query='DATA'
    specData = specTb.getcol(data_query)
    specTime = specTb.getcol('TIME')
    specTb.close()
    mytb.close()

    time = []
    for ii in range(len(specTime)):
        timesi = str(specTime[ii])+'s'
        #timesi = qa.time(timesi, prec=9, form='fits')
        timesi = qa.time(timesi, prec=10, form='fits') # changed prec from 9 to 10 for 2-kHz SQLD data
        time.append(tu.get_datetime_from_isodatetime(timesi[0]))

    return [time, specData]

"""
Allan standard deviation ASDy(2,T,tau) - BEND-50.00.00.00-324-A-PRO
   - tau : averaging time interval y(t)
   - y(t): average over tau from t to t+tau
   - T   : differencing time interval between data averages y(t+T) and y(t)
    - t>=10*T: >= 10 data points for ASDy calculation
    - T ~ k*tau (k=1,2,3,..,Tmax/tau) 
   - ASDy2(2,T,tau) = sqrt( 0.5*<(y(t+T)-y(t))^2>

Phase stability
- Req-264: ASDy(2,T,tau)<0.042 deg phase 1) for T=0.05-100sec, tau=0.05sec and 2) for T=300sec, tau=10sec

Gain stability is defined as ASDg(2,T) = ASDy(2,T,tau=T) - ALMA-50.00.00.00-583-A-CRE
- Req-261: ASDg(2,T)<1e-3 for T=1~300sec
- Req-262: ASDg(2,T)<3e-3 for T=300sec
- Req-263: ASDg(2,T)<4e-3 for T-0.05-1sec
"""
def getASDg(dt, h):
    """
    Gain stability for ALMA
        getASDg(dt, h)
            - ASDg(2,T) = ASDy(2,T,tau=T)
            - dt = sampling interval for raw data h
            - h = dG/G = 1-V(SQLD)/Average(V(SQLD))
                -  V(SQLD)/Average(V(SQLD)) is also fine as h because "1" is compensated in the calculation below
            - return T, ASDg
    """
    # Total number of samples in raw data set
    S = len(h)

    # Total number of ASDg calculation
    # - Define tau = M*dt to be calculated from 'S'
    # Number of raw data samples in each tau/T interval
    # - M: only divisors of 'S'
    list_M = getDivisors(S)

    if list_M.size == 0:
        return np.array([]), np.array([])

    list_N = (S // list_M).astype(int)
    valid = list_N >= 10
    list_M = list_M[valid]
    list_N = list_N[valid]

    if list_M.size == 0:
        return np.array([]), np.array([])

    T = list_M.astype(float) * dt
    ASDg = np.empty_like(T, dtype=float)

    h = np.asarray(h, dtype=float)

    for idx, (M, N) in enumerate(zip(list_M, list_N)):
        Hj = h.reshape(N, M).mean(axis=1)
        diff = np.diff(Hj)
        ASDg[idx] = np.sqrt(0.5 * np.mean(diff * diff)) if diff.size else np.nan

    return T, ASDg

def getDivisors(N):
    if N <= 0:
        return np.array([], dtype=int)

    divisors = []
    limit = int(np.sqrt(N)) + 1
    for i in range(1, limit):
        if N % i == 0:
            divisors.append(i)
            if i * i != N:
                divisors.append(N // i)
    return np.array(sorted(divisors), dtype=int)

def getASDy(dt, h, tau):
    """
    Phase stability for ALMA
        getASDy(dt, h, tau)
            - dt = sampling interval for raw data H
            - tau = averaging time
            return T, ASDy
            - T = difference time
            - ASDy = ASDy(2, T, tau)
    """
    # Total number of samples in raw data set
    S = len(h)
    # Number of raw data samples in each tau interval
    M = int(tau/dt)
    # Total number of tau intervals in raw data set tmax/tau = S/M
    N = int(S/M) 

    # Minimum number of samples for calculating ASDy
    Nsample_max = 10
    # Total number of ASDy calculation
    N_T = int(S/Nsample_max)

    if M == 0 or N < 2:
        return [], []

    Hj = np.asarray(h[:N * M], dtype=float).reshape(N, M).mean(axis=1)

    max_lag = min(N_T, N - 1)
    if max_lag <= 0:
        return [], []

    diffT = (np.arange(1, max_lag + 1) * tau).tolist()
    ASDy = []

    for lag in range(1, max_lag + 1):
        diff = Hj[lag:] - Hj[:-lag]
        ASDy.append(np.sqrt(0.5 * np.mean(diff * diff)))

    return diffT, ASDy

#------------------------------------------------------------------------------    
def getScansForIntent(vis, intent):
    mytb = au.createCasaTool(au.msmdtool)
    mytb.open(vis)
    scanlist = mytb.scansforintent(intent)
    mytb.close()
    return scanlist

#------------------------------------------------------------------------------
def getSpwsForScan(vis, scan, intent):
    SpwsSQLD = au.getScienceSpwsForScan(vis, scan, intent, tdm=False, fdm=False, sqld=True)
    SpwsFR   = au.getScienceSpwsForScan(vis, scan, intent, tdm=True, fdm=True, sqld=False)

    mymsmd = au.createCasaTool(msmdtool)
    mymsmd.open(vis)
    scanSpws = mymsmd.spwsforscan(scan)
    mymsmd.close()

    tmp_SpwsCA = np.setdiff1d(scanSpws,SpwsSQLD)
    tmp_SpwsCA = np.setdiff1d(tmp_SpwsCA,SpwsFR)
    
    # Remove WVR's SPW
    SpwsCA = []
    for spw in tmp_SpwsCA:
        freq = getFrequenciesGHz(vis, spw)
        if len(freq)==1:
            SpwsCA.append(spw)
    return [SpwsSQLD, SpwsFR, SpwsCA]

#------------------------------------------------------------------------------
def getFrequenciesGHz(vis, spw):
    return au.getFrequencies(vis, spw)/1e9

#------------------------------------------------------------------------------
def checkSqldRotation(vis, polarizer_file=None, Xpol=False, interactive=False):

    intent = 'CALIBRATE_DELAY#ON_SOURCE'
    repdir = 'report_'+vis.split('.')[0]
    prefix = vis.split('_')[-1][:-3]

    if (os.path.isdir(repdir) != True):
        os.system("mkdir %s" % repdir)

    antlist  = au.getAntennaNames(vis)
    scanlist = getScansForIntent(vis, intent)
    scan = scanlist[0]
    SpwsSQLD, SpwsFR, SpwsCA = getSpwsForScan(vis, scan, intent)

    nant = len(antlist) 
    nspw = len(SpwsSQLD)

    # Check polarization angle
    if polarizer_file != None and polarizer_file!='None':
        p_time, p_angle = readPolarizerFile(polarizer_file)

    for antidx in range(nant):
        
        ant = antlist[antidx]
        antid = au.getAntennaIndex(vis,ant)
        print("Antenna: %s [%d/%d]" % (ant, antidx+1, nant))

        #plt.ioff()
        figTS, axsTS = plt.subplots(4, 2, figsize=[8.27,11.69],dpi=200)
        figTS.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.95,hspace=0.3,wspace=0.3)

        rawdata, vardata = [], []

        for spwidx in range(len(SpwsSQLD)):

            spw  = SpwsSQLD[spwidx]
            ddid = au.getDataDescriptionId(vis, spw)

            print("SPW    : %d [%d/%d]" % (spw, spwidx+1, nspw))

            freq = getFrequenciesGHz(vis, spw)
            if    84.0 <= freq[0] < 116: band='3'
            elif 211.0 <= freq[0] < 275: band='6'
            elif 275.0 <= freq[0] < 373: band='7'

            time,data,fdata,state = getSpectralAutoData_hack(vis,antid,ddid,scan)
            npol  = len(data)
            ndump = len(data[0][0])

            dT = []
            for ii in range(ndump):
                dT.append( (time[ii]-time[0]).total_seconds())

            rawdata.append(data)
            tmp_var = []

            if polarizer_file != None and polarizer_file!='None':
                if Xpol:
                    pol_idx = convertTimeAngle_Xpol(time, p_time, p_angle)
                else:
                    pol_idx = convertTimeAngle(time, p_time, p_angle)
                Pstop_idx = [x for row in pol_idx for x in row]
                all_idx = range(len(dT))
                Pmove_idx = list(set(all_idx) - set(Pstop_idx))          

                dT_Pmove, signal_Pmove = [], []
                dT_Pstop, signal_Pstop = [], []

                for pol in range(npol):
                    
                    tmp_dT_Pmove, tmp_signal_Pmove = [], []
                    tmp_dT_Pstop, tmp_signal_Pstop = [], []

                    for ii in range(len(Pmove_idx)):
                        tmp_dT_Pmove.append(dT[Pmove_idx[ii]])
                        tmp_signal_Pmove.append(data[pol][0][Pmove_idx[ii]])
                    
                    for ii in range(len(Pstop_idx)):
                        tmp_dT_Pstop.append(dT[Pstop_idx[ii]])
                        tmp_signal_Pstop.append(data[pol][0][Pstop_idx[ii]])

                    dT_Pmove.append(tmp_dT_Pmove)
                    dT_Pstop.append(tmp_dT_Pstop)
                    signal_Pmove.append(tmp_signal_Pmove)
                    signal_Pstop.append(tmp_signal_Pstop)

                for pol in range(npol):
                    
                    if pol==0:
                        color = 'b'
                        title = "%s %s Pol-X SPW%d Scan%d" % (prefix,ant,spw,scan)
                    elif pol==1:
                        color = 'g'
                        title = "%s %s Pol-Y SPW%d Scan%d" % (prefix,ant,spw,scan)

                    # Time series plot
                    axsTS[spwidx][pol].plot(dT_Pmove[pol],np.real(signal_Pmove[pol])*1e3, color='k', marker='x', linestyle='None')
                    axsTS[spwidx][pol].plot(dT_Pstop[pol],np.real(signal_Pstop[pol])*1e3, color=color, marker='.', linestyle='None')
                    axsTS[spwidx][pol].set_title(title)
                    axsTS[spwidx][pol].set_xlabel("Time [UTC]")
                    axsTS[spwidx][pol].set_ylabel("BB Power [mW]")
                    axsTS[spwidx][pol].tick_params(axis='both', which='major')
                    axsTS[spwidx][pol].grid(visible=True, linestyle='dashed')
            
                vardata.append(tmp_var)

        if interactive:
            plt.show(block=True)
        else:
            figTS.savefig('%s/%s_%s_Sqld_debug.png' % (repdir,prefix,ant))
            figTS.clf()
        plt.close('all')


#------------------------------------------------------------------------------
def doPCSAnalysisForSQLD(vis, polarizer_file=None, list_az=None, Xpol=False):

    intent = 'CALIBRATE_DELAY#ON_SOURCE'
    repdir = Path(f"report_{vis.split('.')[0]}")
    prefix = vis.split('_')[-1][:-3]

    repdir.mkdir(exist_ok=True)

    antlist = au.getAntennaNames(vis)
    scanlist = getScansForIntent(vis, intent)
    scan = scanlist[0]
    SpwsSQLD, SpwsFR, SpwsCA = getSpwsForScan(vis, scan, intent)

    nant = len(antlist)
    nspw = len(SpwsSQLD)

    if nspw == 0:
        logger.warning("No SQLD spectral windows found for %s", vis)
        return

    # Check polarization angle
    if polarizer_file not in (None, 'None'):
        p_time, p_angle = readPolarizerFile(polarizer_file)
        ffit = (repdir / f"{prefix}_fitting_SQLD.csv").open('w')
        fit_wrt = csv.writer(ffit)
        fit_wrt.writerow(['AE','SPW','Pol','WG1','WG1_err','RX','RX_err','Amp','Amp_err','Offset','Offset_err'])
    else:
        ffit = None
        fit_wrt = None
        p_time = p_angle = None

    for antidx, ant in enumerate(antlist):

        antid = au.getAntennaIndex(vis, ant)
        print("Antenna: %s [%d/%d]" % (ant, antidx+1, nant))

        plt.ioff()
        figTS, axsTS = plt.subplots(4, 2, figsize=[8.27, 11.69], dpi=200)
        figTS.subplots_adjust(left=0.10, right=0.95, bottom=0.05, top=0.95, hspace=0.3, wspace=0.3)

        figGS, axsGS = plt.subplots(4, 2, figsize=[8.27, 11.69], dpi=200)
        figGS.subplots_adjust(left=0.10, right=0.95, bottom=0.05, top=0.95, hspace=0.3, wspace=0.3)

        time_axis = None
        raw_columns: List[List[np.ndarray]] = []
        asd_columns: List[Tuple[np.ndarray, List[np.ndarray]]] = []

        if polarizer_file not in (None, 'None'):
            figPol, axsPol = plt.subplots(2, 1, figsize=[8.27, 11.69], dpi=200)
            figPol.subplots_adjust(left=0.10, right=0.95, bottom=0.05, top=0.95, hspace=0.3, wspace=0.3)

        n_ts_rows, n_ts_cols = axsTS.shape
        n_gs_rows, n_gs_cols = axsGS.shape

        for spwidx, spw in enumerate(SpwsSQLD):

            ddid = au.getDataDescriptionId(vis, spw)

            print("SPW    : %d [%d/%d]" % (spw, spwidx+1, nspw))

            freq = getFrequenciesGHz(vis, spw)
            if 84.0 <= freq[0] < 116:
                band = '3'
            elif 211.0 <= freq[0] < 275:
                band = '6'
            elif 275.0 <= freq[0] < 373:
                band = '7'
            else:
                band = ''

            time, data, fdata, state = getSpectralAutoData_hack(vis, antid, ddid, scan)
            data = np.asarray(data)
            npol = data.shape[0]
            signals = np.asarray(data[:, 0, :])  # SQLD is 1 channel per baseband

            if signals.ndim != 2:
                logger.error("Unexpected SQLD data shape for %s SPW %s: %s", ant, spw, signals.shape)
                continue

            ndump = signals.shape[1]

            if time_axis is None:
                base_time = time[0]
                time_axis = np.fromiter(((t - base_time).total_seconds() for t in time), dtype=float, count=len(time))
            else:
                current_time_axis = np.fromiter(((t - time[0]).total_seconds() for t in time), dtype=float, count=len(time))
                if current_time_axis.size != time_axis.size or not np.allclose(current_time_axis, time_axis):
                    logger.warning("Time axis mismatch for %s SPW %s; using first SPW timing", ant, spw)

            raw_columns.append([])
            asd_columns.append((np.array([], dtype=float), []))

            dtime = time_axis[1] - time_axis[0] if time_axis.size > 1 else 0.0

            for pol in range(npol):

                signal_series = signals[pol]
                real_signal = np.real(signal_series)
                mean_signal = np.mean(real_signal)
                dG = real_signal / mean_signal if mean_signal else np.zeros_like(real_signal)
                diffT, ASDg = getASDg(dtime, dG)

                asd_columns[-1] = (diffT if diffT.size else asd_columns[-1][0], asd_columns[-1][1] + [ASDg])
                raw_columns[-1].append(np.abs(signal_series))

                if pol == 0:
                    color = 'b'
                    title = "%s %s Pol-X SPW%d Scan%d" % (prefix, ant, spw, scan)
                elif pol == 1:
                    color = 'g'
                    title = "%s %s Pol-Y SPW%d Scan%d" % (prefix, ant, spw, scan)
                else:
                    color = None
                    title = "%s %s Pol-%d SPW%d Scan%d" % (prefix, ant, pol, spw, scan)

                if spwidx < n_ts_rows and pol < n_ts_cols:
                    target_axes = axsTS[spwidx][pol]
                    target_axes.plot(time_axis, real_signal * 1e3, color=color)
                    target_axes.set_title(title)
                    target_axes.set_xlabel("Time [UTC]")
                    target_axes.set_ylabel("BB Power [mW]")
                    target_axes.tick_params(axis='both', which='major')
                    target_axes.grid(visible=True, linestyle='dashed')

                if spwidx < n_gs_rows and pol < n_gs_cols:
                    gain_axes = axsGS[spwidx][pol]
                    gain_axes.plot(diffT, ASDg, color=color)
                    gain_axes.set_xscale('log')
                    gain_axes.set_yscale('log')
                    gain_axes.set_xlim(1e-2, 1e3)
                    gain_axes.set_ylim(1e-4, 1)
                    gain_axes.set_title(title)
                    gain_axes.set_xlabel("Time [sec]")
                    gain_axes.set_ylabel("Allan Standard Deviation ASDg(2,T)")
                    gain_axes.tick_params(axis='both', which='major')
                    gain_axes.grid(visible=True, linestyle='dashed')

            if polarizer_file not in (None, 'None') and list_az is not None and band:

                checkSqldRotation(vis=vis, polarizer_file=polarizer_file, Xpol=Xpol, interactive=False)

                idx_arrays = convertTimeAngle_Xpol(time, p_time, p_angle) if Xpol else convertTimeAngle(time, p_time, p_angle)
                idx_arrays = [np.asarray(idx, dtype=int) for idx in idx_arrays]

                flat_stop = np.concatenate([idx for idx in idx_arrays if idx.size], dtype=int) if any(idx.size for idx in idx_arrays) else np.array([], dtype=int)
                all_idx = np.arange(time_axis.size, dtype=int)
                move_idx = np.setdiff1d(all_idx, flat_stop, assume_unique=True)

                colorlist = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

                pol_ave, pol_rms = [], []
                for pol in range(npol):
                    series_abs = np.abs(signals[pol]) * 1e3
                    ave = []
                    rms = []
                    for idx in idx_arrays:
                        if idx.size:
                            chunk = series_abs[idx]
                            ave.append(np.mean(chunk))
                            rms.append(np.std(chunk))
                        else:
                            ave.append(np.nan)
                            rms.append(np.nan)
                    pol_ave.append(np.array(ave))
                    pol_rms.append(np.array(rms))

                    move_times = time_axis[move_idx]
                    stop_times = time_axis[flat_stop] if flat_stop.size else np.array([], dtype=float)
                    move_signal = signals[pol][move_idx]
                    stop_signal = signals[pol][flat_stop] if flat_stop.size else np.array([], dtype=signals.dtype)

                    if spwidx < n_ts_rows and pol < n_ts_cols:
                        axsTS[spwidx][pol].plot(move_times, np.real(move_signal) * 1e3, color='k', marker='x', linestyle='None')
                        axsTS[spwidx][pol].plot(stop_times, np.real(stop_signal) * 1e3, color=colorlist[spwidx % len(colorlist)], marker='.', linestyle='None')

                angWG1 = PolPCS[band]
                angRX = PolRX[band]
                for pol in range(npol):

                    """
                    popt,pcov = fitCurvePCS(p_angle,pol_ave[pol],pol_rms[pol],angWG1,angRX[pol])
                    angleWG1f = popt[0]
                    angleRXf  = popt[1]
                    Af        = popt[2]
                    Bf        = popt[3]

                    perr = np.sqrt(np.diag(pcov))
                    angleWG1f_err = perr[0]
                    angleRXf_err  = perr[1]
                    Af_err        = perr[2]
                    Bf_err        = perr[3]

                    ang_num = 360
                    fitangle = np.linspace(0,ang_num,num=ang_num)*(np.max(p_angle)-np.min(p_angle))/ang_num+np.min(p_angle)
                    fitAmp = CurvePCS(fitangle, angleWG1f, angleRXf, Af, Bf)
                    fit_wrt.writerow([ant,spw,pol,angleWG1f,angleWG1f_err,angleRXf,angleRXf_err,Af,Af_err,Bf,Bf_err])

                    axsPol[pol].errorbar(p_angle, pol_ave[pol], yerr=pol_rms[pol], fmt='o', color=colorlist[spwidx])
                    axsPol[pol].plot(fitangle, fitAmp, color=colorlist[spwidx])
                    """

                    phi_WG2 = 0.0
                    theta_RX = angRX[pol]
                    theta_WG1 = angWG1
                    phi_WG1 = 20.0
                    alpha = list_az[antid]

                    popt, pcov = fittingPCS(p_angle, pol_ave[pol], pol_rms[pol], phi_WG2, theta_RX, theta_WG1, phi_WG1, alpha)

                    phi_WG2, theta_RX, theta_WG1, phi_WG1, alpha, P0, Poffset = popt
                    perr = np.sqrt(np.diag(pcov))
                    phi_WG2_err, theta_RX_err, theta_WG1_err, phi_WG1_err, alpha_err, P0_err, Poffset_err = perr

                    ang_num = 360
                    fitangle = np.linspace(0, ang_num, num=ang_num) * (np.max(p_angle) - np.min(p_angle)) / ang_num + np.min(p_angle)
                    fitAmp = responsePCS(fitangle, phi_WG2, theta_RX, theta_WG1, phi_WG1, alpha, P0, Poffset)
                    axsPol[pol].errorbar(p_angle, pol_ave[pol], yerr=pol_rms[pol], fmt='o', color=colorlist[spwidx % len(colorlist)])
                    axsPol[pol].plot(fitangle, fitAmp, color=colorlist[spwidx % len(colorlist)])

                    if ffit is not None:
                        fit_wrt.writerow([ant, spw, pol, phi_WG2, phi_WG2_err, theta_RX, theta_RX_err, P0, P0_err, Poffset, Poffset_err])

                    print(ant, spw, pol, phi_WG2, theta_RX, theta_WG1, phi_WG1, alpha, P0, Poffset)

                    if pol == 0:
                        title = "%s %s Pol-X" % (prefix, ant)
                    elif pol == 1:
                        title = "%s %s Pol-Y" % (prefix, ant)
                    else:
                        title = "%s %s Pol-%d" % (prefix, ant, pol)
                    axsPol[pol].set_title(title)
                    axsPol[pol].set_xlabel("WG2 Polarizer angle [deg]")
                    axsPol[pol].set_ylabel("BB Power [mW]")
                    axsPol[pol].tick_params(axis='both', which='major')
                    axsPol[pol].grid(visible=True, linestyle='dashed')

        figTS.savefig(repdir / f"{prefix}_{ant}_Sqld.png")
        figGS.savefig(repdir / f"{prefix}_{ant}_Sqld_AllanVar.png")
        figTS.clf()
        figGS.clf()

        if polarizer_file not in (None, 'None'):
            figPol.savefig(repdir / f"{prefix}_{ant}_SQLD_Pol.png")
            figPol.clf()

        plt.close('all')

        if time_axis is None:
            continue

        header = ['Time']
        flat_raw_cols: List[np.ndarray] = []
        for spwidx, spw_cols in enumerate(raw_columns):
            for pol_idx, col in enumerate(spw_cols):
                header.append(f"P{pol_idx} BB_{spwidx + 1}")
                flat_raw_cols.append(np.asarray(col, dtype=float))

        raw_matrix = np.column_stack([time_axis] + flat_raw_cols) if flat_raw_cols else time_axis[:, None]

        with (repdir / f"{prefix}_{ant}_SQLD_Raw.csv").open('w', newline='') as fraw:
            raw_wrt = csv.writer(fraw)
            raw_wrt.writerow(header)
            raw_wrt.writerows(raw_matrix)

        valid_diffTs = [diffT for diffT, _ in asd_columns if diffT.size]
        if valid_diffTs:
            min_len = min(len(diffT) for diffT in valid_diffTs)
            diffT_ref = valid_diffTs[0][:min_len]

            var_header = ['Time']
            flat_var_cols: List[np.ndarray] = []
            for spwidx, (diffT, cols) in enumerate(asd_columns):
                for pol_idx, col in enumerate(cols):
                    var_header.append(f"P{pol_idx} BB_{spwidx + 1}")
                    flat_var_cols.append(np.asarray(col[:min_len], dtype=float))

            var_matrix = np.column_stack([diffT_ref] + flat_var_cols)

            with (repdir / f"{prefix}_{ant}_SQLD_AllanVar.csv").open('w', newline='') as fvar:
                var_wrt = csv.writer(fvar)
                var_wrt.writerow(var_header)
                var_wrt.writerows(var_matrix)

    if ffit is not None:
        ffit.close()


#------------------------------------------------------------------------------
def getCWSignal_old(spec):

    nflag = 3
    ndata = len(spec)
    nch   = len(spec[0])

    ch = np.linspace(0, nch-1, num=nch)
    cwCh, noiseSpec, cwSpec  = [], [], []

    tmp_spec = np.copy(spec)
    
    ave_spec = np.nanmean(np.absolute(tmp_spec), axis=0)
    diff_ave = np.diff(ave_spec, n=1)
    cwCh_ave = np.argmax(diff_ave)+1
    #print("debug : ", cwCh_ave)

    for ii in range(ndata):
        diff   = np.diff(np.absolute(tmp_spec[ii]), n=1)
        cw_ch  = np.argmax(diff)+1
        cwCh.append(cw_ch)

        for jj in range(nflag):
            fch = cwCh_ave-int(nflag/2)+jj
            tmp_spec[ii][fch] = np.nan

        ma_data = np.ma.masked_invalid(tmp_spec[ii])
        
        val_x = ch[~ma_data.mask]
        val_y = np.absolute(tmp_spec[ii][~ma_data.mask])
        #f = interp1d(val_x, val_y, kind='cubic')
        f = interp1d(val_x, val_y, kind='quadratic')

        for jj in range(nflag):
            fch = cwCh_ave-int(nflag/2)+jj
            tmp_spec[ii][fch] = f(fch)

        y_smooth = savgol_filter(np.absolute(tmp_spec[ii]), 61, 8, mode='nearest')

        noiseSpec.append(y_smooth)
        cwSpec.append(spec[ii]-y_smooth)

    #return [cwCh, cwSpec, noiseSpec]
    return [cwCh_ave, cwSpec, noiseSpec]
#------------------------------------------------------------------------------
def spectraBaseline(x, a, b):
    return a + b*x

def getCWSignal2(spec):

    ndata = len(spec)
    nch   = len(spec[0])

    ch = np.linspace(0, nch-1, num=nch)
    cwCh, noiseSpec, cwSpec  = [], [], []

    tmp_spec = np.copy(spec)
    
    ave_spec = np.nanmean(np.absolute(tmp_spec), axis=0)
    diff_ave = np.diff(ave_spec, n=1)

    Ch_diffmax = np.argmax(diff_ave)+1
    Ch_diffmin = np.argmin(diff_ave)

    if Ch_diffmax==Ch_diffmin:
        cwCh_ave = np.array([Ch_diffmax])
    elif np.abs(Ch_diffmax-Ch_diffmin)==1:
        cwCh_ave = np.array([Ch_diffmax, Ch_diffmin])
    else:
        print("debug : spike CH error", Ch_diffmax, Ch_diffmin )

    # Baseline substraction
    cwCh_min = np.min(cwCh_ave)
    cwCh_max = np.max(cwCh_ave)

    ch_fit = []
    #for ii in range(-6,-1):
    for ii in range(-10,-5):
        ch_fit.append(cwCh_min+ii)
    #for ii in range(2,7):
    for ii in range(5,10):
        ch_fit.append(cwCh_max+ii)        
    ch_fit = np.array(ch_fit)

    noiseCh_r, noiseCh_l = [], []
    for ii in range(len(cwCh_ave)):
        #noiseCh_r.append(cwCh_min-4-ii)
        #noiseCh_l.append(cwCh_max+4+ii)
        noiseCh_r.append(cwCh_min-5-ii)
        noiseCh_l.append(cwCh_max+5+ii)
    noiseCh_r = np.array(noiseCh_r)
    noiseCh_l = np.array(noiseCh_l)

    cwPower, noisePower, noisePower_r, noisePower_l, rms = [], [], [], [], []
    for ii in range(ndata):
        
        val_x = ch_fit
        val_y = []
        for jj in ch_fit:
            val_y.append(np.absolute(tmp_spec[ii][jj]))
        val_y = np.array(val_y)
        popt,pcov = curve_fit(spectraBaseline, val_x, val_y)   

        tmp_noise  = 0.0
        tmp_cwPower = 0.0
        for jj in cwCh_ave:
            tmp_noise  += spectraBaseline(jj, *popt)
            tmp_cwPower += np.absolute(tmp_spec[ii][jj]) - spectraBaseline(jj, *popt)
        cwPower.append(tmp_cwPower)
        noisePower.append(tmp_noise)
        
        tmp_noise = 0.0
        for jj in noiseCh_r:
            tmp_noise += np.absolute(tmp_spec[ii][jj])
        noisePower_r.append(tmp_noise)

        tmp_noise = 0.0
        for jj in noiseCh_l:
            tmp_noise += np.absolute(tmp_spec[ii][jj])
        noisePower_l.append(tmp_noise)

        tmp_data = []
        for jj in ch_fit:
            tmp_data.append(np.absolute(tmp_spec[ii][jj])- spectraBaseline(jj, *popt))
        tmp_data = np.array(tmp_data)
        rms.append(np.std(tmp_data)*np.sqrt(len(cwCh_ave))) # (rms at 1ch) * sqrt(Nch)
    #
    cwPower = np.array(cwPower)
    noisePower  = np.array(noisePower)
    return [cwCh_ave, noiseCh_r, noiseCh_l, cwPower, noisePower, noisePower_r, noisePower_l, rms]

#------------------------------------------------------------------------------
# Return all products of CW signal
def getCWSignal(spec):

    spec = np.asarray(spec)
    npol, ndata, nch = spec.shape

    ch = np.linspace(0, nch-1, num=nch)
    cwCh, noiseSpec, cwSpec  = [], [], []

    tmp_spec = np.copy(spec)
    
    ave_spec = np.nanmean(np.absolute(tmp_spec[0]), axis=0)
    diff_ave = np.diff(ave_spec, n=1)

    Ch_diffmax = np.argmax(diff_ave)+1
    Ch_diffmin = np.argmin(diff_ave)

    if Ch_diffmax==Ch_diffmin:
        cwCh_ave = np.array([Ch_diffmax])
    elif np.abs(Ch_diffmax-Ch_diffmin)==1:
        cwCh_ave = np.array([Ch_diffmax, Ch_diffmin])
    else:
        print("debug : spike CH error", Ch_diffmax, Ch_diffmin )

    # Vector average of correlation
    cw_corr = spec[:, :, cwCh_ave].sum(axis=2)

    return [cw_corr[pol].tolist() for pol in range(npol)]

#------------------------------------------------------------------------------
def readPolarizerFile(polfile):

    f = open(polfile, 'r')
    pollist = f.readlines()
    f.close()

    p_time, p_angle = [], []

    for pol in pollist:
        time_txt = pol.split(',')[0]
        pol_txt  = pol.split(',')[1].split(' ')
        if(len(pol_txt)==2):
            pol_txt = pol_txt[0]
        else:
            pol_txt = pol_txt[1]

        p_time.append(dt.datetime.strptime(time_txt, '%Y/%m/%d %H:%M:%S.%f'))
        p_angle.append(float(pol_txt))

    p_angle = np.array(p_angle).astype(np.float)
    return [p_time, p_angle]

#------------------------------------------------------------------------------
def convertTimeAngle(time, p_time, p_angle):

    ntime = len(time)
    nstep = len(p_angle)

    pol_idx = []
    if p_time[0]<dt.datetime(2024,1,1):
        # T0              : command a new polarizer angle
        # T0+0.5          : start moving a polarizer
        # T0+0.5+dtheta/7 : stop moving a polarzer, 2 deg/(7 deg/sec)~0.3sec, actually 0.30~0.35 sec 
        for ii in range(nstep):
            tmp_idx = []
            stime = p_time[ii]+dt.timedelta(seconds=1.6) # ocassionally instability was observed between T0+1.0 and T0+1.6
            if ii < nstep-1:
                etime = p_time[ii+1] + dt.timedelta(seconds=0.30)
            elif ii==nstep-1:
                etime = p_time[ii]+dt.timedelta(seconds=10.0)
            for jj in range(ntime):
                if stime <= time[jj] < etime:
                   tmp_idx.append(jj)
            pol_idx.append(tmp_idx)
    elif p_time[0]>=dt.datetime(2024,1,1):
        # for tests in 2024
        # t_sample for SQLD: ~48 msec?
        # T0                   : command a new polarizer angle
        # T0+ 4~5 * t_sample   : start moving a polarizer
        # T0+ 14~16 * t_sample : stop moving a polarzer
        t_offsett0 = 0.75 # glitch between 0.50 and 0.75 ocassionally
        t_offeset1 = 0.10
        for ii in range(nstep):
            tmp_idx = []
            stime = p_time[ii]+dt.timedelta(seconds=t_offsett0)
            if ii < nstep-1:
                etime = p_time[ii+1] + dt.timedelta(seconds=t_offeset1)
            elif ii==nstep-1:
                etime = p_time[ii]+dt.timedelta(seconds=5.0)
            for jj in range(ntime):
                if stime <= time[jj] < etime:
                   tmp_idx.append(jj)
            pol_idx.append(tmp_idx)        
    return pol_idx

#------------------------------------------------------------------------------
def convertTimeAngle_Xpol(time, p_time, p_angle):
    """
    This function is only used when step time is set to 2 seconds in the measurement
    """
    ntime = len(time)
    nstep = len(p_angle)

    pol_idx = []
    if p_time[0]<dt.datetime(2024,1,1):
        # for tests in 2023
        # T0              : command a new polarizer angle
        # T0+0.5          : start moving a polarizer
        # T0+0.5+dtheta/7 : stop moving a polarzer, 2 deg/(7 deg/sec)~0.3sec, actually 0.30~0.35 sec
        t_offset0 = -3.0
        t_offset1 = 0.10
        t_offset2 = 0.75
        for ii in range(nstep):
            tmp_idx = []
            if ii < nstep-1:
                stime = p_time[ii+1] + dt.timedelta(seconds=t_offset0)
                etime = p_time[ii+1] + dt.timedelta(seconds=t_offset1)
            elif ii==nstep-1:
                stime = p_time[ii] + dt.timedelta(seconds=t_offset2)
                etime = p_time[ii] + dt.timedelta(seconds=t_offset1-t_offset0+t_offset2)
            for jj in range(ntime):
                if stime <= time[jj] < etime:
                    tmp_idx.append(jj)
            pol_idx.append(tmp_idx)
    elif p_time[0]>=dt.datetime(2024,1,1):
        # for tests in 2024
        # t_sample for SQLD: ~48 msec?
        # T0                   : command a new polarizer angle
        # T0+ 4~5 * t_sample   : start moving a polarizer
        # T0+ 14~16 * t_sample : stop moving a polarzer, 2 deg/(7 deg/sec)~0.3sec, actually 0.30~0.35 sec
        t_offset0 = -3.3
        t_offset1 = 0.10
        t_offset2 = 0.75
        for ii in range(nstep):
            tmp_idx = []
            if ii < nstep-1:
                stime = p_time[ii+1] + dt.timedelta(seconds=t_offset0)
                etime = p_time[ii+1] + dt.timedelta(seconds=t_offset1)
            elif ii==nstep-1:
                stime = p_time[ii] + dt.timedelta(seconds=t_offset2)
                etime = p_time[ii] + dt.timedelta(seconds=t_offset1-t_offset0+t_offset2)
            for jj in range(ntime):
                if stime <= time[jj] < etime:
                    tmp_idx.append(jj)
            pol_idx.append(tmp_idx)

    return pol_idx

#------------------------------------------------------------------------------
def weightedAveStd(val, weight):
    # Weighted average
    n = len(val)
    wAve = 0.0
    for ii in range(n):
        wAve += val[ii]*weight[ii]
    wAve = wAve/np.sum(weight)
    # Weighted standard deviation
    wVar = 0.0
    for ii in range(n):
        wVar += weight[ii]*np.power((val[ii]-wAve),2.0)
    wVar = wVar/((n-1)*np.sum(weight)/n)
    wStd = np.sqrt(wVar)
    return wAve,wStd

#------------------------------------------------------------------------------
def doPCSAnalysisForACFR(vis, polarizer_file=None, Xpol=False):

    intent = 'CALIBRATE_DELAY#ON_SOURCE'
    repdir = 'report_'+vis.split('.')[0]
    prefix = vis.split('_')[-1][:-3]

    if (os.path.isdir(repdir) != True):
        os.system("mkdir %s" % repdir)

    antlist  = au.getAntennaNames(vis)
    scanlist = getScansForIntent(vis, intent)
    scan = scanlist[0]
    SpwsSQLD, SpwsFR, SpwsCA = getSpwsForScan(vis, scan, intent)

    nant  = len(antlist) 
    nspw = len(SpwsFR)

    spw_color = ['r', 'b', 'g', 'c']

    # Check polarization angle
    if polarizer_file != None and polarizer_file!='None':
        p_time, p_angle = readPolarizerFile(polarizer_file)
        ffit = open('%s/%s_fitting_ACFR.csv' %(repdir, prefix), 'w')
        fit_wrt = csv.writer(ffit)
        fit_wrt.writerow(['AE','SPW','Pol','WG1','WG1_err','RX','RX_err','Amp','Amp_err','Offset','Offset_err'])

        plt.ioff()
        figRxNull, axsRxNull = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
        figRxNull.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90,hspace=0.3,wspace=0.3)
    
    for antidx in range(nant):

        ant = antlist[antidx]
        antid = au.getAntennaIndex(vis,ant)
        print("Antenna: %s [%d/%d]" % (ant, antidx+1, nant))

        figTS, axsTS = plt.subplots(4,2,figsize=[8.27, 11.69],dpi=200)
        figGS, axsGS = plt.subplots(4,2,figsize=[8.27, 11.69],dpi=200)
        figTS.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.95,hspace=0.3,wspace=0.3)
        figGS.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.95,hspace=0.3,wspace=0.3)

        if polarizer_file!=None:
            figPol, axsPol = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
            figPol.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.95,hspace=0.3,wspace=0.3)

            figPolxNull, axsPolxNull = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
            figPolxNull.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90,hspace=0.3,wspace=0.3)

            figPolyNull, axsPolyNull = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
            figPolyNull.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90,hspace=0.3,wspace=0.3)

            figWG1Null, axsWG1Null = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
            figWG1Null.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90,hspace=0.3,wspace=0.3)

            figPolAngle, axsPolAngle = plt.subplots(3,1,figsize=[8.27, 11.69],dpi=200)
            figPolAngle.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90,hspace=0.3,wspace=0.3)

        for spwidx in range(nspw):

            spw  = SpwsFR[spwidx]
            ddid = au.getDataDescriptionId(vis, spw)

            print("SPW    : %d [%d/%d]" % (spw, spwidx+1, nspw))

            freq = getFrequenciesGHz(vis, spw)
            if    84.0 <= freq[0] < 116: band='3'
            elif 211.0 <= freq[0] < 275: band='6'
            elif 275.0 <= freq[0] < 373: band='7'

            time,data,fdata,state = getSpectralAutoData_hack(vis,antid,ddid,scan)
            data  = np.array(data).transpose((0,2,1))
            npol  = len(data)
            ndump = len(data[0])

            T = []
            for ii in range(ndump):
                T.append( (time[ii]-time[0]).total_seconds())

            # XX,YY,XY*,YX* spectra
            if npol==2:
                fig, axs = plt.subplots(1, 2, figsize=[8.27, 11.69],dpi=200)
            elif npol==4:
                fig, axs = plt.subplots(3, 2, figsize=[8.27, 11.69],dpi=200)
            fig.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.85,hspace=0.3,wspace=0.3)

            for pol in range(2):
                for ii in range(ndump):
                    axs[0][pol].plot(freq,np.absolute(data[(npol-pol)%npol][ii]))
                if pol==0:
                    title = "%s %s SPW%s XX" % (prefix,ant,spw)
                elif pol==1:
                    title = "%s %s SPW%s YY" % (prefix,ant,spw)
                axs[0][pol].set_title(title)                  
                axs[0][pol].set_xlabel("Frequency [GHz]")
                axs[0][pol].set_ylabel("Amplitude [a.u.]")
                axs[0][pol].tick_params(axis='both', which='major')
                axs[0][pol].grid(visible=True, linestyle='dashed')

            if npol==4:
                # XY* & YX*
                for pol in range(2):

                    for ii in range(ndump):
                        axs[1][pol].plot(freq,np.absolute(data[pol+1][ii]))
                        axs[2][pol].plot(freq,np.angle(data[pol+1][ii]),marker='.',linestyle='None')

                    if pol==0:
                        title = "%s %s SPW%s XY*" % (prefix,ant,spw)
                    elif pol==1:
                        title = "%s %s SPW%s YX*" % (prefix,ant,spw)
                    #
                    axs[1][pol].set_title(title)
                    axs[1][pol].set_xlabel("Frequency [GHz]")
                    axs[1][pol].set_ylabel("Amplitude [a.u.]")
                    axs[1][pol].tick_params(axis='both', which='major')
                    axs[1][pol].grid(visible=True, linestyle='dashed')
                    #
                    axs[2][pol].set_title(title)
                    axs[2][pol].set_title(title)
                    axs[2][pol].set_xlabel("Frequency [GHz]")
                    axs[2][pol].set_ylabel("Phase [rad]")
                    axs[2][pol].tick_params(axis='both', which='major')
                    axs[2][pol].grid(visible=True, linestyle='dashed')
            
            fig.savefig('%s/%s_%s_SPW%d.png' % (repdir,prefix,ant,spw))
            fig.clf()
            plt.close(fig)

            # CW peak plot
            cwChXX, noiseChXX_R, noiseChXX_L, cwPeakXX, noiseXX, noiseXX_R, noiseXX_L, rmsXX = getCWSignal2(data[0])
            if npol==2:
                cwChYY, noiseChYY_R, noiseChYY_L, cwPeakYY, noiseYY, noiseYY_R, noiseYY_L, rmsYY = getCWSignal2(data[1])
            elif npol==4:
                cwChYY, noiseChYY_R, noiseChYY_L, cwPeakYY, noiseYY, noiseYY_R, noiseYY_L, rmsYY = getCWSignal2(data[3])
                # make time-series XY* phase
                cwXYphase, noiseXYphase_r, noiseXYphase_l = [], [], []
                for ii in range(len(data[1])):
                    tmp_cwXYphase = []
                    for ch in cwChXX:
                        tmp_cwXYphase.append(np.angle(data[1][ii][ch]))
                    cwXYphase.append(np.average(tmp_cwXYphase))
    
                    tmp_noiseCWphase_r, tmp_noiseCWphase_l = [], []
                    for ch in noiseChXX_R:
                        tmp_noiseCWphase_r.append(np.angle(data[1][ii][ch]))
                    noiseXYphase_r.append(np.average(tmp_noiseCWphase_r))

                    for ch in noiseChXX_L:
                        tmp_noiseCWphase_l.append(np.angle(data[1][ii][ch]))
                    noiseXYphase_l.append(np.average(tmp_noiseCWphase_l))

            # Correlation of CW signal
            corrCW = getCWSignal(data)

            cwCh   = [cwChXX, cwChYY]
            cwPeak = [cwPeakXX, cwPeakYY]
            noise  = [noiseXX, noiseYY]
            noise_R = [noiseXX_R, noiseYY_R]
            noise_L = [noiseXX_L, noiseYY_L]
            rms    = [rmsXX, rmsYY]

            for pol in range(2):
                if pol==0:
                    color='b'
                    title = "%s %s Pol-X SPW%d" % (prefix,ant,spw)
                elif pol==1:
                    color='g' 
                    title = "%s %s Pol-Y SPW%d" % (prefix,ant,spw)
                #
                axsTS[spwidx][pol].errorbar(T, cwPeak[pol], yerr=rms[pol], fmt='o',
                                            color=color, ecolor=color, markeredgecolor=color)    
                #
                axsTS[spwidx][pol].set_title(title)
                axsTS[spwidx][pol].set_xlabel("Time [sec]")
                axsTS[spwidx][pol].set_ylabel("CW/comb amplitude [a.u.]")
                axsTS[spwidx][pol].tick_params(axis='both', which='major')
                axsTS[spwidx][pol].grid(visible=True, linestyle='dashed')

            # Gain stability plot
            diffT, ASDg = [], []
            for pol in range(2):
                dtime = T[1] - T[0]
                dG = cwPeak[pol]/np.average(cwPeak[pol])
                tmp_diffT, tmp_ASDg = getASDg(dtime, dG)
                diffT.append(tmp_diffT)
                ASDg.append(tmp_ASDg)

            for pol in range(2):
                if pol==0:
                    title = "%s %s Pol-X SPW%d" % (prefix,ant,spw)
                elif pol==1:
                    title = "%s %s Pol-Y SPW%d" % (prefix,ant,spw)
                    
                axsGS[spwidx][pol].plot(diffT[pol], ASDg[pol], color='b')
                axsGS[spwidx][pol].set_xscale('log')
                axsGS[spwidx][pol].set_yscale('log')
                axsGS[spwidx][pol].set_xlim(1e-2, 1e3)
                axsGS[spwidx][pol].set_ylim(1e-4, 1.0)
                axsGS[spwidx][pol].set_title(title)
                axsGS[spwidx][pol].set_xlabel("Time [sec]")
                axsGS[spwidx][pol].set_ylabel("Allan Standard Deviation ASDg(2,T)")
                axsGS[spwidx][pol].tick_params(axis='both', which='major')
                axsGS[spwidx][pol].grid(visible=True, linestyle='dashed')

            # Phase stability plot
            if npol==4:
                figPhase, axsPhase = plt.subplots(3,1,figsize=[8.27, 11.69],dpi=200)
                figPhase.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.95,hspace=0.3,wspace=0.3)
                #
                #axsPhase[0].plot(T,cwXYphase*180.0/np.pi)
                axsPhase[0].plot(T,cwXYphase)

                #
                title = "%s %s SPW%d XY*" % (prefix,ant,spw)
                axsPhase[0].set_title(title)
                axsPhase[0].set_xlabel("Time [sec]")
                axsPhase[0].set_ylabel("Phase [deg]")
                axsPhase[0].tick_params(axis='both', which='major')
                axsPhase[0].grid(visible=True, linestyle='dashed')

                dtime = T[1] - T[0]
                freqCW = np.average([freq[i] for i in cwChXX])
                print("debug: dtime = %.3f" % dtime)
                print("debug: CW freq = %.3e" % freqCW)
                Tk, yk = [], []
                for k in range(len(cwXYphase)-1):
                    Tk.append(T[k])
                    tmp = (cwXYphase[k+1] - cwXYphase[k])/(2.0*np.pi*freqCW*1e9*dtime)
                    yk.append(tmp)

                diffT, ASDy = getASDg(dtime, yk)

                axsPhase[1].plot(Tk, yk)
                axsPhase[1].set_xlabel("Time [sec]")
                axsPhase[1].set_ylabel("Fractional frequency deviation")
                axsPhase[1].tick_params(axis='both', which='major')
                axsPhase[1].grid(visible=True, linestyle='dashed')

                axsPhase[2].plot(diffT, ASDy)
                axsPhase[2].set_xscale('log')
                axsPhase[2].set_yscale('log')
                axsPhase[2].set_xlabel("Time [sec]")
                axsPhase[2].set_ylabel("ASDy(2,T, tau=T)")
                axsPhase[2].tick_params(axis='both', which='major')
                axsPhase[2].grid(visible=True, linestyle='dashed')

                figPhase.savefig('%s/%s_%s_SPW%d_XYphase.png' % (repdir,prefix,ant,spw))
                figPhase.clf()
                plt.close(figPhase)

            # polarizer rotaion
            if polarizer_file != None and polarizer_file != 'None' and Xpol==False:

                colorlist = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
                pol_idx = convertTimeAngle(time, p_time, p_angle)

                pol_ave, pol_rms = [], []
                for pol in range(2):

                    tmp_ave, tmp_rms = [], []
                    for ii in range(len(p_angle)):
                        tmp_peak, tmp_weight = [], []
                        for jj in range(len(pol_idx[ii])):
                            tmp_peak.append(cwPeak[pol][pol_idx[ii][jj]])
                            tmp_weight.append(np.power(rms[pol][pol_idx[ii][jj]],-2.0))

                        wAve, wStd = weightedAveStd(tmp_peak, tmp_weight)
                        tmp_ave.append(wAve)
                        tmp_rms.append(wStd)
                                
                    pol_ave.append(tmp_ave)
                    pol_rms.append(tmp_rms)
                
                angWG1 = PolPCS[band]
                angRX  = PolRX[band]

                for pol in range(2):

                    # fitting
                    popt,pcov = fitCurvePCS(p_angle,pol_ave[pol],pol_rms[pol],angWG1,angRX[pol])
                    angleWG1f = popt[0]
                    angleRXf  = popt[1]
                    Af        = popt[2]
                    Bf        = popt[3]

                    perr = np.sqrt(np.diag(pcov))
                    angleWG1f_err = perr[0]
                    angleRXf_err  = perr[1]
                    Af_err        = perr[2]
                    Bf_err        = perr[3]
                    
                    ang_num = 360
                    fitangle = np.linspace(0,ang_num,num=ang_num)*(np.max(p_angle)-np.min(p_angle))/ang_num+np.min(p_angle)
                    fitAmp = CurvePCS(fitangle, angleWG1f, angleRXf, Af, Bf)
                    fit_wrt.writerow([ant,spw,pol,angleWG1f,angleWG1f_err,angleRXf,angleRXf_err,Af,Af_err,Bf,Bf_err])
                    
                    axsPol[pol].errorbar(p_angle, pol_ave[pol], yerr=pol_rms[pol], fmt='o', color=colorlist[spwidx])
                    axsPol[pol].plot(fitangle, fitAmp, color=colorlist[spwidx])
                    
                    if pol==0:
                        title = "%s %s Pol-X" % (prefix,ant)
                    elif pol==1:
                        title = "%s %s Pol-Y" % (prefix,ant)
                    axsPol[pol].set_title(title)
                    axsPol[pol].set_xlabel("WG2 Polarizer angle [deg]")
                    axsPol[pol].set_ylabel("CW/comb amplitude [a.u.]")
                    axsPol[pol].tick_params(axis='both', which='major')
                    axsPol[pol].grid(visible=True, linestyle='dashed')
                
                # Polarzer Rotation for all correlation products
                pol_cw = []
                for pol in range(npol):
                    tmp_pol = []
                    for ii in range(len(p_angle)):
                        tmp = []
                        for jj in range(len(pol_idx[ii])):
                            tmp.append(corrCW[pol][pol_idx[ii][jj]])

                        tmp_pol.append(np.average(tmp))
                    pol_cw.append(tmp_pol)

                if npol==2:
                    axsPolxNull[0].plot(p_angle, np.real(pol_cw[0]), marker='.', color=spw_color[spwidx], label='SPW%d-XX'%spw)

                    axsPolyNull[0].plot(p_angle, np.real(pol_cw[1]), marker='.', color=spw_color[spwidx], label='SPW%d-YY'%spw, linestyle='dashed')

                    axsWG1Null[0].plot(p_angle, np.real(pol_cw[0]), marker='.', color=spw_color[spwidx], label='SPW%d-XX'%spw)
                    axsWG1Null[0].plot(p_angle, np.real(pol_cw[1]), marker='.', color=spw_color[spwidx], label='SPW%d-YY'%spw, linestyle='dashed')

                elif npol==4:
                    axsPolxNull[0].plot(p_angle, np.real(pol_cw[0]), marker='.', color=spw_color[spwidx], label='SPW%d'%spw)
                    axsPolxNull[1].plot(p_angle, np.absolute(pol_cw[1]), marker='.', color=spw_color[spwidx], label='SPW%d'%spw)

                    axsPolyNull[0].plot(p_angle, np.real(pol_cw[3]), marker='.', color=spw_color[spwidx], label='SPW%d'%spw, linestyle='dashed')
                    axsPolyNull[1].plot(p_angle, np.absolute(pol_cw[1]), marker='.', color=spw_color[spwidx], label='SPW%d'%spw)

                    axsWG1Null[0].plot(p_angle, np.real(pol_cw[0]), marker='.', color=spw_color[spwidx], label='SPW%d-XX'%spw)
                    axsWG1Null[0].plot(p_angle, np.real(pol_cw[3]), marker='.', color=spw_color[spwidx], label='SPW%d-YY'%spw, linestyle='dashed')
                    axsWG1Null[1].plot(p_angle, np.absolute(pol_cw[1]), marker='.', color=spw_color[spwidx], label='SPW%d'%spw)

                # Estimate polarized angle by amplitude
                if npol==2:
                    XX, YY, gain_diff = AutoCorrGainCorrectionPolarizerRotaion(p_angle, angRX, pol_cw[0], pol_cw[1])
                    angle_amp = measurePolarizedAngle(p_angle, angRX[0], XX, YY)
                elif npol==4:
                    XX, YY, gain_diff = AutoCorrGainCorrectionPolarizerRotaion(p_angle, angRX, pol_cw[0], pol_cw[3])
                    angle_amp = measurePolarizedAngle(p_angle, angRX[0], XX, YY)
                axsPolAngle[0].plot(p_angle, XX, marker='.', color=spw_color[spwidx], label='SPW%d-XX'%spw)
                axsPolAngle[0].plot(p_angle, YY, marker='.', color=spw_color[spwidx], label='SPW%d-YY'%spw, linestyle='dashed')
                axsPolAngle[1].plot(p_angle, angle_amp, marker='.', color=spw_color[spwidx], label='SPW%d'%spw)
                axsPolAngle[2].plot(p_angle, angle_amp-p_angle, marker='.', color=spw_color[spwidx], label='SPW%d'%spw)

                # gain correction
                pol_ave[1] = pol_ave[1]/gain_diff
                pol_rms[1] = pol_rms[1]/gain_diff

                if spwidx==0:
                    axsRxNull[0].errorbar(p_angle, pol_ave[0], yerr=pol_rms[0], fmt='o', linestyle='dashed', label=ant)
                    axsRxNull[1].errorbar(p_angle, pol_ave[1], yerr=pol_rms[1], fmt='o', linestyle='dashed', label=ant)

        if polarizer_file != None and polarizer_file != 'None':
            title = "%s %s" % (prefix,ant)
            # Pol-X Null -> around Pol-Y angle
            axsPolxNull[0].set_title(title)
            axsPolxNull[0].set_xlabel("WG2 Polarizer angle [deg]")
            axsPolxNull[0].set_ylabel("Amplitute [a.u.]")
            axsPolxNull[0].tick_params(axis='both', which='major')
            axsPolxNull[0].grid(visible=True, linestyle='dashed')
            axsPolxNull[0].legend()

            axsPolxNull[1].set_xlabel("WG2 Polarizer angle [deg]")
            axsPolxNull[1].set_ylabel("CC Amplitute [a.u.]")
            axsPolxNull[1].tick_params(axis='both', which='major')
            axsPolxNull[1].grid(visible=True, linestyle='dashed')
            axsPolxNull[1].legend()

            xmin = PolRX[band][1]-10.0
            xmax = PolRX[band][1]+10.0
            idx = np.where( (p_angle>=xmin) & (p_angle<=xmax))[0]
            ymin = 0.7*np.real(pol_cw[0])[idx].min()
            ymax = 1.1*np.real(pol_cw[0])[idx].max()
            axsPolxNull[0].set_xlim(xmin, xmax)
            axsPolxNull[0].set_ylim(ymin, ymax)
            ymin = 0.0
            ymax = 1.1*np.absolute(pol_cw[1])[idx].max()
            axsPolxNull[1].set_xlim(xmin, xmax)
            axsPolxNull[1].set_ylim(ymin, ymax)

            # Pol-Y Null -> around Pol-X angle
            axsPolyNull[0].set_title(title)
            axsPolyNull[0].set_xlabel("WG2 Polarizer angle [deg]")
            axsPolyNull[0].set_ylabel("Amplitute [a.u.]")
            axsPolyNull[0].tick_params(axis='both', which='major')
            axsPolyNull[0].grid(visible=True, linestyle='dashed')
            axsPolyNull[0].legend()

            axsPolyNull[1].set_xlabel("WG2 Polarizer angle [deg]")
            axsPolyNull[1].set_ylabel("CC Amplitute [a.u.]")
            axsPolyNull[1].tick_params(axis='both', which='major')
            axsPolyNull[1].grid(visible=True, linestyle='dashed')
            axsPolyNull[1].legend()

            xmin = PolRX[band][0]-10.0
            xmax = PolRX[band][0]+10.0
            idx = np.where( (p_angle>=xmin) & (p_angle<=xmax))[0]
            if npol==2:
                ymin = 0.7*np.real(pol_cw[1])[idx].min()
                ymax = 1.1*np.real(pol_cw[1])[idx].max()
            elif npol==4:
                ymin = 0.7*np.real(pol_cw[3])[idx].min()
                ymax = 1.1*np.real(pol_cw[3])[idx].max()
            axsPolyNull[0].set_xlim(xmin, xmax)
            axsPolyNull[0].set_ylim(ymin, ymax)
            ymin = 0.0
            ymax = 1.2*np.absolute(pol_cw[1])[idx].max()
            axsPolyNull[1].set_xlim(xmin, xmax)
            axsPolyNull[1].set_ylim(ymin, ymax)

            # WG1 Null -> WG1 +- 90deg
            axsWG1Null[0].set_title(title)
            axsWG1Null[0].set_xlabel("WG2 Polarizer angle [deg]")
            axsWG1Null[0].set_ylabel("Amplitute [a.u.]")
            axsWG1Null[0].tick_params(axis='both', which='major')
            axsWG1Null[0].grid(visible=True, linestyle='dashed')
            axsWG1Null[0].legend()

            axsWG1Null[1].set_xlabel("WG2 Polarizer angle [deg]")
            axsWG1Null[1].set_ylabel("CC Amplitute [a.u.]")
            axsWG1Null[1].tick_params(axis='both', which='major')
            axsWG1Null[1].grid(visible=True, linestyle='dashed')
            axsWG1Null[1].legend()

            if band=='3':
                xmin = -90+PolPCS[band]-10.0
                xmax = -90+PolPCS[band]+10.0
            else:
                xmin = 90+PolPCS[band]-10.0
                xmax = 90+PolPCS[band]+10.0
            idx = np.where( (p_angle>=xmin) & (p_angle<=xmax))[0]
            ymin = 0.7*np.real(pol_cw[0])[idx].min()
            ymax = 1.1*np.real(pol_cw[0])[idx].max()
            axsWG1Null[0].set_xlim(xmin, xmax)
            axsWG1Null[0].set_ylim(ymin, ymax)
            ymin = 0.0
            ymax = 1.1*np.absolute(pol_cw[1])[idx].max()
            axsWG1Null[1].set_xlim(xmin, xmax)
            axsWG1Null[1].set_ylim(ymin, ymax)

            # Polarized angle estimated by amplitude
            axsPolAngle[0].set_title(title)

            axsPolAngle[0].set_xlabel("WG2 Polarizer angle [deg]")
            axsPolAngle[0].set_ylabel("Normalized amplitude [a.u.]")
            axsPolAngle[0].tick_params(axis='both', which='major')
            axsPolAngle[0].grid(visible=True, linestyle='dashed')
            axsPolAngle[0].legend()

            axsPolAngle[1].set_xlabel("WG2 Polarizer angle [deg]")
            axsPolAngle[1].set_ylabel("Estimated angle [deg]")
            axsPolAngle[1].tick_params(axis='both', which='major')
            axsPolAngle[1].grid(visible=True, linestyle='dashed')
            axsPolAngle[1].legend()

            axsPolAngle[2].set_xlabel("WG2 Polarizer angle [deg]")
            axsPolAngle[2].set_ylabel("Difference angle [deg]")
            axsPolAngle[2].tick_params(axis='both', which='major')
            axsPolAngle[2].grid(visible=True, linestyle='dashed')
            axsPolAngle[2].legend()

            axsPolAngle[0].set_xlim(-100.0, 100.0)
            axsPolAngle[1].set_xlim(-100.0, 100.0)
            axsPolAngle[1].set_ylim(-100.0, 100.0)
            axsPolAngle[2].set_xlim(-100, 100)
            axsPolAngle[2].set_ylim(-10, 10)
        #
        figTS.savefig('%s/%s_%s_CW.png' % (repdir,prefix,ant))
        figGS.savefig('%s/%s_%s_CW_AllanVar.png' % (repdir,prefix,ant))

        figTS.clf()
        figGS.clf()
        plt.close(figTS)
        plt.close(figGS)

        if polarizer_file!=None:
            figPol.savefig('%s/%s_%s_CW_Pol.png' % (repdir,prefix,ant))
            figPol.clf()
            plt.close(figPol)

            figPolxNull.savefig('%s/%s_%s_CW_PolX_Null.png' % (repdir,prefix,ant))
            figPolxNull.clf()
            plt.close(figPolxNull)

            figPolyNull.savefig('%s/%s_%s_CW_PolY_Null.png' % (repdir,prefix,ant))
            figPolyNull.clf()
            plt.close(figPolyNull)

            figWG1Null.savefig('%s/%s_%s_CW_WG1_Null.png' % (repdir,prefix,ant))
            figWG1Null.clf()
            plt.close(figWG1Null)

            figPolAngle.savefig('%s/%s_%s_CW_PolAngle.png' % (repdir,prefix,ant))
            figPolAngle.clf()
            plt.close(figPolAngle)

        # Write results in csv file - pending


    if polarizer_file != None and polarizer_file != 'None':
        title = "%s" % (prefix)
        # Pol-X Null -> around Pol-Y angle
        axsRxNull[0].set_title(title)
        axsRxNull[0].set_xlabel("WG2 Polarizer angle [deg]")
        axsRxNull[0].set_ylabel("Pol-X Amplitute [a.u.]")
        axsRxNull[0].tick_params(axis='both', which='major')
        axsRxNull[0].grid(visible=True, linestyle='dashed')
        axsRxNull[0].legend()

        xmin = PolRX[band][1]-10.0
        xmax = PolRX[band][1]+10.0
        axsRxNull[0].set_xlim(xmin, xmax)
        axsRxNull[0].set_ylim(-0.5, 3.0)

        # Pol-Y Null -> around Pol-X angle
        axsRxNull[1].set_title(title)
        axsRxNull[1].set_xlabel("WG2 Polarizer angle [deg]")
        axsRxNull[1].set_ylabel("Pol-Y Amplitute [a.u.]")
        axsRxNull[1].tick_params(axis='both', which='major')
        axsRxNull[1].grid(visible=True, linestyle='dashed')
        axsRxNull[1].legend()

        xmin = PolRX[band][0]-10.0
        xmax = PolRX[band][0]+10.0
        axsRxNull[1].set_xlim(xmin, xmax)
        axsRxNull[1].set_ylim(-0.5, 3.0)

        figRxNull.savefig('%s/%s_RX_Null.png' % (repdir,prefix))
        figRxNull.clf()
        plt.close(figRxNull)

        ffit.close()
    
    plt.close('all')

#------------------------------------------------------------------------------
def doPCSAnalysisForACCA(vis, polarizer_file=None, Xpol=False):

    intent = 'CALIBRATE_DELAY#ON_SOURCE'
    repdir = 'report_'+vis.split('.')[0]
    prefix = vis.split('_')[-1][:-3]

    if (os.path.isdir(repdir) != True):
        os.system("mkdir %s" % repdir)

    antlist  = au.getAntennaNames(vis)
    scanlist = getScansForIntent(vis, intent)
    scan = scanlist[0]
    SpwsSQLD, SpwsFR, SpwsCA = getSpwsForScan(vis, scan, intent)

    nant  = len(antlist) 
    nspw = len(SpwsCA)

    # Check polarization angle
    if polarizer_file != None and polarizer_file!='None':
        p_time, p_angle = readPolarizerFile(polarizer_file)
        ffit = open('%s/%s_fitting_ACCA.csv' %(repdir, prefix), 'w')
        fit_wrt = csv.writer(ffit)
        fit_wrt.writerow(['AE','SPW','Pol','WG1','WG1_err','RX','RX_err','Amp','Amp_err','Offset','Offset_err'])

    for antidx in range(nant):
        
        ant = antlist[antidx]
        antid = au.getAntennaIndex(vis,ant)
        print("Antenna: %s [%d/%d]" % (ant, antidx+1, nant))

        plt.ioff()
        figTS, axsTS = plt.subplots(4, 2, figsize=[8.27,11.69],dpi=200)
        figTS.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.95,hspace=0.3,wspace=0.3)

        figGS, axsGS = plt.subplots(4, 2, figsize=[8.27,11.69],dpi=200)
        figGS.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.95,hspace=0.3,wspace=0.3)

        rawdata, vardata = [], []

        if polarizer_file!=None:
            figPol, axsPol = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
            figPol.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.95,hspace=0.3,wspace=0.3)

        for spwidx in range(len(SpwsCA)):

            spw  = SpwsCA[spwidx]
            ddid = au.getDataDescriptionId(vis, spw)

            print("SPW    : %d [%d/%d]" % (spw, spwidx+1, nspw))

            freq = getFrequenciesGHz(vis, spw)
            if    84.0 <= freq[0] < 116: band='3'
            elif 211.0 <= freq[0] < 275: band='6'
            elif 275.0 <= freq[0] < 373: band='7'

            time,data,fdata,state = getSpectralAutoData_hack(vis,antid,ddid,scan)
            npol  = len(data)
            nch   = len(data[0])
            ndump = len(data[0][0])
            print("npol, nch, ndump:", npol, nch, ndump)

            T = []
            for ii in range(ndump):
                T.append( (time[ii]-time[0]).total_seconds())

            rawdata.append(data)
            tmp_ASDg = []

            # XX & YY channel-averaged AC
            for pol in range(2):
                
                dtime = T[1] - T[0]
                dG = np.real(data[(npol-pol)%npol][0])/np.average(np.real(data[(npol-pol)%npol][0]))
                diffT, ASDg = getASDg(dtime, dG)
                tmp_ASDg.append(ASDg)
                
                if pol==0:
                    color = 'b'
                    title = "%s %s Pol-X SPW%d Scan%d" % (prefix,ant,spw,scan)
                elif pol==1:
                    color = 'g'
                    title = "%s %s Pol-Y SPW%d Scan%d" % (prefix,ant,spw,scan)

                # Time series plot
                axsTS[spwidx][pol].plot(T,np.real(data[(npol-pol)%npol][0]), color=color)
                axsTS[spwidx][pol].set_title(title)
                axsTS[spwidx][pol].set_xlabel("Time [UTC]")
                axsTS[spwidx][pol].set_ylabel("AC power [a.u.]")
                axsTS[spwidx][pol].tick_params(axis='both', which='major')
                axsTS[spwidx][pol].grid(visible=True, linestyle='dashed')

                # Gain stability plot
                axsGS[spwidx][pol].plot(diffT, ASDg, color=color)
                axsGS[spwidx][pol].set_xscale('log')
                axsGS[spwidx][pol].set_yscale('log')
                axsGS[spwidx][pol].set_xlim(1e-2, 1e3)
                axsGS[spwidx][pol].set_ylim(1e-8, 1)
                axsGS[spwidx][pol].set_title(title)
                axsGS[spwidx][pol].set_xlabel("Time [sec]")
                axsGS[spwidx][pol].set_ylabel("Allan variance")
                axsGS[spwidx][pol].tick_params(axis='both', which='major')
                axsGS[spwidx][pol].grid(visible=True, linestyle='dashed')
            
            vardata.append(tmp_ASDg)

            # polarizer rotaion
            if polarizer_file != None and polarizer_file!='None':

                colorlist = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
                pol_idx = convertTimeAngle(time, p_time, p_angle)

                pol_ave, pol_rms = [], []
                for pol in range(npol):

                    tmp_ave, tmp_rms = [], []
                    for ii in range(len(p_angle)):
                        tmp_signal = []
                        for jj in range(len(pol_idx[ii])):
                            tmp_signal.append(data[(npol-pol)%npol][0][pol_idx[ii][jj]])

                        wAve = np.mean(np.absolute(tmp_signal))
                        wStd = np.std(np.absolute(tmp_signal))
                        tmp_ave.append(wAve*1e3)
                        tmp_rms.append(wStd*1e3)
                                
                    pol_ave.append(tmp_ave)
                    pol_rms.append(tmp_rms)
                
                angWG1 = PolPCS[band]
                angRX  = PolRX[band]

                #for pol in range(npol):
                for pol in range(2):

                    # fitting
                    popt,pcov = fitCurvePCS(p_angle,pol_ave[pol],pol_rms[pol],angWG1,angRX[pol])
                    angleWG1f = popt[0]
                    angleRXf  = popt[1]
                    Af        = popt[2]
                    Bf        = popt[3]

                    perr = np.sqrt(np.diag(pcov))
                    angleWG1f_err = perr[0]
                    angleRXf_err  = perr[1]
                    Af_err        = perr[2]
                    Bf_err        = perr[3]
                    
                    ang_num = 360
                    fitangle = np.linspace(0,ang_num,num=ang_num)*(np.max(p_angle)-np.min(p_angle))/ang_num+np.min(p_angle)
                    fitAmp = CurvePCS(fitangle, angleWG1f, angleRXf, Af, Bf)
                    fit_wrt.writerow([ant,spw,pol,angleWG1f,angleWG1f_err,angleRXf,angleRXf_err,Af,Af_err,Bf,Bf_err])
                    
                    axsPol[pol].errorbar(p_angle, pol_ave[pol], yerr=pol_rms[pol], fmt='o', color=colorlist[spwidx])
                    axsPol[pol].plot(fitangle, fitAmp, color=colorlist[spwidx])
                    
                    if pol==0:
                        title = "%s %s Pol-X" % (prefix,ant)
                    elif pol==1:
                        title = "%s %s Pol-Y" % (prefix,ant)
                    axsPol[pol].set_title(title)
                    axsPol[pol].set_xlabel("WG2 Polarizer angle [deg]")
                    axsPol[pol].set_ylabel("AC Power [a.u.]")
                    axsPol[pol].tick_params(axis='both', which='major')
                    axsPol[pol].grid(visible=True, linestyle='dashed')

        figTS.savefig('%s/%s_%s_ACCA.png' % (repdir,prefix,ant))
        figGS.savefig('%s/%s_%s_ACCA_AllanVar.png' % (repdir,prefix,ant))
        figTS.clf()
        figGS.clf()

        if polarizer_file!=None:
            figPol.savefig('%s/%s_%s_ACCA_Pol.png' % (repdir,prefix,ant))
            figPol.clf()

        plt.close('all')

#------------------------------------------------------------------------------
def plotAllSpectra(vis):

    intent = 'CALIBRATE_DELAY#ON_SOURCE'
    repdir = 'report_'+vis.split('.')[0]
    prefix = vis.split('_')[-1][:-3]

    if (os.path.isdir(repdir) != True):
        os.system("mkdir %s" % repdir)

    antlist  = au.getAntennaNames(vis)
    scanlist = getScansForIntent(vis, intent)
    scan = scanlist[0]
    SpwsSQLD, SpwsFR, SpwsCA = getSpwsForScan(vis, scan, intent)

    nant  = len(antlist) 
    nspw = len(SpwsFR)

    antidlist = []
    for ant in antlist:
        antid = au.getAntennaIndex(vis,ant)
        antidlist.append(antid)
    #
    cclist = list(itertools.combinations(antidlist,2))

    for spwidx in range(nspw):
        
        spw  = SpwsFR[spwidx]
        ddid = au.getDataDescriptionId(vis, spw)
        freq = getFrequenciesGHz(vis, spw)

        time0,data0,fdata0,state0 = getSpectralAutoData_hack(vis,0,ddid,scan)
        npol = len(data0)

        fig, axs =[], []
        plt.ioff()
        for pol in range(npol):
            tmp_fig, tmp_axs = plt.subplots(nant,nant,figsize=[12.8,12.8],dpi=200)
            tmp_fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,hspace=0.0,wspace=0.0)
            fig.append(tmp_fig)
            axs.append(tmp_axs)

        for pol in range(npol):
            for iant1 in range(nant):
                for iant2 in range(nant):
                    # for amplitude plots
                    if( iant1!=nant-1 and iant2==0 ):
                        axs[pol][iant1][iant2].tick_params(labelbottom=False,labelleft=True)
                        if(iant1==0):
                            axs[pol][iant1][iant2].set_ylabel("Amp")
                        else:
                            axs[pol][iant1][iant2].set_ylabel("Phase")
                    elif( iant1==nant-1 and iant2==0 ):
                        axs[pol][iant1][iant2].tick_params(labelbottom=True,labelleft=True)
                        axs[pol][iant1][iant2].set_ylabel("Phase")
                    elif( iant1==nant-1 and iant2!=0 and iant2!=nant-1):
                        axs[pol][iant1][iant2].tick_params(labelbottom=True,labelleft=False)
                        axs[pol][iant1][iant2].set_ylabel("Freq")
                    elif( iant1==nant-1 and iant2==nant-1 ):
                        axs[pol][iant1][iant2].tick_params(labelbottom=True,labelleft=False,labelright=True)
                        axs[pol][iant1][iant2].set_xlabel("Freq")
                        axs[pol][iant1][iant2].yaxis.set_label_position('right')
                        axs[pol][iant1][iant2].set_ylabel("Amp")
                    elif( iant1!=nant-1 and iant2==nant-1 ):
                        axs[pol][iant1][iant2].tick_params(labelbottom=False,labelleft=False,labelright=True)
                        axs[pol][iant1][iant2].yaxis.set_label_position('right')
                        axs[pol][iant1][iant2].set_ylabel("Amp")
                    else:
                        axs[pol][iant1][iant2].tick_params(labelbottom=False,labelleft=False)
                       
        #AC
        for antid in antidlist:
            time,data,fdata,state = getSpectralAutoData_hack(vis,antid,ddid,scan)
            data = np.array(data).transpose((0,2,1))
            data = np.average(data, axis=1)
            for pol in range(npol):
                axs[pol][antid][antid].plot(freq,np.absolute(data[pol]))
                ID = '%s*%s' % (antid,antid)
                axs[pol][antid][antid].text(0.99, 0.99, ID, va='top', ha='right', transform=axs[pol][antid][antid].transAxes)    

        #CC
        for cc in cclist:
            iant1,iant2 = cc
            time,data = getSpectralData_hack(vis,iant1,iant2,ddid,scan)
            data = np.array(data).transpose((0,2,1))
            data = np.average(data, axis=1)
            for pol in range(npol):
                axs[pol][iant1][iant2].plot(freq,np.absolute(data[pol]))
                axs[pol][iant2][iant1].plot(freq,np.angle(data[pol]),marker='.',linestyle='None')
                ID = '%s*%s' % (iant1,iant2)
                axs[pol][iant1][iant2].text(0.99, 0.99, ID, va='top', ha='right', transform=axs[pol][iant1][iant2].transAxes)
                axs[pol][iant2][iant1].text(0.99, 0.99, ID, va='top', ha='right', transform=axs[pol][iant2][iant1].transAxes)
                #
                axs[pol][iant2][iant1].set_ylim(-np.pi,np.pi)

        for pol in range(npol):
            fig[pol].savefig('%s/%s_SPW%d_%d.png' % (repdir,prefix,spw,pol))
            fig[pol].clf()
        plt.close('all')    

#------------------------------------------------------------------------------
def AutoCorrGainCorrectionPolarizerRotaion(p_angle, RX_pol, Vxx, Vyy):

    XX = np.real(Vxx).copy()
    YY = np.real(Vyy).copy()

    # search +45 deg with respect to RX
    mid_angle = (RX_pol[0]+RX_pol[1])/2.0
    if mid_angle<-90.0:
        mid_angle += 180.0
    elif mid_angle>90.0:
        mid_angle -= -180.0

    for ii in range(len(p_angle)-1):
        if p_angle[ii]==mid_angle:
            alpha = YY[ii]/XX[ii]
        else:
            if (p_angle[ii]-mid_angle)*(p_angle[ii+1]-mid_angle)<0.0:
                # simple interpolaration
                tmp_XX = (XX[ii+1]-XX[ii])/(p_angle[ii+1]-p_angle[ii])*(mid_angle-p_angle[ii])+XX[ii]
                tmp_YY = (YY[ii+1]-YY[ii])/(p_angle[ii+1]-p_angle[ii])*(mid_angle-p_angle[ii])+YY[ii]
                alpha = tmp_YY/tmp_XX

    # gain correction
    #print("YY/XX gain diff.: ", alpha)
    YY = YY /alpha

    return XX, YY, alpha

def measurePolarizedAngle(p_angle, polRX_x, Vxx, Vyy):
    # 
    tmp_Vxx = Vxx - np.min(Vxx)
    tmp_Vyy = Vyy - np.min(Vyy)

    angle_amp = np.arctan2(np.sqrt(tmp_Vyy), np.sqrt(tmp_Vxx))
    angle_amp = np.rad2deg(angle_amp)

    mid_angle = polRX_x + 45.0
    for ii in range(len(p_angle)-1):
        if p_angle[ii]<=mid_angle and p_angle[ii+1]>mid_angle:
            ref_idx = ii
            break

    # decrement
    for ii in range(ref_idx, 0, -1):
        if angle_amp[ii]<angle_amp[ii-1]:
            p0_idx = ii-1
            break
    # increment
    for ii in range(ref_idx, len(p_angle), 1):
        if angle_amp[ii]>angle_amp[ii+1]:
            p1_idx = ii+1
            break

    for ii in range(len(angle_amp)):
        if ii<=p0_idx:
            angle_amp[ii] = -angle_amp[ii]
        elif ii>=p1_idx:
            angle_amp[ii] = 180.0 - angle_amp[ii]

    print(ref_idx, p0_idx, p1_idx)
    # align to the PCS system coordinate
    angle_amp = angle_amp + polRX_x

    return angle_amp


def doPCSAnalysisForXpol(vis, polarizer_file=None):

    intent = 'CALIBRATE_DELAY#ON_SOURCE'
    repdir = 'report_'+vis.split('.')[0]
    prefix = vis.split('_')[-1][:-3]

    if (os.path.isdir(repdir) != True):
        os.system("mkdir %s" % repdir)

    antlist  = au.getAntennaNames(vis)
    scanlist = getScansForIntent(vis, intent)
    scan = scanlist[0]
    SpwsSQLD, SpwsFR, SpwsCA = getSpwsForScan(vis, scan, intent)

    nant = len(antlist) 
    nspw = len(SpwsFR)

    spw_color = ['r', 'b', 'g', 'c']

    # Check polarization angle
    p_time, p_angle = readPolarizerFile(polarizer_file)

    plt.ioff()
    figRxNull, axsRxNull = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
    figRxNull.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90,hspace=0.3,wspace=0.3)
    
    for antidx in range(nant):

        ant = antlist[antidx]
        antid = au.getAntennaIndex(vis,ant)
        print("Antenna: %s [%d/%d]" % (ant, antidx+1, nant))

        figPol, axsPol = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
        figPol.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.95,hspace=0.3,wspace=0.3)

        figPolxNull, axsPolxNull = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
        figPolxNull.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90,hspace=0.3,wspace=0.3)

        figPolyNull, axsPolyNull = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
        figPolyNull.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90,hspace=0.3,wspace=0.3)

        figWG1Null, axsWG1Null = plt.subplots(2,1,figsize=[8.27, 11.69],dpi=200)
        figWG1Null.subplots_adjust(left=0.10,right=0.90,bottom=0.10,top=0.90,hspace=0.3,wspace=0.3)

        for spwidx in range(nspw):

            spw  = SpwsFR[spwidx]
            ddid = au.getDataDescriptionId(vis, spw)

            print("SPW    : %d [%d/%d]" % (spw, spwidx+1, nspw))

            freq = getFrequenciesGHz(vis, spw)
            if    84.0 <= freq[0] < 116: band='3'
            elif 211.0 <= freq[0] < 275: band='6'
            elif 275.0 <= freq[0] < 373: band='7'

            time,data,fdata,state = getSpectralAutoData_hack(vis,antid,ddid,scan)
            data  = np.array(data).transpose((0,2,1))
            npol  = len(data)
            ndump = len(data[0])

            dT = []
            for ii in range(ndump):
                dT.append( (time[ii]-time[0]).total_seconds())

            # XX,YY,XY*,YX* spectra
            if npol==2:
                fig, axs = plt.subplots(1, 2, figsize=[8.27, 11.69],dpi=200)
            elif npol==4:
                fig, axs = plt.subplots(3, 2, figsize=[8.27, 11.69],dpi=200)
            fig.subplots_adjust(left=0.10,right=0.95,bottom=0.05,top=0.85,hspace=0.3,wspace=0.3)

            for pol in range(2):
                for ii in range(ndump):
                    axs[0][pol].plot(freq,np.absolute(data[(npol-pol)%npol][ii]))
                if pol==0:
                    title = "%s %s SPW%s XX" % (prefix,ant,spw)
                elif pol==1:
                    title = "%s %s SPW%s YY" % (prefix,ant,spw)
                axs[0][pol].set_title(title)                  
                axs[0][pol].set_xlabel("Frequency [GHz]")
                axs[0][pol].set_ylabel("Amplitude [a.u.]")
                axs[0][pol].tick_params(axis='both', which='major')
                axs[0][pol].grid(visible=True, linestyle='dashed')

            if npol==4:
                # XY* & YX*
                for pol in range(2):

                    for ii in range(ndump):
                        axs[1][pol].plot(freq,np.absolute(data[pol+1][ii]))
                        axs[2][pol].plot(freq,np.angle(data[pol+1][ii]),marker='.',linestyle='None')

                    if pol==0:
                        title = "%s %s SPW%s XY*" % (prefix,ant,spw)
                    elif pol==1:
                        title = "%s %s SPW%s YX*" % (prefix,ant,spw)
                    #
                    axs[1][pol].set_title(title)
                    axs[1][pol].set_xlabel("Frequency [GHz]")
                    axs[1][pol].set_ylabel("Amplitude [a.u.]")
                    axs[1][pol].tick_params(axis='both', which='major')
                    axs[1][pol].grid(visible=True, linestyle='dashed')
                    #
                    axs[2][pol].set_title(title)
                    axs[2][pol].set_title(title)
                    axs[2][pol].set_xlabel("Frequency [GHz]")
                    axs[2][pol].set_ylabel("Phase [rad]")
                    axs[2][pol].tick_params(axis='both', which='major')
                    axs[2][pol].grid(visible=True, linestyle='dashed')
            
            fig.savefig('%s/%s_%s_Xpol_SPW%d.png' % (repdir,prefix,ant,spw))
            fig.clf()
            plt.close(fig)

            # CW peak plot
            cwChXX, noiseChXX_R, noiseChXX_R, cwPeakXX, noiseXX, noiseXX_R, noiseXX_L, rmsXX = getCWSignal2(data[0])
            if npol==2:
                cwChYY, noiseChYY_L, noiseChYY_L, cwPeakYY, noiseYY, noiseYY_R, noiseYY_L, rmsYY = getCWSignal2(data[1])
            elif npol==4:
                cwChYY, noiseChYY_L, noiseChYY_L, cwPeakYY, noiseYY, noiseYY_R, noiseYY_L, rmsYY = getCWSignal2(data[3])

            # Correlation of CW signal
            corrCW = getCWSignal(data)

            cwCh   = [cwChXX, cwChYY]
            cwPeak = [cwPeakXX, cwPeakYY]
            noise  = [noiseXX, noiseYY]
            noise_R = [noiseXX_R, noiseYY_R]
            noise_L = [noiseXX_L, noiseYY_L]
            rms    = [rmsXX, rmsYY]


            colorlist = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
            pol_idx = convertTimeAngle_Xpol(time, p_time, p_angle)

            pol_ave, pol_rms = [], []
            for pol in range(2):

                tmp_ave, tmp_rms = [], []
                for ii in range(len(p_angle)):
                    tmp_peak, tmp_weight = [], []
                    for jj in range(len(pol_idx[ii])):
                        tmp_peak.append(cwPeak[pol][pol_idx[ii][jj]])
                        tmp_weight.append(np.power(rms[pol][pol_idx[ii][jj]],-2.0))

                    wAve, wStd = weightedAveStd(tmp_peak, tmp_weight)
                    tmp_ave.append(wAve)
                    tmp_rms.append(wStd)
                                
                pol_ave.append(tmp_ave)
                pol_rms.append(tmp_rms)
                
            angWG1 = PolPCS[band]
            angRX  = PolRX[band]

            for pol in range(2):          
                axsPol[pol].errorbar(p_angle, pol_ave[pol], yerr=pol_rms[pol], fmt='o', color=colorlist[spwidx])
                    
                if pol==0:
                    title = "%s %s Pol-X" % (prefix,ant)
                elif pol==1:
                    title = "%s %s Pol-Y" % (prefix,ant)
                axsPol[pol].set_title(title)
                axsPol[pol].set_xlabel("WG2 Polarizer angle [deg]")
                axsPol[pol].set_ylabel("CW/comb amplitude [a.u.]")
                axsPol[pol].tick_params(axis='both', which='major')
                axsPol[pol].grid(visible=True, linestyle='dashed')
                
            # Polarzer Rotation for all correlation products
            pol_cw = []
            for pol in range(npol):
                tmp_pol = []
                for ii in range(len(p_angle)):
                    tmp = []
                    for jj in range(len(pol_idx[ii])):
                        tmp.append(corrCW[pol][pol_idx[ii][jj]])

                    tmp_pol.append(np.average(tmp))
                pol_cw.append(tmp_pol)

            if npol==2:
                axsPolxNull[0].scatter(p_angle, np.real(pol_cw[0]), marker='.', color=spw_color[spwidx], label='SPW%d-XX'%spw)

                axsPolyNull[0].scatter(p_angle, np.real(pol_cw[1]), marker='.', color=spw_color[spwidx], label='SPW%d-YY'%spw, linestyle='dashed')

                axsWG1Null[0].scatter(p_angle, np.real(pol_cw[0]), marker='.', color=spw_color[spwidx], label='SPW%d-XX'%spw)
                axsWG1Null[0].scatter(p_angle, np.real(pol_cw[1]), marker='.', color=spw_color[spwidx], label='SPW%d-YY'%spw, linestyle='dashed')

            elif npol==4:
                axsPolxNull[0].scatter(p_angle, np.real(pol_cw[0]), marker='.', color=spw_color[spwidx], label='SPW%d'%spw)
                axsPolxNull[1].scatter(p_angle, np.absolute(pol_cw[1]), marker='.', color=spw_color[spwidx], label='SPW%d'%spw)

                axsPolyNull[0].scatter(p_angle, np.real(pol_cw[3]), marker='.', color=spw_color[spwidx], label='SPW%d'%spw, linestyle='dashed')
                axsPolyNull[1].scatter(p_angle, np.absolute(pol_cw[1]), marker='.', color=spw_color[spwidx], label='SPW%d'%spw)

                axsWG1Null[0].scatter(p_angle, np.real(pol_cw[0]), marker='.', color=spw_color[spwidx], label='SPW%d-XX'%spw)
                axsWG1Null[0].scatter(p_angle, np.real(pol_cw[3]), marker='.', color=spw_color[spwidx], label='SPW%d-YY'%spw, linestyle='dashed')
                axsWG1Null[1].scatter(p_angle, np.absolute(pol_cw[1]), marker='.', color=spw_color[spwidx], label='SPW%d'%spw)

            if npol==2:
                XX, YY, gain_diff = AutoCorrGainCorrectionPolarizerRotaion(p_angle, angRX, pol_cw[0], pol_cw[1])
            elif npol==4:
                XX, YY, gain_diff = AutoCorrGainCorrectionPolarizerRotaion(p_angle, angRX, pol_cw[0], pol_cw[3])

            # gain correction
            pol_ave[1] = pol_ave[1]/gain_diff
            pol_rms[1] = pol_rms[1]/gain_diff

            if spwidx==0:
                #axsRxNull[0].errorbar(p_angle, pol_ave[0]/pol_ave[1], yerr=pol_rms[0], fmt='o', linestyle='dashed', label=ant)
                #axsRxNull[1].errorbar(p_angle, pol_ave[1]/pol_ave[0], yerr=pol_rms[1], fmt='o', linestyle='dashed', label=ant)
                axsRxNull[0].scatter(p_angle, 10.0*np.log10(pol_ave[0]/pol_ave[1]), marker='.', label=ant)
                axsRxNull[1].scatter(p_angle, 10.0*np.log10(pol_ave[1]/pol_ave[0]), marker='.', label=ant)


        title = "%s %s" % (prefix,ant)
        # Pol-X Null -> around Pol-Y angle
        axsPolxNull[0].set_title(title)
        axsPolxNull[0].set_xlabel("WG2 Polarizer angle [deg]")
        axsPolxNull[0].set_ylabel("Amplitute [a.u.]")
        axsPolxNull[0].tick_params(axis='both', which='major')
        axsPolxNull[0].grid(visible=True, linestyle='dashed')
        axsPolxNull[0].legend()

        axsPolxNull[1].set_xlabel("WG2 Polarizer angle [deg]")
        axsPolxNull[1].set_ylabel("CC Amplitute [a.u.]")
        axsPolxNull[1].tick_params(axis='both', which='major')
        axsPolxNull[1].grid(visible=True, linestyle='dashed')
        axsPolxNull[1].legend()

        xmin = PolRX[band][1]-10.0
        xmax = PolRX[band][1]+10.0
        idx = np.where( (p_angle>=xmin) & (p_angle<=xmax))[0]
        ymin = 0.7*np.real(pol_cw[0])[idx].min()
        ymax = 1.1*np.real(pol_cw[0])[idx].max()
        axsPolxNull[0].set_xlim(xmin, xmax)
        axsPolxNull[0].set_ylim(ymin, ymax)
        ymin = 0.0
        ymax = 1.1*np.absolute(pol_cw[1])[idx].max()
        axsPolxNull[1].set_xlim(xmin, xmax)
        axsPolxNull[1].set_ylim(ymin, ymax)

        # Pol-Y Null -> around Pol-X angle
        axsPolyNull[0].set_title(title)
        axsPolyNull[0].set_xlabel("WG2 Polarizer angle [deg]")
        axsPolyNull[0].set_ylabel("Amplitute [a.u.]")
        axsPolyNull[0].tick_params(axis='both', which='major')
        axsPolyNull[0].grid(visible=True, linestyle='dashed')
        axsPolyNull[0].legend()

        axsPolyNull[1].set_xlabel("WG2 Polarizer angle [deg]")
        axsPolyNull[1].set_ylabel("CC Amplitute [a.u.]")
        axsPolyNull[1].tick_params(axis='both', which='major')
        axsPolyNull[1].grid(visible=True, linestyle='dashed')
        axsPolyNull[1].legend()

        xmin = PolRX[band][0]-10.0
        xmax = PolRX[band][0]+10.0
        idx = np.where( (p_angle>=xmin) & (p_angle<=xmax))[0]
        if npol==2:
            ymin = 0.7*np.real(pol_cw[1])[idx].min()
            ymax = 1.1*np.real(pol_cw[1])[idx].max()
        elif npol==4:
            ymin = 0.7*np.real(pol_cw[3])[idx].min()
            ymax = 1.1*np.real(pol_cw[3])[idx].max()
        axsPolyNull[0].set_xlim(xmin, xmax)
        axsPolyNull[0].set_ylim(ymin, ymax)
        ymin = 0.0
        ymax = 1.2*np.absolute(pol_cw[1])[idx].max()
        axsPolyNull[1].set_xlim(xmin, xmax)
        axsPolyNull[1].set_ylim(ymin, ymax)

        # WG1 Null -> WG1 +- 90deg
        axsWG1Null[0].set_title(title)
        axsWG1Null[0].set_xlabel("WG2 Polarizer angle [deg]")
        axsWG1Null[0].set_ylabel("Amplitute [a.u.]")
        axsWG1Null[0].tick_params(axis='both', which='major')
        axsWG1Null[0].grid(visible=True, linestyle='dashed')
        axsWG1Null[0].legend()

        axsWG1Null[1].set_xlabel("WG2 Polarizer angle [deg]")
        axsWG1Null[1].set_ylabel("CC Amplitute [a.u.]")
        axsWG1Null[1].tick_params(axis='both', which='major')
        axsWG1Null[1].grid(visible=True, linestyle='dashed')
        axsWG1Null[1].legend()

        if band=='3':
            xmin = -90+PolPCS[band]-10.0
            xmax = -90+PolPCS[band]+10.0
        else:
            xmin = 90+PolPCS[band]-10.0
            xmax = 90+PolPCS[band]+10.0
        idx = np.where( (p_angle>=xmin) & (p_angle<=xmax))[0]
        ymin = 0.7*np.real(pol_cw[0])[idx].min()
        ymax = 1.1*np.real(pol_cw[0])[idx].max()
        axsWG1Null[0].set_xlim(xmin, xmax)
        axsWG1Null[0].set_ylim(ymin, ymax)
        ymin = 0.0
        ymax = 1.1*np.absolute(pol_cw[1])[idx].max()
        axsWG1Null[1].set_xlim(xmin, xmax)
        axsWG1Null[1].set_ylim(ymin, ymax)
        #

        figPol.savefig('%s/%s_%s_Xpol_CW_Pol.png' % (repdir,prefix,ant))
        figPol.clf()
        plt.close(figPol)

        figPolxNull.savefig('%s/%s_%s_Xpol_CW_PolX_Null.png' % (repdir,prefix,ant))
        figPolxNull.clf()
        plt.close(figPolxNull)

        figPolyNull.savefig('%s/%s_%s_Xpol_CW_PolY_Null.png' % (repdir,prefix,ant))
        figPolyNull.clf()
        plt.close(figPolyNull)

        figWG1Null.savefig('%s/%s_%s_Xpol_CW_WG1_Null.png' % (repdir,prefix,ant))
        figWG1Null.clf()
        plt.close(figWG1Null)

    title = "%s" % (prefix)
    # Pol-X Null -> around Pol-Y angle
    axsRxNull[0].set_title(title)
    axsRxNull[0].set_xlabel("WG2 Polarizer angle [deg]")
    axsRxNull[0].set_ylabel("Cross-pol leakage from Pol-Y to Pol-X [dB]")
    axsRxNull[0].tick_params(axis='both', which='major')
    axsRxNull[0].grid(visible=True, linestyle='dashed')
    axsRxNull[0].legend()

    xmin = PolRX[band][1]-10.0
    xmax = PolRX[band][1]+10.0
    axsRxNull[0].set_xlim(xmin, xmax)
    axsRxNull[0].set_ylim(-50, -10)

    # Pol-Y Null -> around Pol-X angle
    axsRxNull[1].set_title(title)
    axsRxNull[1].set_xlabel("WG2 Polarizer angle [deg]")
    axsRxNull[1].set_ylabel("Cross-pol leakage from Pol-X to Pol-Y [dB]")
    axsRxNull[1].tick_params(axis='both', which='major')
    axsRxNull[1].grid(visible=True, linestyle='dashed')
    axsRxNull[1].legend()

    xmin = PolRX[band][0]-10.0
    xmax = PolRX[band][0]+10.0
    axsRxNull[1].set_xlim(xmin, xmax)
    axsRxNull[1].set_ylim(-50, -10)

    figRxNull.savefig('%s/%s_Xpol_RX_Null.png' % (repdir,prefix))
    figRxNull.clf()
    plt.close(figRxNull)

    plt.close('all')

#------------------------------------------------------------------------------
# for mapping observations
#------------------------------------------------------------------------------
def plotPCSBeamMap(vis):

    intent = 'OBSERVE_TARGET#ON_SOURCE'
    repdir = 'report_'+vis.split('.')[0]
    prefix = vis.split('_')[-1][:-3]

    if (os.path.isdir(repdir) != True):
        os.system("mkdir %s" % repdir)

    antlist  = au.getAntennaNames(vis)
    scanlist = getScansForIntent(vis, intent)
    scan = scanlist[0]
    SpwsSQLD, SpwsFR, SpwsCA = getSpwsForScan(vis, scan, intent)

    nant  = len(antlist) 
    nspw = len(SpwsFR)
    
    antidlist = []
    for ant in antlist:
        antid = au.getAntennaIndex(vis,ant)
        antidlist.append(antid)

    plt.ioff()

    for antid in antidlist:
        ant = antlist[antid]

        # scan positions
        mytb = au.createCasaTool(tbtool)
        mytb.open(vis+'/POINTING')
        myt = mytb.query("ANTENNA_ID == %s" % str(antid))
        alltimes = myt.getcol('TIME')
        alldirections = myt.getcol('DIRECTION')
        alltargets = myt.getcol('TARGET')
        myt.close()
        mytb.close()

        x_pointing = np.degrees(alldirections[0][0])
        y_pointing = np.degrees(alldirections[1][0])
        x_target = np.degrees(alltargets[0][0])
        y_target = np.degrees(alltargets[1][0])

        for ii in range(len(x_pointing)):
            if x_pointing[ii]<0.0: x_pointing[ii] += 360.0
            if x_target[ii]<0.0: x_target[ii] += 360.0

        time_pointing = []
        for ii in range(len(alltimes)):
            timesi = str(alltimes[ii])+'s'
            timesi = qa.time(timesi, prec=10, form='fits')
            time_pointing.append(tu.get_datetime_from_isodatetime(timesi[0]))
        unixt_pointing = [t.timestamp() for t in time_pointing]

        #print("Debug: %d %d %d" % (len(time_pointing), len(x_pointing), len(y_pointing)))

        # scan grid for Correlator outputs
        spw0 = SpwsFR[0]
        ddid0 = au.getDataDescriptionId(vis, spw0)
        time0,data0,fdata0,state0 = getSpectralAutoData_hack(vis,antid,ddid0,scan)

        # find peak channel
        idx_peak_XX = np.unravel_index(np.argmax(data0[0]), data0[0].shape)
        #print("Debug: pol-X peak ", idx_peak_XX)
        idx_peak_YY = np.unravel_index(np.argmax(data0[-1]), data0[-1].shape)
        #print("Debug: pol-Y peak ", idx_peak_YY)

        if idx_peak_XX[0]==idx_peak_YY[0]:
            CHpeak = idx_peak_XX[0]
            print("peak channel = %d" % CHpeak)
            CHsignal = [CHpeak-2, CHpeak-1, CHpeak, CHpeak+1, CHpeak+2]

        # calculate total power
        data0 = np.sum(data0, axis=1)
        npol = len(data0)
        
        _, idx = np.unique(state0, return_index=True)
        list_subScan = state0[np.sort(idx)]
        ny = len(list_subScan)

        idx_subScan_grid_ac, coord_ac = [], []
        for subScan in list_subScan:
            #print('### subScan %d ###' % subScan)
            idx_subScan = np.where(state0==subScan)[0]
            nx = len(idx_subScan)
            tmp_x, tmp_y, tmp_idx, tmp_coord = [], [], [], []
            for idx in idx_subScan:
                x = np.interp(time0[idx].timestamp(), unixt_pointing, x_pointing)
                y = np.interp(time0[idx].timestamp(), unixt_pointing, y_pointing)
                xr = np.interp(time0[idx].timestamp(), unixt_pointing, x_target)
                yr = np.interp(time0[idx].timestamp(), unixt_pointing, y_target)
                tmp_coord.append([x-xr, y-yr])
                tmp_x.append(x)
                tmp_y.append(y)
                tmp_idx.append(idx)
            coord_ac.append(tmp_coord)

            if (tmp_x[0]-tmp_x[-1])>0.1:
                tmp_idx.reverse()
            idx_subScan_grid_ac.append(tmp_idx)

        # print grid size
        x = [ coord_ac[0][j][0] for j in range(len(coord_ac[0]))]
        xmax = np.max(x)
        xmin = np.min(x)
        dx = [np.absolute(coord_ac[i][j+1][0]-coord_ac[i][j][0]) for j in range(len(coord_ac[0])-1) for i in range(len(coord_ac))]
        print("Debug: x spacing %f (~%.3f m) " % (np.average(dx), 4.1e3*np.sin(np.deg2rad(np.average(dx)))))
        print("Debug: xmax, xmin = (%f, %f) " % (xmin*3600, xmax*3600) )

        coord_average = np.average(coord_ac, axis=1)
        y = [ coord_average[i] for i in range(len(coord_average))]
        ymax = np.max(y)
        ymin = np.min(y) 
        dy = [np.absolute(coord_average[i+1][1]-coord_average[i][1]) for i in range(len(coord_average)-1)]
        print("Debug: y spacing %f (~%.3f m)" % (np.average(dy), 4.1e3*np.sin(np.deg2rad(np.average(dy)))))
        print("Debug: ymax, ymin = (%f, %f) " % (ymin*3600, ymax*3600) )

        extent_imshow_ac  = (xmin*3600.0,xmax*3600.0,ymin*3600.0,ymax*3600.0)
        extent_contour_ac = (xmin*3600.0,xmax*3600.0,ymax*3600.0,ymin*3600.0)
        
        # Map for SPW FR TP
        fig, axs = plt.subplots(2,4, figsize=(12.8,9.6))
        fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,hspace=0.0,wspace=0.0)
        fig.suptitle('%s' % ant)

        fig_fr, axs_fr = plt.subplots(2,4, figsize=(12.8,9.6))
        fig_fr.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,hspace=0.0,wspace=0.0)
        fig_fr.suptitle('%s' % ant)

        for spwidx in range(nspw):

            spw  = SpwsFR[spwidx]
            ddid = au.getDataDescriptionId(vis, spw)
            freq = getFrequenciesGHz(vis, spw)

            # read auto-correlation data
            print("Debug: Antid=%d, SPW=%d, DD=%d" % (antid,spw,ddid))
            time,data,fdata,state = getSpectralAutoData_hack(vis,antid,ddid,scan)

            # baseline substraction & signal power
            SignalXX, SignalYY= [], []
            raw = np.array(data).transpose((0,2,1))
            print("Debug: ", len(raw), len(raw[0]), len(raw[0][0]))
            for dump in range(len(raw[0])):
                Chfit, XXfit, YYfit = [], [], []
                rawXX = np.absolute(raw[0][dump])
                rawYY = np.absolute(raw[-1][dump])
                for ch in range(-10,-4):
                    Chfit.append(CHpeak+ch)
                    XXfit.append(rawXX[CHpeak+ch])
                    YYfit.append(rawYY[CHpeak+ch])
                for ch in range(4,10):
                    Chfit.append(CHpeak+ch)
                    XXfit.append(rawXX[CHpeak+ch])
                    YYfit.append(rawYY[CHpeak+ch])
                #print("Debug: interpolation", len(Chfit), len(XXfit), len(YYfit))
                fxx = interp1d(Chfit, XXfit, kind='quadratic')
                fyy = interp1d(Chfit, YYfit, kind='quadratic')
                tmp_SignalXX = [np.absolute(raw[0][dump][ch])-fxx(ch) for ch in CHsignal]
                tmp_SignalYY = [np.absolute(raw[-1][dump][ch])-fyy(ch) for ch in CHsignal]
                SignalXX.append(np.sum(tmp_SignalXX))
                SignalYY.append(np.sum(tmp_SignalYY))
            
            # calclulate total power
            data = np.sum(data, axis=1)
            print("Debug: ", len(data[0]), len(SignalXX), len(SignalYY))

            intensity_XX, intensity_YY = [], []
            intensity_signal_XX, intensity_signal_YY = [], []
            for idx_subScan in idx_subScan_grid_ac:
                tmp_intensity_XX, tmp_intensity_YY = [], []
                tmp_intensity_signal_XX, tmp_intensity_signal_YY = [], []
                for idx in idx_subScan:
                    tmp_intensity_XX.append(np.abs(data[0][idx]))
                    tmp_intensity_YY.append(np.abs(data[-1][idx]))
                    tmp_intensity_signal_XX.append(SignalXX[idx])
                    tmp_intensity_signal_YY.append(SignalYY[idx])
                intensity_XX.append(tmp_intensity_XX)
                intensity_YY.append(tmp_intensity_YY)
                intensity_signal_XX.append(tmp_intensity_signal_XX)
                intensity_signal_YY.append(tmp_intensity_signal_YY)
            intensity_XX = intensity_XX[:][::-1]
            intensity_YY = intensity_YY[:][::-1]
            intensity_signal_XX = intensity_signal_XX[:][::-1]
            intensity_signal_YY = intensity_signal_YY[:][::-1]
            intensity = [intensity_XX, intensity_YY]
            intensity_signal = [intensity_signal_XX, intensity_signal_YY]

            # 2D gaussian fitting
            for i in range(2):
                img = np.array(intensity_signal[i])
                if not np.isnan(img).any():
                    x, y = np.meshgrid(np.linspace(0,img.shape[1],img.shape[1]),np.linspace(0,img.shape[0],img.shape[0]))
                    initial = (np.max(img), img.shape[1]/2.0,img.shape[0]/2.0,2,2)
                    print("Debug: initial ", initial)
                    img_ravel = img.ravel()
                    popt,pcov=curve_fit(gaussian_2d,(x,y),img_ravel,initial)
                    print("Debug: fitting ", popt)
                else:
                    print("Skipped 2D gaussian fitting")

            for i in range(2):
                axs[i][spwidx].imshow(intensity[i], extent=extent_imshow_ac, aspect='equal')
                axs[i][spwidx].contour(intensity[i], extent=extent_contour_ac, alpha=0.5, colors='white')

                axs_fr[i][spwidx].imshow(intensity_signal[i], extent=extent_imshow_ac, aspect='equal')
                axs_fr[i][spwidx].contour(intensity_signal[i], extent=extent_contour_ac, alpha=0.5, colors='white')

                if i==0:
                    axs[i][spwidx].set_title('SPW%d Pol-X' % spw)
                    axs_fr[i][spwidx].set_title('SPW%d Pol-X' % spw)
                elif i==1:
                    axs[i][spwidx].set_title('SPW%d Pol-Y' % spw)
                    axs_fr[i][spwidx].set_title('SPW%d Pol-Y' % spw)
                axs[i][spwidx].set_xlabel('Azumith offset [arcsec]')
                axs_fr[i][spwidx].set_xlabel('Azumith offset [arcsec]')
                if spwidx==0:
                    axs[i][spwidx].set_ylabel('Elevation offset [arcsec]')
                    axs[i][spwidx].tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
                    axs_fr[i][spwidx].set_ylabel('Elevation offset [arcsec]')
                    axs_fr[i][spwidx].tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
                else:
                    axs[i][spwidx].tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
                    axs_fr[i][spwidx].tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
                axs[i][spwidx].tick_params(direction='inout', bottom=True, left=True, right=True, top=True)
                axs_fr[i][spwidx].tick_params(direction='inout', bottom=True, left=True, right=True, top=True)
            
        fig.savefig('%s/%s_%s_SpwFrTP.png' % (repdir,prefix,ant))
        fig.clf()
        fig_fr.savefig('%s/%s_%s_SpwFr.png' % (repdir,prefix,ant))
        fig_fr.clf()
        plt.close('all')

        # Maps for SQLD
        spw0 = SpwsSQLD[0]
        ddid0 = au.getDataDescriptionId(vis, spw0)
        time0,data0,fdata0,state0 = getSpectralAutoData_hack(vis,antid,ddid0,scan)
        data0 = np.sum(data0, axis=1)
        npol = len(data0)
        
        _, idx = np.unique(state0, return_index=True)
        list_subScan = state0[np.sort(idx)]
        ny = len(list_subScan)

        idx_subScan_grid_tp, coord_tp = [], []
        for subScan in list_subScan:
            #print('### subScan %d ###' % subScan)
            idx_subScan = np.where(state0==subScan)[0]
            nx = len(idx_subScan)
            tmp_x, tmp_y, tmp_idx, tmp_coord = [], [], [], []
            for idx in idx_subScan:
                x = np.interp(time0[idx].timestamp(), unixt_pointing, x_pointing)
                y = np.interp(time0[idx].timestamp(), unixt_pointing, y_pointing)
                xr = np.interp(time0[idx].timestamp(), unixt_pointing, x_target)
                yr = np.interp(time0[idx].timestamp(), unixt_pointing, y_target)
                tmp_coord.append([x-xr, y-yr])
                tmp_x.append(x)
                tmp_y.append(y)
                tmp_idx.append(idx)
            coord_tp.append(tmp_coord)

            if (tmp_x[0]-tmp_x[-1])>0.1:
                tmp_idx.reverse()
            idx_subScan_grid_tp.append(tmp_idx)

        # print grid size
        x = [ coord_tp[0][j][0] for j in range(len(coord_tp[0]))]
        xmax = np.max(x)
        xmin = np.min(x)
        dx = [np.absolute(coord_tp[i][j+1][0]-coord_tp[i][j][0]) for j in range(len(coord_tp[0])-1) for i in range(len(coord_tp))]
        print("Debug: x spacing %f (~%.3f m) " % (np.average(dx), 4.1e3*np.sin(np.deg2rad(np.average(dx)))))
        print("Debug: xmax, xmin = (%f, %f) " % (xmin*3600, xmax*3600) )

        coord_average = np.average(coord_tp, axis=1)
        y = [ coord_average[i] for i in range(len(coord_average))]
        ymax = np.max(y)
        ymin = np.min(y) 
        dy = [np.absolute(coord_average[i+1][1]-coord_average[i][1]) for i in range(len(coord_average)-1)]
        print("Debug: y spacing %f (~%.3f m)" % (np.average(dy), 4.1e3*np.sin(np.deg2rad(np.average(dy)))))
        print("Debug: ymax, ymin = (%f, %f) " % (ymin*3600, ymax*3600) )

        extent_imshow_tp  = (xmin*3600.0,xmax*3600.0,ymin*3600.0,ymax*3600.0)
        extent_contour_tp = (xmin*3600.0,xmax*3600.0,ymax*3600.0,ymin*3600.0)

        # Map for SPW SQLD
        fig, axs = plt.subplots(2,4, figsize=(12.8,9.6))
        fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,hspace=0.0,wspace=0.0)
        fig.suptitle('%s' % ant)

        for spwidx in range(nspw):

            spw  = SpwsSQLD[spwidx]
            ddid = au.getDataDescriptionId(vis, spw)
            freq = getFrequenciesGHz(vis, spw)

            # read auto-correlation data
            print("Debug: Antid=%d, SPW=%d, DD=%d" % (antid,spw,ddid))
            time,data,fdata,state = getSpectralAutoData_hack(vis,antid,ddid,scan)

            intensity_XX, intensity_YY = [], []
            for idx_subScan in idx_subScan_grid_tp:
                tmp_intensity_XX, tmp_intensity_YY = [], []
                for idx in idx_subScan:
                    tmp_intensity_XX.append(np.abs(data[0][0][idx]))
                    tmp_intensity_YY.append(np.abs(data[-1][0][idx]))
                intensity_XX.append(tmp_intensity_XX)
                intensity_YY.append(tmp_intensity_YY)
            intensity_XX = intensity_XX[:][::-1]
            intensity_YY = intensity_YY[:][::-1]
            intensity = [intensity_XX, intensity_YY]

            for i in range(2):
                axs[i][spwidx].imshow(intensity[i], extent=extent_imshow_tp, aspect='equal')
                axs[i][spwidx].contour(intensity[i], extent=extent_contour_tp, alpha=0.5, colors='white')

                if i==0: axs[i][spwidx].set_title('SPW%d Pol-X' % spw)
                elif i==1: axs[i][spwidx].set_title('SPW%d Pol-Y' % spw)
                axs[i][spwidx].set_xlabel('Azumith offset [arcsec]')
                if spwidx==0:
                    axs[i][spwidx].set_ylabel('Elevation offset [arcsec]')
                    axs[i][spwidx].tick_params(labelbottom=True, labelleft=True, labelright=False, labeltop=False)
                else:
                    axs[i][spwidx].tick_params(labelbottom=True, labelleft=False, labelright=False, labeltop=False)
                axs[i][spwidx].tick_params(direction='inout', bottom=True, left=True, right=True, top=True)
            
        fig.savefig('%s/%s_%s_SpwSqld.png' % (repdir,prefix,ant))
        fig.clf()     
        plt.close('all')

def gaussian_2d(X, height, mu_x, mu_y, sigma_x, sigma_y):
    x, y = X
    z = height*np.exp(-(x-mu_x)**2/(2*sigma_x**2))* np.exp(-(y-mu_y)**2/(2*sigma_y**2))
    return z.ravel()

def doPCSAnalysisForBroadbandNoise(vis, sch, ech, polarizer_file=None):

    intent = 'CALIBRATE_DELAY#ON_SOURCE'
    repdir = 'report_'+vis.split('.')[0]
    prefix = vis.split('_')[-1][:-3]

    if (os.path.isdir(repdir) != True):
        os.system("mkdir %s" % repdir)

    antlist  = au.getAntennaNames(vis)
    scanlist = getScansForIntent(vis, intent)
    scan = scanlist[0]
    SpwsSQLD, SpwsFR, SpwsCA = getSpwsForScan(vis, scan, intent)

    nant  = len(antlist) 
    nspw = len(SpwsFR)

    antidlist = []
    for ant in antlist:
        antid = au.getAntennaIndex(vis,ant)
        antidlist.append(antid)
    #
    cclist = list(itertools.combinations(antidlist,2))

    list_coordinates = getDirectionToAntenna(vis,debug=True)
    print("Debug: ", list_coordinates)

    for spwidx in range(nspw):

        spw  = SpwsFR[spwidx]
        ddid = au.getDataDescriptionId(vis, spw)
        freq = getFrequenciesGHz(vis, spw)
        if    84.0 <= freq[0] < 116: band='3'
        elif 211.0 <= freq[0] < 275: band='6'
        elif 275.0 <= freq[0] < 373: band='7'

        print("SPW    : %d [%d/%d]" % (spw, spwidx+1, nspw))

        time0,data0,fdata0,state0 = getSpectralAutoData_hack(vis,0,ddid,scan)
        data0 = np.array(data0).transpose((0,2,1))
        npol  = len(data0)
        ndump = len(data0[0])
        nch   = len(data0[0][0])

        T = []
        for ii in range(ndump):
            T.append( (time0[ii]-time0[0]).total_seconds())

        # Check polarization angle
        p_time, p_angle = readPolarizerFile(polarizer_file)
        pol_idx = convertTimeAngle(time0, p_time, p_angle)

        # Auto-correlation
        AC, AC_pol = [], []
        for antid in antidlist:
            time,data,fdata,state = getSpectralAutoData_hack(vis,antid,ddid,scan)
            data = np.array(data).transpose((0,2,1))

            # channel averaging
            CHflag = 4
            ave_spec = []
            for ii in range(npol):
                tmp_dump = []
                for jj in range(ndump):
                    #tmp_spec = data[ii][jj][CHflag:-CHflag]
                    tmp_spec = data[ii][jj][sch:ech]
                    # vector average
                    tmp_dump.append(np.average(tmp_spec))
                ave_spec.append(tmp_dump)
            AC.append(ave_spec)

            # polarization rotation
            pol_amp, pol_std = [], []
            for pol in range(npol):
                tmp_amp, tmp_std = [], []
                for ii in range(len(p_angle)):
                    tmp_signal = []
                    for jj in range(len(pol_idx[ii])):
                        tmp_signal.append(ave_spec[pol][pol_idx[ii][jj]])
                    tmp_amp.append(np.average(np.absolute(tmp_signal)))
                    tmp_std.append(np.std(np.absolute(tmp_signal)))
                pol_amp.append(tmp_amp)
                pol_std.append(tmp_std)
            AC_pol.append(pol_amp)

        # Cross-correlation
        CC, CC_pol = [], []
        for cc in cclist:
            iant1,iant2 = cc
            time,data = getSpectralData_hack(vis,iant1,iant2,ddid,scan)
            data = np.array(data).transpose((0,2,1))

            # channel averaging
            CHflag = 4
            ave_spec = []
            for ii in range(npol):
                tmp_dump = []
                for jj in range(ndump):
                    #tmp_spec = data[ii][jj][CHflag:-CHflag]
                    tmp_spec = data[ii][jj][sch:ech]
                    # vector average
                    tmp_dump.append(np.average(tmp_spec))
                ave_spec.append(tmp_dump)
            CC.append(ave_spec)

            # polarization rotation
            pol_amp, pol_phs = [], []
            for pol in range(npol):
                tmp_amp, tmp_phs = [], []
                for ii in range(len(p_angle)):
                    tmp_signal = []
                    for jj in range(len(pol_idx[ii])):
                        tmp_signal.append(ave_spec[pol][pol_idx[ii][jj]])
                    # vector average
                    ave = np.average(tmp_signal)
                    tmp_amp.append(np.absolute(ave))

                    tmp_phs.append(np.angle(ave))
                pol_amp.append(tmp_amp)
                pol_phs.append(tmp_phs)
            CC_pol.append([pol_amp, pol_phs])

        print("Debug: ", len(AC_pol), len(AC_pol[0]))
        print("Debug: ", len(CC_pol), len(CC_pol[0]), len(CC_pol[0][0]))

        # local minimum in AC
        p_nullWG1  = np.where((p_angle>PolPCS[band]-10) & (p_angle<PolPCS[band]+10))
        p_nullPolX = np.where((p_angle>PolRX[band][0]-10) & (p_angle<PolRX[band][0]+10))
        p_nullPolY = np.where((p_angle>PolRX[band][1]-10) & (p_angle<PolRX[band][1]+10))
        print("Debug: ", p_nullWG1[0])
        print("Debug: ", p_nullPolX[0])
        print("Debug: ", p_nullPolY[0])

        idx_AC_nullWG1, idx_AC_nullPolX, idx_AC_nullPolY = [], [], []
        for antid in antidlist:
            # Null by WG1
            idx_nullWG1, idx_nullPolX, idx_nullPolY =[], [], []
            for pol in range(npol):
                tmp = np.array([AC_pol[antid][pol][i] for i in p_nullWG1[0]])
                idx_nullWG1.append(np.argmin(tmp))

                tmp = np.array([AC_pol[antid][pol][i] for i in p_nullPolX[0]])
                idx_nullPolX.append(np.argmin(tmp))

                tmp = np.array([AC_pol[antid][pol][i] for i in p_nullPolY[0]])
                idx_nullPolY.append(np.argmin(tmp))
            #
            idx_AC_nullWG1.append(idx_nullWG1)
            idx_AC_nullPolX.append(idx_nullPolX)
            idx_AC_nullPolY.append(idx_nullPolY)

        print("Debug: ", idx_AC_nullWG1)
        print("Debug: ", idx_AC_nullPolX)
        print("Debug: ", idx_AC_nullPolY)

        # Plot: all antennas
        plt.ioff()
        fig, axs =[], []
        for pol in range(npol):
            tmp_fig, tmp_axs = plt.subplots(nant,nant,figsize=[12.8,12.8],dpi=300)
            tmp_fig.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,hspace=0.3,wspace=0.3)
            fig.append(tmp_fig)
            axs.append(tmp_axs)

        # AC
        for antid in antidlist:
            for pol in range(npol):
                axs[pol][antid][antid].plot(p_angle,AC_pol[antid][pol])
                #ID = '%s*%s' % (antid,antid)
                ID = '%s' % (antlist[antid])
                axs[pol][antid][antid].text(0.99, 0.99, ID, va='top', ha='right', transform=axs[pol][antid][antid].transAxes)

        # CC
        for ii in range(len(cclist)):
            iant1,iant2 = cclist[ii]
            for pol in range(npol):
                axs[pol][iant1][iant2].plot(p_angle,CC_pol[ii][0][pol])
                axs[pol][iant2][iant1].plot(p_angle,CC_pol[ii][1][pol],marker='.',linestyle='None')
                #ID = '%s*%s' % (iant1,iant2)
                ID = '%s*%s' % (antlist[iant1], antlist[iant2])
                axs[pol][iant1][iant2].text(0.99, 0.99, ID, va='top', ha='right', transform=axs[pol][iant1][iant2].transAxes)
                axs[pol][iant2][iant1].text(0.99, 0.99, ID, va='top', ha='right', transform=axs[pol][iant2][iant1].transAxes)
                #
                axs[pol][iant1][iant2].set_ylim(-0.05, 1.2)
                #axs[pol][iant2][iant1].set_ylim(-np.pi,np.pi)
                axs[pol][iant2][iant1].set_ylim(-3.5,3.5)

        for pol in range(npol):
            if npol==2:
                if pol==0: pol_name = 'XX'
                elif pol==1: pol_name = 'YY'
            elif npol==4:
                if pol==0: pol_name = 'XX'
                elif pol==1: pol_name = 'XY*'
                elif pol==2: pol_name = 'YX*'
                elif pol==3: pol_name = 'YY'
            fig[pol].savefig('%s/%s_SPW%d_%s_continuum.png' % (repdir,prefix,spw,pol_name))
            fig[pol].clf()
        
        # All AC in one plot
        for pol in range(npol):
            if npol==2:
                if pol==0: pol_name = 'XX'
                elif pol==1: pol_name = 'YY'
            elif npol==4:
                if pol==0: pol_name = 'XX'
                elif pol==1: pol_name = 'XY*'
                elif pol==2: pol_name = 'YX*'
                elif pol==3: pol_name = 'YY'
            
            fig_ac, axs_ac = plt.subplots(1,1,figsize=[12.8,12.8],dpi=300)

            for antid in antidlist:
                axs_ac.plot(p_angle,AC_pol[antid][pol], label=antlist[antid])
            axs_ac.legend()
            axs_ac.set_xlim(-100.0, 100.0)

            fig_ac.savefig('%s/%s_SPW%d_AC_%s.png' % (repdir,prefix,spw,pol_name))
            fig_ac.clf()

        # All CC amplitude in one plot
        for pol in range(npol):
            if npol==2:
                if pol==0: pol_name = 'XX'
                elif pol==1: pol_name = 'YY'
            elif npol==4:
                if pol==0: pol_name = 'XX'
                elif pol==1: pol_name = 'XY*'
                elif pol==2: pol_name = 'YX*'
                elif pol==3: pol_name = 'YY'
            
            fig_cc, axs_cc = plt.subplots(1,1,figsize=[12.8,12.8],dpi=300)

            for ii in range(len(cclist)):
                iant1,iant2 = cclist[ii]
                axs_cc.plot(p_angle,CC_pol[ii][0][pol], label='%s*%s'% (antlist[iant1], antlist[iant2]))
            axs_cc.legend()
            axs_cc.set_xlim(-100.0, 100.0)

            fig_cc.savefig('%s/%s_SPW%d_CC_%s.png' % (repdir,prefix,spw,pol_name))
            fig_cc.clf()
        plt.close('all')
