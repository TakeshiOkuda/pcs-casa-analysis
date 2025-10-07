#!/users/tokuda/casa-latest/bin/python3
"""
PCS analysis script: only for personal use at red-osf.osf.alma.cl
    - 2017-09-03: TOkuda - created, based on atmcalScanAna.py
    - 2022-06-10: TOkuda - modified for Python3.6 & CASA-6.5.0
    - 2022-06-22: TOkuda - updated using analysisUtils
    - 2022-07-29: TOkuda - modified for PCS analyses
    - 2024-04-25: TOkuda - modified for X-pol leak
"""
import sys
import os
import argparse
import time
import subprocess

import logging
from logging_setup import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

scriptsDir = '/users/tokuda/scripts/PCS_analysis_dev'
path2Casa  = '/users/tokuda/casa-latest'

#------------------------------------------------------------------------------
def initOptionsParser():

    parser = argparse.ArgumentParser(description="Parse command-line arguments.")
    parser.add_argument('-f', '--file', dest='file', type=str, required=True, help='EB name (ASDM identifier).')
    parser.add_argument('-p', '--polarizer', dest='polarizer', type=str, help='Polarizer angles file')
    parser.add_argument('-x', '--xpol', dest='xpol', action='store_true', default=False,
                        help='Enable analysis of cross-polarization data.')
    parser.add_argument('--path2au', dest='path2au', type=str,
                        default=os.environ.get('PATH_TO_AU', '/users/tokuda/local/scripts/analysis_scripts'),
                        help='Path to the analysisUtils package (defaults to PATH_TO_AU or a local fallback).')
    parser.add_argument('--path2casa', dest='path2casa', type=str,
                        default=os.environ.get('PATH_TO_CASA', '/users/tokuda/casa-latest/lib/py/lib/python3.6/site-packages'),
                        help='Path to the CASA python packages (defaults to PATH_TO_CASA or a local fallback).')
    return parser

#------------------------------------------------------------------------------
def checkOptions(args):
    print(f"ExUid: {args.file}")
    # Check antenna
    if args.polarizer is None:
        print("A file of polarizer angles is not specified. Skip an analysis of a polarizer rotation measurement.")
    else:
        if os.path.isfile(args.polarizer):
            print(f"{args.polarizer} is found.")
        else:
            print(f"{args.polarizer} is not found. Aborted this script.")
            sys.exit(0)
        
#------------------------------------------------------------------------------
def downloadASDM(args):

    # check ASDM name
    if args.file.count('_') == 0:
        exportedASDM1 = args.file
        exportedASDM2 = args.file.replace(':', '_').replace('/', '_')
    else:
        tmp_ASDM = args.file.split('_')
        tmp_ASDM = list(filter(lambda s: s != '', tmp_ASDM))
        if len(tmp_ASDM) != 4:
            logger.error('Unexpected ASDM identifier format: %s', args.file)
            sys.exit(1)
        exportedASDM1 = tmp_ASDM[0] + '://' + tmp_ASDM[1] + '/' + tmp_ASDM[2] + '/' + tmp_ASDM[3]
        exportedASDM2 = args.file

    if not os.path.isdir(f'./{exportedASDM2}'):
        logger.info("asdmExport... ")
        try:
            subprocess.run(['asdmExport', exportedASDM1], check=True)
        except subprocess.CalledProcessError as exc:
            logger.error("asdmExport failed: %s", exc)
            sys.exit(1)
    else:
        logger.info("asdmExport is skipped")

#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Initialize the argument parser and validate arguments
    parser = initOptionsParser()
    args = parser.parse_args()
    checkOptions(args)

    # Configure external module paths
    path2au = args.path2au
    path2casa = args.path2casa

    os.environ['PATH_TO_AU'] = path2au
    os.environ['PATH_TO_CASA'] = path2casa

    if not os.path.isdir(path2au):
        logger.error(f"analysisUtils directory not found: {path2au}")
        sys.exit(1)
    if not os.path.isdir(path2casa):
        logger.error(f"CASA python path not found: {path2casa}")
        sys.exit(1)

    sys.path.append(path2au)
    sys.path.append(path2casa)

    try:
        import analysisUtils as au
        from casatasks import (
            importasdm, listobs,
        )        
        from casatools import msmetadata as msmdtool
        from casatools import table as tbtool
        
        # personal modules
        import analysisPCS_utils as pcsu
        #from asdmUtils import getScanDic, getFieldId
    except Exception as e:
        logger.error(f"Failed to import required modules: {e}")
        sys.exit(1)

    logger.info('Step-1: Downloading ASDM is running...')
    downloadASDM(args)

    # Prepare for the analysis
    asdm = args.file.replace(":", "_").replace("/", "_")
    msfile = asdm + '.ms'

    if not os.path.isdir(f'./{msfile}'):
        logger.info("Step-2: importasdm is running...")
        asis = 'Antenna Station Receiver Source CalAtmosphere CalWVR CorrelatorMode SBSummary'
        importasdm(asdm=asdm, asis=asis, bdfflags=True, lazy=False, process_caldevice=False)
    else:
        logger.info("Step-2: importasdm is skipped ...")
    
    sfsdr = au.stuffForScienceDataReduction() 
    sfsdr.fixForCSV2555(msfile)

    # make plot antenna positions
    logger.info("Step-3: plot of antenna positions")
    pcsu.showPCSPosition(vis=msfile)

    # Check all spectra
    logger.info("Step-4: Creating all spectra plots")
    pcsu.plotAllSpectra(vis=msfile)

    # analyses using SQLD
    logger.info("Step-5: Creating plots of SQLD")
    pcsu.doPCSAnalysisForSQLD(vis=msfile, polarizer_file=args.polarizer, Xpol=args.xpol)

    # analyses using full-resolution AC
    logger.info("Step-6: Creating plots of full-resolution auto-correlation ###")

    if args.xpol and args.polarizer!=None:
        pcsu.doPCSAnalysisForXpol(vis=msfile, polarizer_file=args.polarizer, Xpol=args.xpol)
    elif args.xpol==False:
        pcsu.doPCSAnalysisForACFR(vis=msfile, polarizer_file=args.polarizer)

    # analyses using channel-averaged AC
    #print("### Creating plots of channel-averaged auto-correlation ###")
    #doPCSAnalysisForACCA(vis=msfile, polarizer_file=pol_file)

# end
