import logging


def setup_logging(level=logging.INFO):
    """Configure the root logger once."""
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(levelname)s - %(message)s')

