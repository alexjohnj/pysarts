import sys
import os
import logging

import yaml

from . import processing

# Parse Arguments
# TODO: Rewrite using argparse
if len(sys.argv) != 2:
    print("USAGE: pysarts PROJECT_DIR")

PROJECT_DIR = sys.argv[1]
os.chdir(PROJECT_DIR)
CONFIG = {}
with open('config.yml') as f:
    CONFIG = yaml.safe_load(f)

logging.basicConfig(level=CONFIG.get("log_level", "WARN"))

# Step 1, find the interferograms matching the dates defined in the configuration file.
ifgs = processing.find_ifgs_for_dates(CONFIG["files"]["uifg_dir"],
                                      CONFIG["master_date"].date(),
                                      CONFIG.get("dates", None))

logging.info("Found %d interferograms matching date criteria.", len(ifgs))
logging.debug("Found interferograms: %s", ifgs)
