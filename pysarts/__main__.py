import sys
import os
from . import workflow

# Parse Arguments
# TODO: Rewrite using argparse
if len(sys.argv) != 2:
    print("USAGE: pysarts PROJECT_DIR")

PROJECT_DIR = sys.argv[1]
os.chdir(PROJECT_DIR)
workflow.load_config('config.yml')
workflow.execute_load_clip_resample_convert_step()
workflow.execute_invert_unwrapped_phase()
workflow.execute_calculate_master_atmosphere()
