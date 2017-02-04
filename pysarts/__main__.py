from datetime import datetime
import argparse
import sys
import os

from . import workflow

def execute_all_steps():
    workflow.execute_load_clip_resample_convert_step()
    workflow.execute_invert_unwrapped_phase()
    workflow.execute_calculate_master_atmosphere()

mainParser = argparse.ArgumentParser(prog='pysarts')
mainParser.set_defaults(func=execute_all_steps)
mainParser.add_argument('-d', '--directory',
                        action='store',
                        default='.',
                        help='Path to the project directory')
mainParser.add_argument('-c', '--config',
                        action='store',
                        default='config.yml',
                        help='Path to a project configuration file')
mainParser.add_argument('-m', '--master',
                        action='store',
                        default=None,
                        help='Master date (overrides what is in the configuration file). Format YYYYmmddTHHMM.')

subparsers = mainParser.add_subparsers()
clip_resample_parser = subparsers.add_parser('clip-resample',
                                             help='Clip and resample unwrapped interferograms')
clip_resample_parser.set_defaults(func=workflow.execute_load_clip_resample_convert_step)
invert_unwrapped_parser = subparsers.add_parser('invert',
                                               help='Invert unwrapped time series interferograms')
invert_unwrapped_parser.set_defaults(func=workflow.execute_invert_unwrapped_phase)
calculate_master_atmos_parser = subparsers.add_parser('master-atmos',
                                                      help='Calculate the master atmosphere')
calculate_master_atmos_parser.set_defaults(func=workflow.execute_calculate_master_atmosphere)
weather_radar_parser = subparsers.add_parser('radar-correlation',
                                             help='Calculate the correlation between weather radar rainfall and master atmosphere.')
weather_radar_parser.set_defaults(func=workflow.execute_master_atmosphere_rainfall_correlation)

# Parse Arguments
args = mainParser.parse_args()
os.chdir(args.directory)
workflow.load_config(args.config)

if args.master:
    workflow.config.MASTER_DATE = datetime.strptime(args.master, '%Y%m%dT%H%M')

args.func()
