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

# Parse Arguments
args = mainParser.parse_args()
os.chdir(args.directory)
workflow.load_config(args.config)

args.func()
