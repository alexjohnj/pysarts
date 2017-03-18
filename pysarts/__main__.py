"""pysarts interactive usage module.
"""
import argparse
from datetime import datetime
import logging
import os

from . import workflow
from . import util

def execute_all_steps():
    workflow.execute_load_clip_resample_convert_step()
    workflow.execute_invert_unwrapped_phase()
    workflow.execute_calculate_dem_matmosphere_error()

def execute_on_all_slc_dates():
    """Executes the full processing stack on all the SLC dates in the UIFG_DIR
    directory that aren't filtered by the config file.

    """
    # First find all the dates in the UIFG_DIR directory
    ifg_paths = workflow.find_ifgs()
    ifg_date_pairs = [util.extract_timestamp_from_ifg_name(path) for path in ifg_paths]
    ifg_dates = set([date for pair in ifg_date_pairs for date in pair])

    # Copy the acquisition
    slc_time = workflow.config.MASTER_DATE.time()

    # Only need to run the clip, resample, convert step once.
    workflow.execute_load_clip_resample_convert_step()

    # Let's go!
    logging.info('Going to run processing workflow on %d master dates', len(ifg_dates))
    for date in ifg_dates:
        logging.info('Switching master date to %s', date.strftime('%Y-%m-%d'))
        workflow.config.MASTER_DATE = datetime.combine(date, slc_time)
        workflow.execute_invert_unwrapped_phase()
        workflow.execute_calculate_dem_matmosphere_error()

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
calculate_master_atmos_parser = subparsers.add_parser('dem-master-atmos',
                                                      help='Calculate the DEM error and master atmosphere.')
calculate_master_atmos_parser.set_defaults(func=workflow.execute_calculate_dem_matmosphere_error)

weather_radar_parser = subparsers.add_parser('radar-correlation',
                                             help='Calculate the correlation between weather radar rainfall and master atmosphere.')
weather_radar_parser.set_defaults(func=workflow.execute_master_atmosphere_rainfall_correlation)
weather_radar_parser.add_argument('-r', '--rain-tolerance',
                                  action='store',
                                  type=float,
                                  default=0,
                                  help='Minimum rainfall intensity to include in calculation')

god_mode_parser = subparsers.add_parser('alldates',
                                        help="""Run the full processing workflow
                                        on all dates allowed in the
                                        configuration file.""")
god_mode_parser.set_defaults(func=execute_on_all_slc_dates)

train_export_parser = subparsers.add_parser('export-train',
                                            help='Export files for processing with TRAIN')
train_export_parser.set_defaults(func=workflow.execute_export_train)

zenith_delay_parser = subparsers.add_parser('era-zenith-delay',
                                            help='Calculate zenith delays from ERA')
zenith_delay_parser.set_defaults(func=workflow.execute_calculate_zenith_delays)

# Parse Arguments
args = mainParser.parse_args()
os.chdir(args.directory)
workflow.load_config(args.config)

if args.master:
    workflow.config.MASTER_DATE = datetime.strptime(args.master, '%Y%m%dT%H%M')

args.func(args)
