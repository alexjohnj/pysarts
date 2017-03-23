"""pysarts interactive usage module.
"""
import argparse
from datetime import datetime
import logging
import os

from . import workflow
from . import util

def execute_all_steps(args):
    workflow.execute_load_clip_resample_convert_step(args)
    workflow.execute_invert_unwrapped_phase(args)
    workflow.execute_calculate_dem_matmosphere_error(args)


def execute_on_all_slc_dates(args):
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
    workflow.execute_load_clip_resample_convert_step(args)

    # Let's go!
    logging.info('Going to run processing workflow on %d master dates', len(ifg_dates))
    for date in ifg_dates:
        logging.info('Switching master date to %s', date.strftime('%Y-%m-%d'))
        workflow.config.MASTER_DATE = datetime.combine(date, slc_time)
        workflow.execute_invert_unwrapped_phase(args)
        workflow.execute_calculate_dem_matmosphere_error(args)


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

era_delay_parser = subparsers.add_parser('era-delay',
                                         help=('Calculate wet and dry delays '
                                               'using ERA-interim weather '
                                               'models'))
era_delay_parser.set_defaults(func=workflow.execute_calculate_era_delays)
era_delay_parser.add_argument('-c', '--max-processes',
                              action='store',
                              type=int,
                              default=2,
                              help='Maximum number of subprocesses to spawn')
era_delay_parser.add_argument('-r', '--rainfall',
                              action='store',
                              nargs=2,
                              default=None,
                              type=int,
                              help=('Include rainfall data in correction.'
                                    'Include the minimum and maximum'
                                    'pressure levels to modify with'
                                    'rainfall data.'))
era_delay_parser.add_argument('-b', '--blur',
                              action='store',
                              default=0,
                              type=float,
                              help=('Standard deviation of Gaussian filter '
                                    'to apply to relative humidity. Only '
                                    'applies if -r is passed.'))
era_delay_parser.add_argument('-d', '--dates',
                              action='store',
                              default=None,
                              nargs='*',
                              help='Limit calculations to these dates.')

ifg_delay_parser = subparsers.add_parser('insar-delay',
                                         help=('Calculate interferometric '
                                               'slant delays'))
ifg_delay_parser.set_defaults(func=workflow.execute_calculate_ifg_delays)

liquid_delay_parser = subparsers.add_parser('liquid-delay',
                                            help=('Estimate the liquid '
                                                  'delay from rainfall data'))
liquid_delay_parser.set_defaults(func=workflow.execute_calculate_liquid_delay)
liquid_delay_parser.add_argument('-c', '--cloud-thickness',
                                 type=int,
                                 action='store',
                                 required=True,
                                 help=('Cloud layer thickness in km'))
liquid_delay_parser.add_argument('-d', '--date',
                                 action='store',
                                 default=None,
                                 help=('Date to calculate delay for.'
                                       'Defaults to configuration master'
                                       'date.'))

correction_parser = subparsers.add_parser('apply-corrections',
                                          help=('Apply corrections to '
                                                'interferograms'))
correction_parser.add_argument('-l', '--liquid', action='store_true',
                               help=('Apply a correction for the liquid delay '
                                     'if it has been calculated.'))
correction_parser.add_argument('-w', '--wet', action='store_true',
                               help=('Apply a correction for the wet delay'))
correction_parser.add_argument('-y', '--dry', action='store_true',
                               help=('Apply a correction for the dry delay'))
correction_parser.add_argument('-t', '--total', action='store_true',
                               help=('Apply a correction for all delays. '
                                     'Overrides -w, -l and -y'))
correction_parser.add_argument('-d', '--dates', action='store', nargs='+',
                               default=None, help=('Apply a correction for '
                                                   'pairings of specific '
                                                   'dates.'))
correction_parser.set_defaults(func=workflow.execute_correction_step)

clean_parser = subparsers.add_parser('clean',
                                     help='Remove pysarts files')
clean_parser.set_defaults(func=workflow.execute_clean_step)

# Parse Arguments
args = mainParser.parse_args()
os.chdir(args.directory)
workflow.load_config(args.config)

if args.master:
    workflow.config.MASTER_DATE = datetime.strptime(args.master, '%Y%m%dT%H%M')

args.func(args)
