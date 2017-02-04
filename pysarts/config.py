"""Configuration for pysarts.

Call config.load_yaml() to initialise from a YAML configuration file.

Configuration can also be handled programmatically. All settings have a default
value that can be used except:

  - MASTER_DATE

These values default to None and must be set manually after importing this
module.

"""
import os.path
import yaml

"""The master date used in all timeseries calculations. This is a datetime
object, the time giving the acquisition time of ALL interferograms in the time
series.

"""
MASTER_DATE = None

"""SLC dates to use in timeseries calculations. This is a list of date
objects. An empty list or `None` means use all available SLC dates.

"""
DATES = None

"""The directory where unwrapped interferograms are stored. Defaults to
'./uifg'"""
UIFG_DIR = './uifg'

"""The directory where output and intermediary files are saved. Defaults to
'.'"""
SCRATCH_DIR = '.'

"""The directory where weather radar images are stored as NetCDF files. Defaults
to './weather_radar'. Images are sorted into subdirectories by YEAR/MONTH and
are named YYYYMMDDHHMM.

"""
WEATHER_RADAR_DIR = './weather_radar'

"""Resolution to resample to. Dict with keys 'delta_x' and 'delta_y' giving
resolution in metres. Default to `None` indicating no resampling.

"""
RESOLUTION = None

"""Bounding box for study region. Dictionary with the keys 'lat_min', 'lat_max',
'lon_min', 'lon_max'. Default `None` indicating no bounding box.

"""
REGION = None

"""String indicating the verbosity of logging during the processing
workflow. Default 'WARN'. One of 'CRITICAL', 'ERROR', 'WARNING', 'INFO' or
'DEBUG'

"""
LOG_LEVEL = 'WARN'

def load_from_yaml(path):
    """Load configuration from a YAML file."""
    conf = {}
    with open(path) as f:
        conf = yaml.safe_load(f)

    global MASTER_DATE, DATES
    global UIFG_DIR, SCRATCH_DIR, WEATHER_RADAR_DIR
    global RESOLUTION, REGION
    global LOG_LEVEL

    MASTER_DATE = conf['master_date']
    DATES = conf.get('dates', DATES)
    UIFG_DIR = os.path.expanduser(conf['files'].get('uifg_dir', UIFG_DIR))
    SCRATCH_DIR = os.path.expanduser(conf['files'].get('scratch_dir', SCRATCH_DIR))
    WEATHER_RADAR_DIR = os.path.expanduser(conf['files'].get('wr_dir', WEATHER_RADAR_DIR))
    RESOLUTION = conf.get('resolution', RESOLUTION)
    LOG_LEVEL = conf.get('log_level', LOG_LEVEL)
    REGION = conf.get('region', REGION)
