"""Functions that make up the workflow of pysarts when it is run as a program.

Almost all of these functions have side-effects.
"""

import logging
import os
from datetime import datetime
import glob
from multiprocessing.pool import Pool
from netCDF4 import Dataset

import numpy as np
import yaml
import scipy.io as scpio

import matplotlib.pyplot as plt

from .era import ERAModel
from . import corrections
from . import config
from . import inversion
from . import nimrod
from . import processing
from . import timeseries
from . import util

def load_config(path):
    """Loads the configuration from the YAML file at `path`, updating the config and
    logging modules."""
    config.load_from_yaml(path)
    logging.basicConfig(level=config.LOG_LEVEL)

def find_ifgs():
    """Find interferograms for the dates defined in the configuration file.

    Returns the path to interferograms satisfying the configuration.

    """
    return processing.find_ifgs_for_dates(config.UIFG_DIR, config.MASTER_DATE.date(), config.DATES)

def load_clip_resample(path):
    """Loads, clips and resamples an interferogram.

    Clipping and resampling is based on the user's configuration.

    Returns an ifg dictionary that contains the clipped data.

    """
    SHOULD_CLIP = config.REGION != None
    SHOULD_RESAMPLE = config.RESOLUTION != None

    logging.info('Processing %s', os.path.splitext(os.path.basename(path))[0])
    logging.debug('Loading %s', path)
    ifg = processing.open_ifg_netcdf(path)

    if SHOULD_CLIP:
        logging.debug('Clipping')
        lon_bounds = (config.REGION['lon_min'], config.REGION['lon_max'])
        lat_bounds = (config.REGION['lat_min'], config.REGION['lat_max'])
        processing.clip_ifg(ifg, lon_bounds, lat_bounds)
    else:
        logging.debug('Clipping disabled')

    if SHOULD_RESAMPLE:
        logging.debug('Resampling')
        processing.resample_ifg(ifg, config.RESOLUTION['delta_x'], config.RESOLUTION['delta_y'])
    else:
        logging.debug('Resampling disabled')

    output_dir = os.path.join(config.SCRATCH_DIR, 'uifg_resampled')
    save_ifg_to_npy(ifg, output_dir)

    logging.info('Extracting grid from %s',
                 os.path.splitext(os.path.basename(path))[0])
    extract_grid_from_ifg(ifg, os.path.join(config.SCRATCH_DIR, 'grid.txt'))

    return None

def save_ifg_to_npy(ifg, out_dir):
    """Saves an ifg dict as a Numpy array.

    The file name will be SLAVEDATE_MASTERDATE.npy
    """
    out_name = (ifg['slave_date'].strftime('%Y%m%d') + '_' + ifg['master_date'].strftime('%Y%m%d')
                + '.npy')
    out_path = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    logging.debug('Saving %s', out_path)

    # MaskedArrays can't be saved yet.
    if isinstance(ifg['data'], np.ma.MaskedArray):
        np.save(out_path, ifg['data'].data)
    else:
        np.save(out_path, ifg['data'])

    return None

def extract_grid_from_ifg(ifg, output_name):
    """Extracts information necessary to reconstruct the grid and saves it to a
    file.

    The output format is a 2 line plain text file:
       lon=lon_min:lon_max:nlon
       lat=lat_min:lat_max:nlat
    """
    lon_min = np.amin(ifg['lons'])
    lon_max = np.amax(ifg['lons'])
    nlon = len(ifg['lons'])
    lat_min = np.amin(ifg['lats'])
    lat_max = np.amax(ifg['lats'])
    nlat = len(ifg['lats'])

    with open(output_name, 'w') as f:
        f.write('lon={:.5f}:{:.5f}:{:d}\n'.format(lon_min, lon_max, nlon))
        f.write('lat={:.5f}:{:.5f}:{:d}\n'.format(lat_min, lat_max, nlat))

    return None

def read_grid_from_file(path):
    """Reads a grid save by extract_grid_from_ifg

    Returns a 2-tuple containing (lons, lats)
    """
    lons = []
    lats = []
    with open(path) as f:
        for line in f:
            key, rng = tuple(line.split('='))
            rng_min, rng_max, rng_len = tuple(rng.split(':'))
            grid_elements = np.linspace(float(rng_min), float(rng_max), int(rng_len))

            if key == 'lon':
                lons = grid_elements
            elif key == 'lat':
                lats = grid_elements
            else:
                raise KeyError

    return (lons, lats)


def find_closest_weather_radar_files(master_date, target_dir=config.WEATHER_RADAR_DIR):
    """Searches the weather radar directory for the closest weather radar images
    before and after a date and time.

    Can be made to search other directories using the target_dir argument.

    Returns
    -------

    before_path : str
      A path to the closest weather radar image before the acquisition.
    after_path : str
      A path to the closest weather radar image after the acquisition.

    Notes
    -----
    If a radar image before the date can not be found then the path to the one
    after will be returned twice and vice-verca.

    If an exact match is found for the given date, the same path will be
    returned twice.
    """
    paths = glob.glob(os.path.join(target_dir, '**/*.nc'),
                      recursive=True)

    dates = []
    for path in paths:
        filename = os.path.basename(path)
        dates += [datetime.strptime(filename, '%Y%m%d%H%M.nc')]

    time_deltas = [master_date - date for date in dates]
    shortest_delta_before = [d for d in sorted(time_deltas) if d.total_seconds() >= 0]
    shortest_delta_after = [d for d in sorted(time_deltas) if d.total_seconds() <= 0]

    wr_before_path = paths[time_deltas.index(shortest_delta_before[0])]
    wr_after_path = paths[time_deltas.index(shortest_delta_after[-1])]

    logging.info('Closest file before %s is %s', master_date, wr_before_path)
    logging.info('Closest file after %s is %s', master_date, wr_after_path)

    return (wr_before_path, wr_after_path)


def get_train_era_slant_dir():
    """Returns the path to the directory containing ERA slant delays calculated by
    TRAIN"""
    return os.path.join(config.SCRATCH_DIR, 'train', 'era_slant_delay')


# MAIN WORKFLOW STEPS
def execute_load_clip_resample_convert_step(args):

    """Execute the first step of the processing flow.

    This step finds all interferograms in the unwrapped interferogram directory
    whose dates satisfy the config dates. It then clips them (if enabled),
    resamples them (if enabled) and saves the (new) IFGs to the 'uifg_resampled'
    directory as `npy` files. This step also extracts the grid dimensions from
    the last processed interferogram and saves them to a file called 'grid.txt'
    in SCRATCH_DIR.

    """
    ifg_paths = find_ifgs()
    with Pool() as p:
        p.map(load_clip_resample, ifg_paths)

    return None

def execute_invert_unwrapped_phase(args):
    """Executes the second step of the processing flow.

    This steps takes the interferograms and inverts the timeseries relative to
    the master date. The time series is saved to the `uifg_ts` directory as a 3D
    npy array called MASTER_DATE.npy. Also saved is a metadata file containing
    the dates of each index along the third dimension of the array. A second
    file containing the perpendicular baselines for each inverted interferogram
    is also generated.

    Input interferograms are loaded from the uifg_resampled directory. Grid
    dimensions are loaded from 'grid.txt'.

    """
    os.makedirs(os.path.join(config.SCRATCH_DIR, 'uifg_ts'), exist_ok=True)
    output_file_base = os.path.join(config.SCRATCH_DIR, 'uifg_ts',
                                    config.MASTER_DATE.strftime('%Y%m%d'))
    output_file_name = output_file_base + '.npy'
    output_file_meta = output_file_base + '.yml'
    output_file_bperp = output_file_base + '_baselines' + '.txt'
    ifg_paths = glob.glob(os.path.join(config.SCRATCH_DIR,
                                       'uifg_resampled',
                                       '*.npy'))

    lons, lats = read_grid_from_file(os.path.join(config.SCRATCH_DIR, 'grid.txt'))
    grid_shape = (len(lats), len(lons))

    # Work out the shape of the output data.
    ifg_date_pairs = map(util.extract_timestamp_from_ifg_name, ifg_paths)
    nslcs = len(set([date for pair in ifg_date_pairs for date in pair]))
    logging.debug('Creating inversion output memory map')
    output_matrix = np.lib.format.open_memmap(output_file_name,
                                              'w+',
                                              shape=(grid_shape + (nslcs,)))
    logging.info('Starting inversion for master date %s', config.MASTER_DATE.strftime('%Y-%m-%d'))
    dates = inversion.calculate_inverse(ifg_paths, config.MASTER_DATE.date(), grid_shape, output_matrix)
    with open(output_file_meta, 'w') as f:
        yaml.dump(dates, f)

    # Calculate the new baselines and save them to a file.
    logging.info('Calculating perpendicular baselines')
    baseline_zip = inversion.calculate_inverse_bperp(config.BPERP_FILE_PATH,
                                                     config.MASTER_DATE.date())
    with open(output_file_bperp, 'w') as f:
        for (date, baseline) in baseline_zip:
            datestamp = date.strftime('%Y%m%d')
            f.write('{:s} {:.4f}\n'.format(datestamp, baseline))

def execute_calculate_dem_matmosphere_error(args):
    """Executes the third step of the processing flow.

    This step calculates the master atmosphere and DEM errors using the time
    series calculated in step 2. It loads the time series saved in
    'uifg_ts/MASTER_DATE.npy'. It saves the master atmosphere to
    'master_atmosphere/MASTER_DATE.npy'. It saves the DEM error to
    'dem_error/MASTER_DATE.npy'.

    """
    # Create output file paths
    os.makedirs(os.path.join(config.SCRATCH_DIR, 'master_atmosphere'),
                exist_ok=True)
    os.makedirs(os.path.join(config.SCRATCH_DIR, 'dem_error'),
                exist_ok=True)
    output_master_atmos_fname = os.path.join(config.SCRATCH_DIR,
                                             'master_atmosphere',
                                             config.MASTER_DATE.strftime('%Y%m%d') + '.npy')
    output_dem_error_fname = os.path.join(config.SCRATCH_DIR,
                                          'dem_error',
                                          config.MASTER_DATE.strftime('%Y%m%d') + '.npy')

    # Load required data
    ts_ifgs = np.load(os.path.join(config.SCRATCH_DIR,
                                   'uifg_ts',
                                   config.MASTER_DATE.strftime('%Y%m%d') + '.npy'),
                      mmap_mode='r')
    ts_dates = []
    with open(os.path.join(config.SCRATCH_DIR,
                           'uifg_ts',
                           config.MASTER_DATE.strftime('%Y%m%d') + '.yml')) as f:
        ts_dates = yaml.safe_load(f)

    ts_baselines = np.loadtxt(os.path.join(config.SCRATCH_DIR,
                                           'uifg_ts',
                                           config.MASTER_DATE.strftime('%Y%m%d') + '_baselines.txt'),
                              usecols=1)
    logging.info('Calculating DEM error and master atmosphere for %s',
                 config.MASTER_DATE.strftime('%Y-%m-%d'))
    master_atmosphere, dem_error = timeseries.calculate_dem_master_atmosphere(ts_ifgs,
                                                                              ts_baselines,
                                                                              ts_dates,
                                                                              config.MASTER_DATE.date())
    np.save(output_master_atmos_fname, master_atmosphere)
    np.save(output_dem_error_fname, dem_error)

def execute_master_atmosphere_rainfall_correlation(args):
    """Calculate the correlation coefficient between weather radar rainfall and
    the master atmosphere.

    This is an optional part of the processing workflow

    The correlation coefficient is printed to STDOUT.

    """
    # Load the master atmosphere ifg
    lons, lats = read_grid_from_file(os.path.join(config.SCRATCH_DIR, 'grid.txt'))
    ifg_data = np.load(os.path.join(config.SCRATCH_DIR,
                                    'master_atmosphere',
                                    config.MASTER_DATE.strftime('%Y%m%d') + '.npy'))
    ifg = {'lons': lons, 'lats': lats, 'data': ifg_data}

    # Load the corresponding weather radar images
    _, wr_after_path = find_closest_weather_radar_files(config.MASTER_DATE)
    wr_after = nimrod.load_from_netcdf(wr_after_path)

    # Clip images
    lon_bounds = (np.amin(lons), np.amax(lons))
    lat_bounds = (np.amin(lats), np.amax(lats))
    nimrod.clip_wr(wr_after, lon_bounds, lat_bounds)

    # Leggo
    (r_after, p_after) = nimrod.calc_wr_ifg_correlation(wr_after, ifg, rain_tol=args.rain_tolerance)
    print('Correlation Coefficient: {}, P-Value: {}'.format(r_after, p_after))



def execute_export_train(args):
    """Exports parts of the pysarts project needed for TRAIN.

    Parts exported are: IFG grid, ifg dates, UTC time of satellite pass.
    """
    lons, lats = read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                  'grid.txt'))

    region_res = max(abs(lons[-1] - lons[-2]), abs(lats[-1] - lats[-2]))

    lons, lats = np.meshgrid(lons, lats)
    ifg_dates = [util.extract_timestamp_from_ifg_name(name) for name in find_ifgs()]

    # Wrangle grid into TRAIN format
    ll_matfile_data = np.stack((lons.ravel(), lats.ravel()), axis=1)

    # Wrangle interferogram dates into TRAIN format. Train expects dates to be
    # represented as integers in YYYYMMDD format. They should be in an Nx2
    # matrix with master and slave dates in the first and second columns
    # respectively.
    ifg_numeric_dates = []
    for (master_date, slave_date) in ifg_dates:
        # Yuck
        master_date_numeric = int(master_date.strftime('%Y%m%d'))
        slave_date_numeric = int(slave_date.strftime('%Y%m%d'))
        ifg_numeric_dates += [[master_date_numeric, slave_date_numeric]]

    ifg_numeric_dates = np.array(ifg_numeric_dates)

    # Wrangle UTC time of satellite pass into TRAIN format.
    utc_sat_time = config.MASTER_DATE.strftime('%H:%M')

    # Calculate a region slightly larger than target region.
    lon_min, lon_max = config.REGION['lon_min'], config.REGION['lon_max']
    lat_min, lat_max = config.REGION['lat_min'], config.REGION['lat_max']

    # TODO: Implement a custom look angle file.
    # TODO: Implement a custom wavelength.

    # Create a setup script
    setup_script_contents = """% THIS FILE WAS AUTOMATICALLY GENERATED BY PYSARTS.
% *****DO NOT EDIT*****
% *****CREATE A FILE CALLED custom_setup.m FOR CUSTOM INITIALISATION*****
setparm_aps('ll_matfile', [pwd '/ll.mat'])
setparm_aps('ifgday_matfile', [pwd '/ifgday.mat'])
setparm_aps('UTC_sat', '{utctime:s}')
setparm_aps('demfile', '{demfile:s}')
setparm_aps('region_res', {res:f})
setparm_aps('region_lon_range', [{lon_min:f} {lon_max:f}])
setparm_aps('region_lat_range', [{lat_min:f} {lat_max:f}])
if exist('custom_setup.m', 'file') == 2
  custom_setup()
end
""".format(utctime=utc_sat_time,
           demfile=os.path.abspath(config.DEM_PATH),
           res=region_res,
           lon_min=lon_min,
           lon_max=lon_max,
           lat_min=lat_min,
           lat_max=lat_max)

    # Save output files
    output_directory = os.path.join(config.SCRATCH_DIR, 'train')
    os.makedirs(output_directory, exist_ok=True)
    scpio.savemat(os.path.join(output_directory, 'll.mat'),
                  {'lonlat': ll_matfile_data})
    scpio.savemat(os.path.join(output_directory, 'ifgday.mat'),
                  {'ifgday': ifg_numeric_dates})

    with open(os.path.join(output_directory, 'setup.m'), 'w') as f:
        f.write(setup_script_contents)

def execute_calculate_zenith_delays(args):
    """For the master date, calculates the one-way zenith delay produced by the
    wet and dry components of the atmosphere. Outputs are saved into
    SCRATCH_DIR/zenith_delay folder with the names DATE_dry.npy, DATE_wet.npy
    and DATE_total.npy.

    This stage uses two weather models, one before and one after the acquisition
    time. The final delay is weighted according to which model is closest.
    """
    # Load the DEM for the target region.
    logging.info('Loading DEM')
    dem = util.load_dem(config.DEM_PATH)

    # Handle NaNs in the DEM
    dem['data'].fill_value = 0
    dem['data'] = dem['data'].filled()

    # Load weather models before and after the acquisition time.
    logging.info('Loading weather models')
    before_path, after_path = find_closest_weather_radar_files(config.MASTER_DATE,
                                                               config.ERA_MODELS_PATH)
    before_model = ERAModel.load_era_netcdf(before_path)
    after_model = ERAModel.load_era_netcdf(after_path)

    # Get some information about the region
    lons, lats = read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                  'grid.txt'))
    lon_min, lon_max = np.amin(lons), np.amax(lons)
    lat_min, lat_max = np.amin(lats), np.amax(lats)

    # Make these slightly larger than the target region.
    lon_bounds = (lon_min - 0.5, lon_max + 0.5)
    lat_bounds = (lat_min - 0.5, lat_max + 0.5)

    # Clip and resample DEM onto master grid.
    logging.info('Clipping and resampling DEM to target region')
    processing.clip_ifg(dem, lon_bounds, lat_bounds)
    processing._resample_ifg(dem, lons, lats)

    # Clip and resample weather model onto master grid.
    logging.info('Clipping and resampling weather models to target region')
    before_model.clip(lon_bounds, lat_bounds)
    after_model.clip(lon_bounds, lat_bounds)
    before_model.resample(lons, lats)
    after_model.resample(lons, lats)

    # Calculate the delay for models before and after the acquisition.
    logging.info('Calculating zenith delay for model before acquisition')
    delay_before = corrections.calculate_era_zenith_delay(before_model, dem)
    logging.info('Calculating zenith delay for model after acquisition')
    delay_after = corrections.calculate_era_zenith_delay(after_model, dem)

    # Calculate the weighting for the two models
    time_diff = (after_model.date - before_model.date).total_seconds()
    before_diff = (config.MASTER_DATE - before_model.date).total_seconds()
    before_weight = 1 - (before_diff / time_diff)

    delay = {}
    delay['total'] = (before_weight * delay_before['data']
                      + (1 - before_weight) * delay_after['data'])
    delay['wet'] = (before_weight * delay_before['wet_delay']
                    + (1 - before_weight) * delay_after['wet_delay'])
    delay['dry'] = (before_weight * delay_before['dry_delay']
                    + (1 - before_weight) * delay_after['dry_delay'])

    # Create output paths
    output_base = os.path.join(config.SCRATCH_DIR,
                               'zenith_delays')
    os.makedirs(output_base, exist_ok=True)
    output_dry = os.path.join(output_base,
                              (config.MASTER_DATE.strftime('%Y%m%d')
                               + '_dry.npy'))
    output_wet = os.path.join(output_base,
                              (config.MASTER_DATE.strftime('%Y%m%d')
                               + '_wet.npy'))
    output_total = os.path.join(output_base,
                                (config.MASTER_DATE.strftime('%Y%m%d')
                                 + '_total.npy'))

    # Save out the results
    np.save(output_dry, delay['dry'])
    np.save(output_wet, delay['wet'])
    np.save(output_total, delay['total'])
