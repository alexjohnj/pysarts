"""Functions that make up the workflow of pysarts when it is run as a program.

Almost all of these functions have side-effects.
"""

import logging
import os
from datetime import datetime
import glob

import numpy as np
import yaml
import scipy.io as scpio

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

    return ifg

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

def find_closest_weather_radar_file(master_date):
    """Looks in the weather radar directory for the weather radar image closest
    to the master date.

    Returns the path to the closest radar image.
    """
    paths = glob.glob(os.path.join(config.WEATHER_RADAR_DIR, '**/*.nc'),
                      recursive=True)

    dates = []
    for path in paths:
        filename = os.path.basename(path)
        dates += [datetime.strptime(filename, '%Y%m%d%H%M.nc')]

    time_deltas = [abs(master_date - date) for date in dates]
    sorted_time_deltas = sorted(time_deltas)

    logging.info('Closest weather radar image: %s', paths[time_deltas.index(sorted_time_deltas[0])])
    return paths[time_deltas.index(sorted_time_deltas[0])]

### MAIN WORKFLOW STEPS
def execute_load_clip_resample_convert_step():
    """Execute the first step of the processing flow.

    This step finds all interferograms in the unwrapped interferogram directory
    whose dates satisfy the config dates. It then clips them (if enabled),
    resamples them (if enabled) and saves the (new) IFGs to the 'uifg_resampled'
    directory as `npy` files. This step also extracts the grid dimensions from
    the last processed interferogram and saves them to a file called 'grid.txt'
    in SCRATCH_DIR.

    """
    ifg_paths = find_ifgs()
    ifg = {}
    for path in ifg_paths:
        logging.info('Processing %s', os.path.splitext(os.path.basename(path))[0])
        ifg = load_clip_resample(path)
        output_dir = os.path.join(config.SCRATCH_DIR, 'uifg_resampled')
        save_ifg_to_npy(ifg, output_dir)

    # Save grid details
    logging.info('Extracting grid')
    extract_grid_from_ifg(ifg, os.path.join(config.SCRATCH_DIR, 'grid.txt'))

    return None

def execute_invert_unwrapped_phase():
    """Executes the second step of the processing flow.

    This steps takes the interferograms and inverts the timeseries relative to
    the master date. The time series is saved to the `uifg_ts` directory as a 3D
    npy array called MASTER_DATE.npy. Also saved is a metadata file containing
    the dates of each index along the third dimension of the array.

    Input interferograms are loaded from the uifg_resampled directory. Grid
    dimensions are loaded from 'grid.txt'.

    """
    os.makedirs(os.path.join(config.SCRATCH_DIR, 'uifg_ts'), exist_ok=True)
    output_file_base = os.path.join(config.SCRATCH_DIR, 'uifg_ts',
                                    config.MASTER_DATE.strftime('%Y%m%d'))
    output_file_name = output_file_base + '.npy'
    output_file_meta = output_file_base + '.yml'
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

def execute_calculate_master_atmosphere():
    """Executes the third step of the processing flow.

    This step calculates the master atmosphere using the time series calculated
    in step 2. It loads the time series saved in 'uifg_ts/MASTER_DATE.npy' and
    saves the master atmosphere to 'master_atmosphere/MASTER_DATE.npy'.

    """
    os.makedirs(os.path.join(config.SCRATCH_DIR, 'master_atmosphere'), exist_ok=True)
    output_file_name = os.path.join(config.SCRATCH_DIR,
                                    'master_atmosphere',
                                    config.MASTER_DATE.strftime('%Y%m%d') + '.npy')

    ts_ifgs = np.load(os.path.join(config.SCRATCH_DIR,
                                   'uifg_ts',
                                   config.MASTER_DATE.strftime('%Y%m%d') + '.npy'),
                      mmap_mode='r')
    ts_dates = []
    with open(os.path.join(config.SCRATCH_DIR,
                           'uifg_ts',
                           config.MASTER_DATE.strftime('%Y%m%d') + '.yml')) as f:
        ts_dates = yaml.safe_load(f)

    logging.info('Starting master atmosphere inversion for %s', config.MASTER_DATE.strftime('%Y-%m-%d'))
    master_atmosphere = timeseries.calculate_master_atmosphere(ts_ifgs, ts_dates,
                                                               config.MASTER_DATE.date())
    np.save(output_file_name, master_atmosphere)

def execute_master_atmosphere_rainfall_correlation():
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

    # Load the corresponding weather radar image
    closest_wrimage_path = find_closest_weather_radar_file(config.MASTER_DATE)
    wr = nimrod.load_from_netcdf(closest_wrimage_path)
    lon_bounds = (np.amin(lons), np.amax(lons))
    lat_bounds = (np.amin(lats), np.amax(lats))
    nimrod.clip_wr(wr, lon_bounds, lat_bounds)

    (r, p) = nimrod.calc_wr_ifg_correlation(wr, ifg, rain_tol=1)
    print('Correlation Coefficient: {}, P-Value: {}'.format(r, p))


def execute_export_train():
    """Exports parts of the pysarts project needed for TRAIN.

    Parts exported are: IFG grid, ifg dates, UTC time of satellite pass.
    """
    lons, lats = read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                  'grid.txt'))
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

    # Save output files
    output_directory = os.path.join(config.SCRATCH_DIR, 'train')
    os.makedirs(output_directory, exist_ok=True)
    scpio.savemat(os.path.join(output_directory, 'll.mat'),
                  {'lonlat': ll_matfile_data})
    scpio.savemat(os.path.join(output_directory, 'ifgday.mat'),
                  {'ifgday': ifg_numeric_dates})

    with open(os.path.join(output_directory, 'utc_time.txt'), 'w') as f:
        f.write(utc_sat_time)
