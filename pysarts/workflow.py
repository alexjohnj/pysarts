"""Functions that make up the workflow of pysarts when it is run as a program.

Almost all of these functions have side-effects.
"""
import logging
import os
import shutil
from datetime import datetime
import glob
from multiprocessing.pool import Pool
from netCDF4 import Dataset

import numpy as np
import yaml
import scipy.io as scpio
from scipy.ndimage import gaussian_filter

from .geogrid import GeoGrid
from .era import ERAModel
from . import corrections
from . import config
from . import inversion
from . import nimrod
from . import insar
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
    return insar.find_ifgs_for_dates(config.UIFG_DIR, config.MASTER_DATE.date(), config.DATES)

def load_clip_resample(path):
    """Loads, clips and resamples an interferogram.

    Clipping and resampling is based on the user's configuration.

    """
    SHOULD_CLIP = config.REGION != None
    SHOULD_RESAMPLE = config.RESOLUTION != None

    logging.info('Processing %s', os.path.splitext(os.path.basename(path))[0])
    logging.debug('Loading %s', path)
    ifg = insar.InSAR.from_netcdf(path)

    if SHOULD_CLIP:
        logging.debug('Clipping')
        lon_bounds = (config.REGION['lon_min'], config.REGION['lon_max'])
        lat_bounds = (config.REGION['lat_min'], config.REGION['lat_max'])
        ifg.clip(lon_bounds, lat_bounds)
    else:
        logging.debug('Clipping disabled')

    if SHOULD_RESAMPLE:
        logging.debug('Resampling')
        ifg.interp_at_res(config.RESOLUTION['delta_x'],
                          config.RESOLUTION['delta_y'])
    else:
        logging.debug('Resampling disabled')

    output_dir = os.path.join(config.SCRATCH_DIR, 'uifg_resampled')
    save_ifg_to_npy(ifg, output_dir)

    logging.info('Extracting grid from %s',
                 os.path.splitext(os.path.basename(path))[0])
    extract_grid_from_ifg(ifg, os.path.join(config.SCRATCH_DIR, 'grid.txt'))

    return None

def save_ifg_to_npy(ifg, out_dir):
    """Saves an insar.InSAR instance as a Numpy array.

    The file name will be SLAVEDATE_MASTERDATE.npy
    """
    out_name = (ifg.slave_date.strftime('%Y%m%d') + '_'
                + ifg.master_date.strftime('%Y%m%d')
                + '.npy')
    out_path = os.path.join(out_dir, out_name)
    os.makedirs(out_dir, exist_ok=True)
    logging.debug('Saving %s', out_path)

    # MaskedArrays can't be saved yet.
    if isinstance(ifg.data, np.ma.MaskedArray):
        np.save(out_path, ifg.data)
    else:
        np.save(out_path, ifg.data)

    return None


def extract_grid_from_ifg(ifg, output_name):
    """Extracts information necessary to reconstruct the grid and saves it to a
    file.

    The output format is a 2 line plain text file:
       lon=lon_min:lon_max:nlon
       lat=lat_min:lat_max:nlat
    """
    lon_min = np.amin(ifg.lons)
    lon_max = np.amax(ifg.lons)
    nlon = ifg.lons.size
    lat_min = np.amin(ifg.lats)
    lat_max = np.amax(ifg.lats)
    nlat = ifg.lats.size

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


def find_closest_weather_radar_files(master_date, target_dir=None):
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
    if target_dir is None:
        target_dir = config.WEATHER_RADAR_DIR

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

    logging.debug('Closest file before %s is %s', master_date, wr_before_path)
    logging.debug('Closest file after %s is %s', master_date, wr_after_path)

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
    # Load the master atmosphere SAR image
    lons, lats = read_grid_from_file(os.path.join(config.SCRATCH_DIR, 'grid.txt'))
    sar_data = np.load(os.path.join(config.SCRATCH_DIR,
                                    'master_atmosphere',
                                    config.MASTER_DATE.strftime('%Y%m%d') + '.npy'))
    sar = insar.SAR(lons,
                    lats,
                    sar_data,
                    config.MASTER_DATE)

    # Load the corresponding weather radar images
    _, wr_after_path = find_closest_weather_radar_files(config.MASTER_DATE)
    wr_after = nimrod.Nimrod.from_netcdf(wr_after_path)

    # Clip images and resample radar image
    lon_bounds = (np.amin(lons), np.amax(lons))
    lat_bounds = (np.amin(lats), np.amax(lats))
    wr_after.clip(lon_bounds, lat_bounds)
    wr_after.interp(sar.lons, sar.lats, method='nearest')

    # Leggo
    (r_after, p_after) = nimrod.calc_wr_ifg_correlation(wr_after, sar, rain_tol=args.rain_tolerance)
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


def execute_calculate_era_delays(args):
    """For all dates calculates the one-way zenith delay produced by the wet and dry
    components of the atmosphere. Outputs are saved into
    SCRATCH_DIR/zenith_delay folder with the names DATE_dry.npy, DATE_wet.npy
    and DATE_total.npy.

    This stage uses two weather models, one before and one after the acquisition
    time. The final delay is weighted according to which model is closest.

    """
    # Load the DEM for the target region.
    logging.info('Loading DEM')
    dem = GeoGrid.from_netcdf(config.DEM_PATH)

    # Handle NaNs in the DEM
    dem.data.fill_value = 0
    dem.data = dem.data.filled()

    # Get some information about the region
    lons, lats = read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                  'grid.txt'))
    lon_min, lon_max = np.amin(lons), np.amax(lons)
    lat_min, lat_max = np.amin(lats), np.amax(lats)

    # Make these slightly larger than the target region.
    lon_bounds = (lon_min - 0.5, lon_max + 0.5)
    lat_bounds = (lat_min - 0.5, lat_max + 0.5)

    # Clip and resample the DEM to the target region
    logging.info('Clipping and resampling DEM to target region')
    dem.clip(lon_bounds, lat_bounds)
    dem.interp(lons, lats)

    # Parse dates passed as command line arguments or load in all dates if none
    # were passed
    dates = []
    if args.dates is None:
        date_file = os.path.join(config.SCRATCH_DIR,
                                 'uifg_ts',
                                 config.MASTER_DATE.strftime('%Y%m%d') + '.yml')
        dates = []
        with open(date_file) as f:
            dates = yaml.safe_load(f)
    else:
        dates = [datetime.strptime(date, '%Y%m%d') for date in args.dates]

    # Build a list of arguments to pass to the parallel helper.
    helper_args = []
    for date in dates:
        datestamp = datetime(date.year, date.month, date.day,
                             config.MASTER_DATE.hour,
                             config.MASTER_DATE.minute)
        helper_args += [(datestamp, dem, lon_bounds, lat_bounds, args.rainfall, args.blur)]

    # Increase the number of processes if you've got enough memory
    with Pool(args.max_processes) as p:
        p.starmap(_parallel_era_delay, helper_args)


def _parallel_era_delay(date, dem, lon_bounds, lat_bounds, plevels=None,
                        filter_std=0):
    """Helper for zenith delay calculation step.

    Used to run the main calculation in parallel.

    Arguments
    ---------
    date : datetime.datetime
      The date of the weather model.
    dem : dict
      DEM dictionary for the region.
    lon_bounds : 2-tuple
      Region longitude bounds
    lat_bounds : 2-tuple
      Region latitude bounds
    plevels : 2-tuple, opt
      The maximum and minimum pressure levels to use in a rainfall enhanced
      correction. Pass `None` to disable the rainfall correction.
    filter_std : float, opt
      The standard deviation of the Gaussian filter applied to the relative
      humidity after modification for rainfall. Default value is 0 which
      effectively means no filter.
    """
    # Load weather models before and after the acquisition time.
    logging.info('Loading weather models for date %s',
                 date.strftime('%Y-%m-%d'))
    before_path, after_path = find_closest_weather_radar_files(date,
                                                               config.ERA_MODELS_PATH)
    before_model = ERAModel.load_era_netcdf(before_path)
    after_model = ERAModel.load_era_netcdf(after_path)

    # Clip and resample weather model onto master grid.
    logging.info('Resampling weather models for date %s',
                 date.strftime('%Y-%m-%d'))
    before_model.clip(lon_bounds, lat_bounds)
    after_model.clip(lon_bounds, lat_bounds)
    before_model.resample(dem.lons, dem.lats)
    after_model.resample(dem.lons, dem.lats)

    # Incorporate rainfall data if needed
    if plevels is not None:
        _, wr_after_path = find_closest_weather_radar_files(date)
        wr = nimrod.Nimrod.from_netcdf(wr_after_path)
        wr.clip(lon_bounds, lat_bounds)
        wr.interp(dem.lons, dem.lats, method='nearest')

        pmin, pmax = min(plevels), max(plevels)
        after_model.add_rainfall(wr.data, pmin, pmax, filter_std)

    # Calculate the delay for models before and after the acquisition.
    logging.info('Calculating zenith delay for model %s',
                 before_model.date.strftime('%Y-%m-%d'))
    wet_before, dry_before, total_before = corrections.calculate_era_zenith_delay(before_model, dem)
    logging.info('Calculating zenith delay for model %s',
                 after_model.date.strftime('%Y-%m-%d'))
    wet_after, dry_after, total_after = corrections.calculate_era_zenith_delay(after_model, dem)

    # Apply time weighting
    wet_delay = insar.SAR.interpolate(wet_before, wet_after, date)
    dry_delay = insar.SAR.interpolate(dry_before, dry_after, date)
    total_delay = insar.SAR.interpolate(total_before, total_after, date)

    # Create output paths
    output_base = os.path.join(config.SCRATCH_DIR,
                               'zenith_delays')
    os.makedirs(output_base, exist_ok=True)
    output_dry = os.path.join(output_base,
                              (date.strftime('%Y%m%d')
                               + '_dry.npy'))
    output_wet = os.path.join(output_base,
                              (date.strftime('%Y%m%d')
                               + '_wet.npy'))
    output_total = os.path.join(output_base,
                                (date.strftime('%Y%m%d')
                                 + '_total.npy'))

    # Save out the results
    np.save(output_dry, dry_delay.data)
    np.save(output_wet, wet_delay.data)
    np.save(output_total, total_delay.data)

    # Calculate the slant delay
    output_base = os.path.join(config.SCRATCH_DIR,
                               'slant_delays')
    os.makedirs(output_base, exist_ok=True)
    output_wet = os.path.join(output_base,
                              date.strftime('%Y%m%d') + '_wet.npy')
    output_dry = output_wet.replace('_wet', '_dry')
    output_total = output_wet.replace('_wet', '_total')

    slant_wet = wet_delay.zenith2slant(np.deg2rad(21))
    slant_dry = dry_delay.zenith2slant(np.deg2rad(21))
    slant_total = total_delay.zenith2slant(np.deg2rad(21))

    np.save(output_wet, slant_wet.data)
    np.save(output_dry, slant_dry.data)
    np.save(output_total, slant_total.data)


def execute_calculate_ifg_delays(args):
    """Calculates interferometric slant delays for all pairings in the bperp
    file."""
    # Load perpendicular baselines
    bperp_contents = inversion.read_bperp_file(config.BPERP_FILE_PATH)

    helper_args = []
    for (master_date, slave_date, _) in bperp_contents:
        helper_args += [(master_date, slave_date)]

    os.makedirs(os.path.join(config.SCRATCH_DIR, 'insar_atmos_delays'),
                exist_ok=True)

    with Pool() as p:
        p.starmap(_execute_calculate_ifg_delays, helper_args)


def _execute_calculate_ifg_delays(master_date, slave_date):
    slant_delay_dir = os.path.join(config.SCRATCH_DIR, 'slant_delays')
    try:
        master_delay = np.load(os.path.join(slant_delay_dir,
                                            (master_date.strftime('%Y%m%d')
                                             + '_total.npy')))
        slave_delay = np.load(os.path.join(slant_delay_dir,
                                           (slave_date.strftime('%Y%m%d') +
                                            '_total.npy')))
    except FileNotFoundError:
        logging.warning('No correction found for %s / %s pairing, skipping',
                        master_date, slave_date)
        return

    ifg_delay = corrections.calc_ifg_delay(master_delay, slave_delay)

    output_dir = os.path.join(config.SCRATCH_DIR, 'insar_atmos_delays')
    output_file_name = (slave_date.strftime('%Y%m%d') + '_' +
                        master_date.strftime('%Y%m%d') + '.npy')
    output_path = os.path.join(output_dir, output_file_name)

    np.save(output_path, ifg_delay)


def execute_calculate_liquid_delay(args):
    cloud_thickness = args.cloud_thickness
    date = args.date or config.MASTER_DATE
    if isinstance(date, str):
        date = datetime.strptime(args.date, '%Y%m%d')
        date = datetime(date.year, date.month, date.day,
                        config.MASTER_DATE.hour, config.MASTER_DATE.minute)

    # Load grid
    lons, lats = read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                  'grid.txt'))

    # Load radar images for target date
    logging.debug('Loading radar image for %s', date.strftime('%Y-%m-%d'))
    _, wr_path = find_closest_weather_radar_files(date)
    wr = nimrod.Nimrod.from_netcdf(wr_path)

    lon_min, lon_max = np.amin(lons), np.amax(lons)
    lat_min, lat_max = np.amin(lats), np.amax(lats)

    # Make these slightly larger than the target region.
    lon_bounds = (lon_min - 0.5, lon_max + 0.5)
    lat_bounds = (lat_min - 0.5, lat_max + 0.5)

    # Clip and resample radar image
    wr.clip(lon_bounds, lat_bounds)
    wr.interp(lons, lats, method='nearest')
    wr.data = gaussian_filter(wr.data, args.blur)

    # Calculate the liquid water content and delay
    lwc = wr.lwc()
    zenith_liquid = corrections.liquid_zenith_delay(lwc, cloud_thickness)
    slant_liquid = zenith_liquid.zenith2slant(np.deg2rad(21))

    # Save the slant delay and the liquid water content
    zenith_output_dir = os.path.join(config.SCRATCH_DIR, 'zenith_delays')
    slant_output_dir = os.path.join(config.SCRATCH_DIR, 'slant_delays')
    lwc_output_dir = os.path.join(config.SCRATCH_DIR, 'lwc')
    delay_output_name = date.strftime('%Y%m%d') + '_liquid' + '.npy'
    lwc_output_name = date.strftime('%Y%m%d') + '.npy'

    zenith_output_path = os.path.join(zenith_output_dir, delay_output_name)
    slant_output_path = os.path.join(slant_output_dir, delay_output_name)
    lwc_output_path = os.path.join(lwc_output_dir, lwc_output_name)

    os.makedirs(zenith_output_dir, exist_ok=True)
    os.makedirs(slant_output_dir, exist_ok=True)
    os.makedirs(lwc_output_dir, exist_ok=True)

    np.save(zenith_output_path, zenith_liquid.data)
    np.save(slant_output_path, slant_liquid.data)
    np.save(lwc_output_path, lwc.data)

    # If it exists, recalculate the total delay so it includes the liquid delay
    total_zenith_delay_path = zenith_output_path.replace('_liquid', '_total')
    total_slant_delay_path = slant_output_path.replace('_liquid', '_total')

    if os.path.exists(total_zenith_delay_path):
        logging.info('Updating total zenith delay for %s',
                     date.strftime('%Y-%m-%d'))
        zenith_wet = np.load(zenith_output_path.replace('_liquid', '_wet'))
        zenith_dry = np.load(zenith_output_path.replace('_liquid', '_dry'))
        total_delay = zenith_wet + zenith_dry + zenith_liquid.data
        np.save(total_zenith_delay_path, total_delay)

    if os.path.exists(total_slant_delay_path):
        logging.info('Updating total slant delay for %s',
                     date.strftime('%Y-%m-%d'))
        slant_wet = np.load(slant_output_path.replace('_liquid', '_wet'))
        slant_dry = np.load(slant_output_path.replace('_liquid', '_dry'))
        total_delay = slant_wet + slant_dry + slant_liquid.data
        np.save(total_slant_delay_path, total_delay)


def execute_correction_step(args):
    delay_dir = os.path.join(config.SCRATCH_DIR, 'slant_delays')
    # Set up the output directory
    output_dir = os.path.join(config.SCRATCH_DIR, 'corrected_ifg')
    os.makedirs(output_dir, exist_ok=True)

    # Flags for the final NetCDF's history
    LIQUID_CORRECTED = False
    WET_CORRECTED = False
    DRY_CORRECTED = False

    # Get a list of interferograms to make the correction for.
    bperp_contents = inversion.read_bperp_file(config.BPERP_FILE_PATH)

    # Load the grid
    lons, lats = read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                  'grid.txt'))

    # Let's go
    for (master_date, slave_date, _) in bperp_contents:
        if (args.dates and (master_date.strftime('%Y%m%d') not in args.dates or
                            slave_date.strftime('%Y%m%d') not in args.dates)):
            continue

        logging.info('Processing %s / %s', master_date, slave_date)
        master_base = os.path.join(delay_dir, master_date.strftime('%Y%m%d'))
        slave_base = os.path.join(delay_dir, slave_date.strftime('%Y%m%d'))

        # Load delays for master and slave dates
        master_delay = np.zeros((lats.size, lons.size))
        slave_delay = np.zeros((lats.size, lons.size))
        if args.total:
            master_delay = np.load(master_base + '_total.npy')
            slave_delay = np.load(slave_base + '_total.npy')

            WET_CORRECTED = True
            DRY_CORRECTED = True
            if (os.path.exists(master_base + '_liquid.npy') or
                os.path.exists(slave_base + '_liquid.npy')):
                LIQUID_CORRECTED = True
        else:
            if args.liquid:
                try:
                    master_delay += np.load(master_base + '_liquid.npy')
                    LIQUID_CORRECTED = True
                except FileNotFoundError:
                    logging.info('No liquid delay for %s', master_date)
                try:
                    slave_delay += np.load(slave_base + '_liquid.npy')
                    LIQUID_CORRECTED = True
                except FileNotFoundError:
                    logging.info('No liquid delay for %s', slave_date)
            if args.dry:
                master_delay += np.load(master_base + '_dry.npy')
                slave_delay += np.load(slave_base + '_dry.npy')
                DRY_CORRECTED = True
            if args.wet:
                master_delay += np.load(master_base + '_wet.npy')
                slave_delay += np.load(slave_base + '_wet.npy')
                WET_CORRECTED = True

        # Calculate interferometric delay
        insar_delay = master_delay - slave_delay

        # Load the original interferogram and apply the correction
        original_path = os.path.join(config.SCRATCH_DIR, 'uifg_resampled',
                                     slave_date.strftime('%Y%m%d') + '_' +
                                     master_date.strftime('%Y%m%d') + '.npy')
        ifg = np.load(original_path)
        ifg = np.ma.masked_values(ifg, 0)
        ifg = ifg - insar_delay

        # Save as a NetCDF
        ifg = insar.InSAR(lons, lats, ifg, master_date, slave_date)
        output_path = os.path.join(output_dir, slave_date.strftime('%Y%m%d')
                                   + '_' + master_date.strftime('%Y%m%d')
                                   + '.nc')
        history_str = 'Processing by pysarts:'
        if config.REGION:
            history_str += ' clipped, '
        if config.RESOLUTION:
            history_str += ' resampled, '
        if DRY_CORRECTED:
            history_str += ' hydrostatic correction from era-interim, '
        if WET_CORRECTED:
            history_str += ' wet correction from era-interim, '
        if LIQUID_CORRECTED:
            history_str += ' liquid correction from radar rainfall, '

        history_str = history_str.strip().strip(',')
        ifg.save_netcdf(output_path, history_str)


def execute_clean_step(args):
    """Removes files generated by pysarts."""
    print("WARNING")
    print("You are about to irreversibly delete all files generated by pysarts!")
    print("This can not be undone!")
    print("Raw data will not be touched")
    yn = input('Continue [Y/n] >> ').lower()
    if yn == 'n':
        print('Aborting')
        exit(0)

    dirs = []
    dirs += [os.path.join(config.SCRATCH_DIR,
                          'uifg_resampled')]
    dirs += [os.path.join(config.SCRATCH_DIR,
                          'uifg_ts')]
    dirs += [os.path.join(config.SCRATCH_DIR,
                          'master_atmosphere')]
    dirs += [os.path.join(config.SCRATCH_DIR,
                          'dem_error')]
    dirs += [os.path.join(config.SCRATCH_DIR,
                          'slant_delays')]
    dirs += [os.path.join(config.SCRATCH_DIR,
                          'zenith_delays')]
    dirs += [os.path.join(config.SCRATCH_DIR,
                          'insar_atmos_delays')]
    dirs += [os.path.join(config.SCRATCH_DIR,
                          'lwc')]

    for dir in dirs:
        try:
            shutil.rmtree(dir)
        except FileNotFoundError:
            pass

    # Remove grid file
    try:
        os.remove(os.path.join(config.SCRATCH_DIR, 'grid.txt'))
    except FileNotFoundError:
        pass
