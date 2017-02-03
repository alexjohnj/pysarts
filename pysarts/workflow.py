"""Functions that make up the workflow of pysarts when it is run as a program.

Almost all of these functions have side-effects.
"""

import logging
import os

import numpy as np

from . import config
from . import processing

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
        ifg = load_clip_resample(path)
        output_dir = os.path.join(config.SCRATCH_DIR, 'uifg_resampled')
        save_ifg_to_npy(ifg, output_dir)

    # Save grid details
    extract_grid_from_ifg(ifg, os.path.join(config.SCRATCH_DIR, 'grid.txt'))

    return None
