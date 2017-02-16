"""
Module dealing with weather radar images.
"""
from datetime import datetime
import logging
import os.path

from netCDF4 import Dataset
import numpy as np
import scipy.interpolate as interp
import scipy.stats as stats

from . import processing

def load_from_netcdf(path):
    """Load a nimrod dictionary from a netcdf file.

    Arguments
    ---------
    path : str
      The path to the weather radar image.

    Returns
    -------
    A nimrod dictionary with the keys 'lons', 'lats', 'data', 'date'. The 'date'
    key maps onto a `datetime` object.

    """
    date = datetime.strptime(os.path.splitext(os.path.basename(path))[0], '%Y%m%d%H%M')
    lons = np.array([])
    lats = np.array([])
    data = np.array([])

    with Dataset(path) as radar_cdf:
        lons = radar_cdf.variables['lon'][:]
        lats = radar_cdf.variables['lat'][:]
        data = radar_cdf.variables['z'][:].data

    return {'lons': lons, 'lats': lats, 'data': data, 'date': date}

def clip_wr(wr, lon_bounds, lat_bounds):
    """Clip a weather radar image to a specified region.

    See the documentation for `processing.clip_ifg` (which this wraps).
    """
    logging.debug('Clipping weather radar image')
    processing.clip_ifg(wr, lon_bounds, lat_bounds)

def resample_wr(wr, new_lons, new_lats):
    """Resample a weather radar image onto a new grid using a RectBivariateSpline.

    Arguments
    ---------
    wr : dict
      A weather radar dict with the keys 'lats', 'lons', 'data'.
    new_lons : ndarray
      A 1D array of longitudes to resample the image at.
    new_lats : ndarray
      A 1D array of latitudes to resample the image at.

    Returns
    -------
    A copy of `wr` resampled at the new grid points.
    """
    logging.debug('Resampling weather radar image')
    wr_new = wr.copy()
    processing._resample_ifg(wr_new, new_lons, new_lats)
    logging.debug('Resampled weather radar image')

    return wr_new

def calc_wr_ifg_correlation(wr, ifg, rain_tol=0):
    """Calculates the correlation coefficient between a weather radar image and
    an interferogram.

    Arguments
    ---------
    wr : dict
      A weather radar dictionary with the keys 'data', 'lons', 'lats' mapping onto `ndrray`s.
    ifg : dict
      An interferogram dictionary with the keys 'data', 'lons', 'lats' mapping
      onto `ndarray`s.
    rain_tol : float
      Pixels with rainfall less than or equal to rain_tol won't be included in
      the calculation.

    Returns
    -------
    r : float
      The correlation coefficient
    p : float
      2-tailed p-value.

    Notes
    -----
    The interferogram will be resampled to match the resolution of the weather radar image.
    """
    if wr['lons'].size != ifg['lons'].size or wr['lats'].size != ifg['lats'].size:
        ifg = resample_wr(ifg, wr['lons'], wr['lats']) # Confusing, I know, but it works


    # Filter pixels with rainfall below tolerance
    wr_below_tol_idxs = np.where(wr['data'].ravel() > rain_tol)
    wr_data = wr['data'].ravel()[wr_below_tol_idxs]
    ifg_data = ifg['data'].ravel()[wr_below_tol_idxs]

    # Filter pixels outside of interferogram
    ifg_zero_idxs = np.where(np.logical_not(np.isclose(0, ifg_data)))
    ifg_data = ifg_data[ifg_zero_idxs]
    wr_data = wr_data[ifg_zero_idxs]

    logging.debug('Calculating correlation coefficient')

    return stats.pearsonr(wr_data, ifg_data)
