"""
Module dealing with weather radar images.
"""
from datetime import datetime
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
    processing.clip_ifg(wr, lon_bounds, lat_bounds)


def resample_wr(wr, new_lons, new_lats):
    """Resample a weather radar image onto a new grid using a nearest neighbour
    algorithm.

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
    wr_new = wr.copy()
    wr_new['lons'] = new_lons
    wr_new['lats'] = new_lats

    # Generate lat, lon pairings of coordinates
    pixel_coords = np.array(np.meshgrid(wr['lats'], wr['lons'])).T.reshape(-1, 2)
    new_pixel_coords = np.array(np.meshgrid(new_lats, new_lons)).T.reshape(-1, 2)
    wr_new['data'] = interp.griddata(pixel_coords,
                                     wr['data'].ravel(),
                                     new_pixel_coords,
                                     method='nearest')
    wr_new['data'].shape = (len(new_lats), len(new_lons))

    return wr_new


def calc_wr_ifg_correlation(wr, ifg, rain_tol=0):
    """Calculates the correlation coefficient between a weather radar image and an
    interferogram.

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
    The weather radar image will be resampled to match the resolution of the
    interferogram.

    """
    if wr['data'].size != ifg['data'].size:
        wr = resample_wr(wr, ifg['lons'], ifg['lats'])

    # Filter pixels with rainfall below tolerance
    wr_below_tol_idxs = np.where(wr['data'].ravel() > rain_tol)
    wr_data = wr['data'].ravel()[wr_below_tol_idxs]
    ifg_data = ifg['data'].ravel()[wr_below_tol_idxs]

    # Filter pixels outside of interferogram
    ifg_zero_idxs = np.where(np.logical_not(np.isclose(0, ifg_data)))
    ifg_data = ifg_data[ifg_zero_idxs]
    wr_data = wr_data[ifg_zero_idxs]


    return stats.pearsonr(wr_data, ifg_data)


def interp_radar(wr_before, wr_after, idate):
    """Interpolate between two radar images at a given time.

    Not really an interpolation, just a weighted sum of the two images.

    Arguments
    ---------
    wr_before : dict
      A dictionary with the keys `data` and `date` where `date` is before
      `idate`. `data` is an NxM ndarray.
    wr_after : dict
      A dictionary with the keys `data` and `date` where `date` is after
      `idate`. Data is an NxM ndarray.
    idate : datetime
      The date to "interpolate" the images to.

    Returns
    -------
    iwr : dict
      An interpolated weather radar image. Is a copy of `wr_before` with the
      `data` and `date` keys modified. The key `interpolated: True` is added.
    """
    if wr_before['data'].size != wr_after['data'].size:
        raise ValueError('Dimensions of weather radar images do not match')

    if not wr_before['date'] <= idate <= wr_after['date']:
        raise ValueError('Interpolation date does not lie between known dates')

    if wr_before['date'] == wr_after['date']:
        return wr_before

    time_delta = (wr_after['date'] - wr_before['date']).total_seconds()
    before_delta = (idate - wr_before['date']).total_seconds()
    before_factor = 1 - (before_delta / time_delta)

    iwr = wr_before.copy()
    iwr['data'] = before_factor * wr_before['data'] + (1 - before_factor) * wr_after['data']
    iwr['date'] = idate
    iwr['interpolated'] = True

    return iwr


def rainfall2lwc(wr):
    """Estimate liquid water content from rainfall.

    Arguments
    ---------
    wr : dict
      A dictionary with the keys 'lats', 'lons' and 'data'. Data is a (n,m)
      nadarray containing rainfall in mm/hr.

    Returns
    -------
    (n,m) ndarray containing the LWC at each pixel in `wr['data']`.

    Notes
    -----
    Liquid water content is estimated using the Marshall-Palmer (1948) droplet
    size distribution. This may not always be appropriate.

    """
    return 8.89 * (10**-2) * (wr['data']**0.84)
