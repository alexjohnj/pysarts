import os.path
from datetime import datetime

from netCDF4 import Dataset
from numba import jit
import numba
import numpy as np

def extract_timestamp_from_ifg_name(file_name):
    """Extract master and slave date objects from the name of an interferogram.

    Expects files to be called */YYYYMMDD_YYYYMMDD.xyz. The first date is
    interpreted as the slave date, the second the master.

    Arguments
    ---------
    file_name : str
      The name of the file (can be a path) to extract the dates from.

    Returns
    -------
    A 2-tuple containing (master, slave) date objects.
    """
    base_name = os.path.basename(file_name)
    base_name, _ = os.path.splitext(base_name)
    name_parts = base_name.split('_')
    slave_date = datetime.strptime(name_parts[0], "%Y%m%d")
    master_date = datetime.strptime(name_parts[1], "%Y%m%d")

    return (master_date.date(), slave_date.date())


def load_dem(path):
    """Loads a NetCDF formatted DEM.

    Returns a dictionary with the keys 'lats', 'lons' and 'data'.
    """
    with Dataset(path) as df:
        dem = {}
        dem['lons'] = df.variables['lon'][:]
        dem['lats'] = df.variables['lat'][:]
        dem['data'] = df.variables['Band1'][:, :]

        return dem


# Taken from David-OConnor/brisk.
@jit(nopython=True)
def bisect(a, x):
    """Similar to bisect.bisect() from the built-in library."""
    M = a.size
    for i in range(M):
        if a[i] > x:
            return i

    return M


@jit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64[:]))
def interp1d(xs, ys, xis):
    """1D linear interpolation of points with linear extrapolation.

    Arguments
    ---------
    xs : (n,) ndarray
      Points that `ys` have been calculated at.
    ys : (n,) ndarray
      Value of function at `xs`.
    xis : (m,) ndarray
      Points to interpolate at.

    Returns
    -------
    (m,) ndarray containing values interpolated from `ys` at points
    `xis`. Points in `xis` falling outside of `xs` will be extrapolated using a
    linear extrapolation scheme.

    Notes
    -----
    This implementation is based on the linear interpolation implemented in
    David-OConnor/brisk. The scheme has been extended to include linear
    extrapolation for points outside of the input points however.
    """
    yis = np.empty(xis.size, dtype=np.float)
    interp1d_jit(xs, ys, xis, yis)
    return yis


@jit(numba.void(numba.float64[:], numba.float64[:], numba.float64[:], numba.float64[:]), nopython=True)
def interp1d_jit(xs, ys, xis, yis):
    for idx in range(xis.size):
        idx1 = bisect(xs, xis[idx])
        idx0 = idx1 - 1

        # Extrapolate if interpolation point falls outside of xs.
        if idx1 == 0:
            idx1 = 1
            idx0 = 0
        elif idx1 == xs.size:
            idx1 = -1
            idx0 = -2

        yis[idx] = ys[idx0] + ((xis[idx] - xs[idx0]) * (ys[idx1] - ys[idx0])
                               / (xs[idx1] - xs[idx0]))


# Implementation based on Numpy implementation
@jit(nopython=True)
def trapz(xs, ys):
    """Integration of points using trapezium rule.

    Arguments
    ---------
    xs : (n,) ndarray
      Points where function to integrate has been evaluated at.
    ys : (n,) ndarray
      Value of function to integrate at points `xs`.

    Returns
    -------
    A float.
    """
    d = np.diff(xs)
    return (d * (ys[1:] + ys[:-1]) / 2.0).sum()
