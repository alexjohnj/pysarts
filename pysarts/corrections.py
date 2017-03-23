"""Defines corrections for interferometric errors."""

import numpy as np
from numba import jit
import numba
from .util import trapz, interp1d_jit
from .insar import SAR


@jit
def calculate_era_zenith_delay(model, dem):
    """Calculates the one-way zenith delay in cm predicted by a weather
    model.

    Arguments
    --------
    model : ERAModel
      The weather model to use in the delay calculation.
    dem : geogrid.GeoGrid
      A GeoGrid instance defining the DEM in metres.

    Returns
    -------
    A three tuple containing (wet_delay, dry_delay and total_delay) that are
    instances of insar.SAR containing the one-way zenith delay in centimetres.

    Notes
    -----
    `model` should be interpolated so it is on the same grid as `dem`.
    """
    if (dem.lats.size != model.lats.size
        or dem.lons.size != model.lons.size):
        raise IndexError('Size of model grid does not match size of dem')

    # Allocate output arrays
    wet_delay = np.empty((model.lats.size, model.lons.size))
    dry_delay = np.empty((model.lats.size, model.lons.size))

    # Cache the height and ppwv to avoid recalculating at every pixel.
    height = model.height
    ppwv = model.ppwv

    dry_delay = np.empty(dem.data.shape)
    wet_delay = np.empty(dem.data.shape)

    _calculate_zenith_delay_jit(dem.data, height, model.temp, ppwv,
                                model.pressure, wet_delay, dry_delay)

    wet_delay = SAR(model.lons, model.lats, wet_delay, model.date)
    dry_delay = SAR(model.lons, model.lats, dry_delay, model.date)
    total_delay = SAR(model.lons, model.lats, dry_delay.data + wet_delay.data,
                      model.date)

    return (wet_delay, dry_delay, total_delay)


@jit(numba.void(numba.float64[:, :],
                numba.float64[:, :, :],
                numba.float64[:, :, :],
                numba.float64[:, :, :],
                numba.float64[:, :, :],
                numba.float64[:, :],
                numba.float64[:, :]),
     nopython=True)
def _calculate_zenith_delay_jit(dem, height, temp, ppwv, pressure, out_wet, out_dry):
    """JIT optimised function to calculate the zenith wet and dry delay.

    Arguments
    ---------
    dem : (n,m) ndarray
      DEM in matrix form.
    height : (n,m,o) ndarray
      Heights of the weather model.
    temp : (n,m,o) ndarray
      Weather model temperatures
    ppwv : (n,m,o) ndarray
      Weather model partial pressure of water vapour
    pressure : (n,m,o) ndarray
      Weather model pressure.
    out_wet : (n,m) ndarray
      Matrix to save wet delay into.
    out_dry : (n,m) ndarray
      Matrix to save dry delay into.

    Returns
    -------
    None

    """
    # The implementation here is loosely based off of the implementation in
    # TRAIN by David Bekaert. There are some differences that give slightly
    # different numerical results. Notably, the dry delay is found by
    # integrating pressure with height rather than by calculating the surface
    # pressure. Note the performance is rather poor due to the reliance on a
    # loop. This can probably be made faster.
    # Constants and formulae from Hanssen 2001
    k1 = 0.776  # K Pa^{-1}
    k2 = 0.716  # K Pa^{-1}
    k3 = 3.75 * 10**3  # K^2 Pa{-1}
    Rd = 287.053
    Rv = 461.524

    nheights = 1024

    # Caches for the interpolated variables. Needed for JIT evaluation of the
    # interpolation function.
    itemp = np.zeros(nheights)
    ippwv = np.zeros(nheights)
    ipressure = np.zeros(nheights)
    for (y, x), _ in np.ndenumerate(dem):
        new_heights = np.linspace(dem[y, x], 15000, nheights)
        interp1d_jit(height[y, x, :], temp[y, x, :], new_heights, itemp)
        interp1d_jit(height[y, x, :], ppwv[y, x, :], new_heights, ippwv)
        interp1d_jit(height[y, x, :], pressure[y, x, :], new_heights, ipressure)

        # Convert pressures from hPa to Pa
        ipressure *= 100
        ippwv *= 100

        wet_refract = ((k2 - (k1 * Rd / Rv)) * (ippwv / itemp)
                       + (k3 * ippwv / (itemp**2)))
        dry_refract = k1 * ipressure / itemp

        # Convert delays to centimetres
        out_wet[y, x] = 10**-6 * trapz(new_heights, wet_refract) * 100
        out_dry[y, x] = 10**-6 * trapz(new_heights, dry_refract) * 100


def liquid_zenith_delay(lwc, cloud_thickness):
    """Calculate the liquid delay assuming a constant liquid water content.

    Arguments
    ---------
    lwc : nimrod.Nimrod
      Nimrod instance containing the liquid water content in g/cm^3.
    cloud_thickness (n,m) ndarry OR float
      A matrix containing the cloud thickness at each grid point. Pass a single
      float for a constant cloud thickness.

    Returns
    -------
    (n,m) ndarray containing the liquid zenith delay in centimetres.
    """
    delay = 0.145 * lwc.data * cloud_thickness
    return SAR(lwc.lons, lwc.lats, delay, lwc.date)


def calc_ifg_delay(master_delay, slave_delay):
    """Calculate the interferometric delay.

    Arguments
    ---------
    master_delay : (n,m) ndarray
      Matrix containing the atmospheric delay on the master date.
    slave_delay : (n,m) ndarray
      Matrix containing the atmospheric delay on the slave date.

    Returns
    -------
    A matrix of size (n,m) containing the interferometric delay.
    """
    return master_delay - slave_delay
