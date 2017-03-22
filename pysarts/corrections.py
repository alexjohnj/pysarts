"""Defines corrections for interferometric errors."""

import numpy as np
from numba import jit
import numba
from .util import trapz, interp1d_jit


@jit
def calculate_era_zenith_delay(model, dem):
    """Calculates the one-way zenith delay in cm predicted by a weather
    model.

    Arguments
    --------
    model : ERAModel
      The weather model to use in the delay calculation.
    dem : dict
      A dictionary with the keys 'lats', 'lons' and 'data'. Defines the DEM
      with height in metres.

    Returns
    -------
    A dictionary with the keys 'lats', 'lons', 'wet_delay', 'dry_delay' and
    'data'. Data is an (n,m) ndarray containing the one-way delay in
    centimetres. 'wet_delay' and 'dry_delay' contain the corresponding delay
    components in centimetres.

    Notes
    -----
    `model` should be interpolated so it is on the same grid as `dem`.
    """
    if (dem['lats'].size != model.lats.size
        or dem['lons'].size != model.lons.size):
        raise IndexError('Size of model grid does not match size of dem')

    # Allocate output arrays
    wet_delay = np.empty((model.lats.size, model.lons.size))
    dry_delay = np.empty((model.lats.size, model.lons.size))

    # Cache the height and ppwv to avoid recalculating at every pixel.
    height = model.height
    ppwv = model.ppwv

    dry_delay = np.empty(dem['data'].shape)
    wet_delay = np.empty(dem['data'].shape)

    _calculate_zenith_delay_jit(dem['data'], height, model.temp, ppwv,
                                model.pressure, wet_delay, dry_delay)

    output = {
        'lons': model.lons,
        'lats': model.lats,
        'data': dry_delay + wet_delay,
        'wet_delay': wet_delay,
        'dry_delay': dry_delay,
    }

    return output


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
    lwc : (n,m) ndarray
      Matrix containing the liquid water content in g/m^3 at each grid point.
    cloud_thickness (n,m) ndarry OR float
      A matrix containing the cloud thickness at each grid point. Pass a single
      float for a constant cloud thickness.

    Returns
    -------
    (n,m) ndarray containing the liquid zenith delay in centimetres.
    """
    return 0.145 * lwc * cloud_thickness


def zenith2slant(zenith_delay, angle):
    """Map zenith delay to slant delay using a cosine mapping.

    Arguments
    ---------
    zenith_delay : (n,m) ndarray
      Matrix containing the zenith delay.
    angle : float or (n,m) ndarray
      The look angle at each point in radians the interferogram or a single
      number for a constant look angle.

    Returns
    -------
    A matrix of size (n,m) containing the slant delay.
    """
    return zenith_delay / np.cos(angle)


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
