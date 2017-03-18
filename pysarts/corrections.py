"""Defines corrections for interferometric errors."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import trapz

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

    # Maximum height and number of height levels to interpolate weather model
    # parameters to.
    maxheight = 15000
    nheights = 1024

    # The implementation here is loosely based off of the implementation in
    # TRAIN by David Bekaert. There are some differences that give slightly
    # different numerical results. Notably, the dry delay is found by
    # integrating pressure with height rather than by calculating the surface
    # pressure. Note the performance is rather poor due to the reliance on a
    # loop. This can probably be made faster.
    for (y, x), _ in np.ndenumerate(dem['data']):
        # Interpolate model parameters vertically from pixel's DEM height to
        # max troposphere height (15 km).
        new_heights = np.linspace(dem['data'][y, x], maxheight, nheights)
        temp_interpf = interp1d(height[y, x, :], model.temp[y, x, :],
                                fill_value="extrapolate")
        hum_interpf = interp1d(height[y, x, :], ppwv[y, x, :],
                               fill_value="extrapolate")
        pressure_interpf = interp1d(height[y, x, :], model.pressure[y, x, :],
                                    fill_value="extrapolate")

        temp_interp = temp_interpf(new_heights)
        hum_interp = hum_interpf(new_heights) * 100   # Pa
        pressure_interp = pressure_interpf(new_heights) * 100  # Pa

        # Calculate refractive index with height.
        # Constants and formulae from Hanssen 2001
        k1 = 0.776  # K Pa^{-1}
        k2 = 0.716  # K Pa^{-1}
        k3 = 3.75 * 10**3  # K^2 Pa{-1}
        Rd = 287.053
        Rv = 461.524

        wet_refract = ((k2 - (k1 * Rd / Rv)) * (hum_interp / temp_interp)
                       + (k3 * hum_interp / (temp_interp ** 2)))
        dry_refract = k1 * pressure_interp / temp_interp
        # Calculate delays and convert to centimetres.
        wet_delay[y, x] = 10**-6 * trapz(wet_refract, new_heights) * 100
        dry_delay[y, x] = 10**-6 * trapz(dry_refract, new_heights) * 100

    output = {
        'lons': model.lons,
        'lats': model.lats,
        'data': dry_delay + wet_delay,
        'wet_delay': wet_delay,
        'dry_delay': dry_delay,
    }

    return output
