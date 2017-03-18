"""The ERA module contains functions for reading and processing weather model
output from era-interim.

The core data structure for this module is the *era_model* dictionary. This is
a dictionary with the following keys:
"""

from datetime import datetime, timedelta

from netCDF4 import Dataset
from scipy.interpolate import RegularGridInterpolator
import numpy as np


class ERAModel(object):
    """
    Represents the output of an era-interim weather model.

    Instance Variables
    ------------------
    rel_hum : (n,m,o) ndarray
      The relative humidity as a percentage at each node.
    temp : (n,m,o) ndarray
      The temperature in Kelvin at each node.
    geopot : (n,m,o) ndarray
      The geopotential in m^2 s^-2 at each node.
    pressure : (n,m,o) ndarray
      The pressure at each model node in hPa. Will be constant for a given
      pressure level. Derived from the pressure levels.
    lats : (n,) ndarray
      Latitudes of weather model nodes.
    lons : (m,) ndarray
      Longitudes of weather model nodes.
    date : datetime
      The date and time of the weather model.

    Properties
    ----------
    ppwv : (n,m,o) ndarray
      The partial pressure of water vapour in hPa. Derived from `temp` and
      `rel_hum`.
    height : (n,m,o) ndarray
      The geometric height of each node in metres. Derived from `geopot`.

    Initialisation
    --------------
    """
    def __init__(self, lats, lons, date, rel_hum, temp, geopot, pressure):
        self.rel_hum = rel_hum
        self.temp = temp
        self.geopot = geopot
        self.pressure = pressure
        self.lats = lats
        self.lons = lons
        self.date = date

    @classmethod
    def load_era_netcdf(cls, path):
        """Load ERA weather model from a netCDF file."""
        with Dataset(path) as df:
            # Extract variables
            lons = df.variables['longitude'][:]
            lats = df.variables['latitude'][:]
            p_levels = df.variables['level'][:]
            rel_hum = df.variables['r'][0, :, :, :]
            temp = df.variables['t'][0, :, :, :]
            geopot = df.variables['z'][0, :, :, :]

            # Transpose temperature, humidity and potential so they have
            # dimensions given by (lat, lon, p_level). By default they have
            # dimensions given by (p_level, lat, lon).
            temp = temp.transpose(1, 2, 0)
            rel_hum = rel_hum.transpose(1, 2, 0)
            geopot = geopot.transpose(1, 2, 0)

            # Reshape pressure levels so they form a pressure grid.
            nlevels = p_levels.size
            p_levels = np.repeat(p_levels, lats.size * lons.size)
            pressures = p_levels.reshape(nlevels,
                                         lons.size,
                                         lats.size).transpose(1, 2, 0)

            # Flip 3D variables the along 3rd axis so that highest pressure
            # level is the first in the stack.
            pressures = np.flip(pressures, 2)
            temp = np.flip(temp, 2)
            rel_hum = np.flip(rel_hum, 2)
            geopot = np.flip(geopot, 2)

            # Flip variables along 1st axis so that var[0, 0] corresponds to
            # lat[0], lon[0].
            pressures = np.flipud(pressures)  # Not really needed
            temp = np.flipud(temp)
            rel_hum = np.flipud(rel_hum)
            geopot = np.flipud(geopot)
            lats = lats[::-1]

            # Extract the date. Date is stored as number of hours since
            # midnight 1900-01-01
            reference_time = datetime(1900, 1, 1, 0, 0, 0)
            hours_since = df.variables['time'][0]
            reference_delta = timedelta(hours=int(hours_since))
            date = reference_time + reference_delta

            # Initialise an instance
            return ERAModel(lats, lons, date, rel_hum, temp, geopot, pressures)

    @property
    def ppwv(self):
        # Calculation of saturated water vapour pressure for water and ice is
        # based on Buck 1981 and Alduchow & Eskridge 1996. Code implementation
        # based on implantation in TRAIN by David Bekaert.
        UPPER_TEMP_BOUND = 273.16  # K
        LOWER_TEMP_BOUND = 250.16  # K

        # Calculate saturated pressure for water (T > 0) and ice (T < 0) in
        # hPa.
        svp_water = 6.1121 * np.exp((17.502 * (self.temp - 273.16))
                                    / (240.97 + self.temp - 273.16))
        svp_ice = 6.1112 * np.exp((22.587 * (self.temp - 273.16))
                                  / (273.86 + self.temp - 273.16))

        wgt = (self.temp - LOWER_TEMP_BOUND) / (UPPER_TEMP_BOUND - LOWER_TEMP_BOUND)
        svp = svp_ice + (wgt**2) * (svp_water - svp_ice)

        svp[self.temp > UPPER_TEMP_BOUND] = svp_water[self.temp > UPPER_TEMP_BOUND]
        svp[self.temp < LOWER_TEMP_BOUND] = svp_ice[self.temp < LOWER_TEMP_BOUND]

        return self.rel_hum * svp / 100

    @property
    def height(self):
        STANDARD_GRAV = 9.80665
        EARTH_RADIUS_MAX = 6378137
        EARTH_RADIUS_MIN = 6356752

        # Calculate the variation in gravity across the model using WGS84
        # gravity formula. The magic number a, b and c make the formula:
        # a * ([1 + b*sin^2(phi)] / sqrt(1 - c*sin^2(lat))) = g
        # Gives gravitational acceleration in m/s^2
        nlevels = self.pressure.shape[2]
        lat_mesh, _ = np.meshgrid(self.lats, self.lons)
        lat_mesh = np.deg2rad(lat_mesh)
        lat_mesh = lat_mesh.repeat(nlevels)
        lat_mesh = lat_mesh.reshape(self.lats.size,
                                    self.lons.size,
                                    nlevels)

        a = 9.7803253359
        b = 0.00193185265241
        c = 0.00669437999013
        grav = (a * (1 + b * np.sin(lat_mesh)**2)
                / np.sqrt(1 - c * np.sin(lat_mesh)**2))

        # Calculate the variation in the earth's radius with latitude
        numerator = ((EARTH_RADIUS_MAX**2 * np.cos(lat_mesh))**2
                     + (EARTH_RADIUS_MIN**2 * np.sin(lat_mesh))**2)
        denominator = ((EARTH_RADIUS_MAX * np.cos(lat_mesh)**2)
                       + (EARTH_RADIUS_MIN * np.sin(lat_mesh)**2))
        earth_radius = np.sqrt(numerator / denominator)

        # Let's go find the height
        geopot_height = self.geopot / STANDARD_GRAV
        height = ((geopot_height * earth_radius)
                  / (grav / STANDARD_GRAV * earth_radius - geopot_height))

        return height

    def clip(self, lon_bounds, lat_bounds):
        """Clip the weather model to a target region.

        Mutates self.

        Arguments
        ---------
        lon_bounds : tuple
          2-tuple containing (lower, upper) bounds for longitude.
        lat_bounds : tuple
          2-tuple containing (lower, upper) bounds for latitude.
        """
        lon_min, lon_max = min(lon_bounds), max(lon_bounds)
        lat_min, lat_max = min(lat_bounds), max(lat_bounds)

        if (not self._point_is_in_grid(lon_min, lat_min)
            or not self._point_is_in_grid(lon_max, lat_max)):
            raise IndexError('Bounding box is outside of the grid.')

        lon_min_idx = np.argmin(np.abs(self.lons - lon_min))
        lon_max_idx = np.argmin(np.abs(self.lons - lon_max))
        lat_min_idx = np.argmin(np.abs(self.lats - lat_min))
        lat_max_idx = np.argmin(np.abs(self.lats - lat_max))

        self.lats = self.lats[lat_min_idx:lat_max_idx+1]
        self.lons = self.lons[lon_min_idx:lon_max_idx+1]
        self.rel_hum = self.rel_hum[lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1, :]
        self.temp = self.temp[lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1, :]
        self.geopot = self.geopot[lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1, :]
        self.pressure = self.pressure[lat_min_idx:lat_max_idx+1, lon_min_idx:lon_max_idx+1, :]

        return None

    def _point_is_in_grid(self, x, y):
        """Check if a point falls within a grid.

        Arguments
        ---------
        x, y : float
          x and y coordinates to test

        Returns
        -------
        `True` if (x, y) lies within the grid. Otherwise `False`.
        """
        return (np.amin(self.lons) <= x <= np.amax(self.lons)
                and
                np.amin(self.lats) <= y <= np.amax(self.lats))

    def resample(self, new_lons, new_lats, new_plevels=None):
        """Resample the model onto a new grid.

        Linear resampling is carried out using a 3D RegularGridInterpolator.

        Arguments
        ---------
        new_lats : (n,) ndarray
          The new latitudes to sample the model at.
        new_lons : (m,) ndarray
          The new longitudes to sample the model at.
        new_plevels : (o,) ndarray, opt
          The new pressure levels to sample the model at in decreasing
          order. If `None`, the pressure levels will be unchanged.

        `self`'s geopotential, humidity and temperature variables will be
        interpolated onto a grid of size (n,m,o).

        """
        # RegularGridInterpolator mandates that the grid points are strictly
        # ascending. All the flips along the second axis of the data matrices
        # is to make pressure levels ascend instead of descend.
        plevels = self.pressure[0, 0, ::-1]  # P-levels from lowest to highest
        if new_plevels is None:
            new_plevels = plevels
        else:
            new_plevels = new_plevels[::-1]

        new_nlevels = new_plevels.size
        grid = (self.lats, self.lons, plevels)
        # Build the new grid for interpolation onto.
        xis, yis, pis = np.broadcast_arrays(new_lats.reshape(-1, 1, 1),
                                            new_lons.reshape(1, -1, 1),
                                            new_plevels)
        interp_coords = np.vstack((xis.flatten(), yis.flatten(),
                                   pis.flatten()))

        # Let's go interpolate some variables
        temp_interp = RegularGridInterpolator(grid, np.flip(self.temp, 2))
        hum_interp = RegularGridInterpolator(grid, np.flip(self.rel_hum, 2))
        geopot_interp = RegularGridInterpolator(grid, np.flip(self.geopot, 2))

        new_temps = temp_interp(interp_coords.T).reshape(new_lats.size,
                                                         new_lons.size,
                                                         new_plevels.size)
        new_hum = hum_interp(interp_coords.T).reshape(new_lats.size,
                                                      new_lons.size,
                                                      new_plevels.size)
        new_geopot = geopot_interp(interp_coords.T).reshape(new_lats.size,
                                                            new_lons.size,
                                                            new_plevels.size)

        self.temp = np.flip(new_temps, 2)
        self.rel_hum = np.flip(new_hum, 2)
        self.geopot = np.flip(new_geopot, 2)

        # Just repeat pressures to interpolate
        self.pressure = (np.repeat(new_plevels, new_lons.size * new_lats.size)
                         .reshape(new_nlevels, new_lats.size, new_lons.size)
                         .transpose(1, 2, 0))
        self.pressure = np.flip(self.pressure, 2)
        self.lats = new_lats
        self.lons = new_lons

        return None
