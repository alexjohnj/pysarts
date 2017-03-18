"""The ERA module contains functions for reading and processing weather model
output from era-interim.

The core data structure for this module is the *era_model* dictionary. This is
a dictionary with the following keys:
"""

from datetime import datetime, timedelta

from netCDF4 import Dataset
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
