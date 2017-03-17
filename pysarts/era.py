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
    e : (n,m,o) ndarray
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
            temp = temp.transpose(2, 1, 0)
            rel_hum = rel_hum.transpose(2, 1, 0)
            geopot = geopot.transpose(2, 1, 0)

            # Reshape pressure levels so they form a pressure grid.
            nlevels = p_levels.size
            p_levels = np.repeat(p_levels, lats.size * lons.size)
            pressures = p_levels.reshape(nlevels,
                                         lons.size,
                                         lats.size).transpose(2, 1, 0)

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
