"""The ERA module contains functions for reading and processing weather model
output from era-interim.

The core data structure for this module is the *era_model* dictionary. This is
a dictionary with the following keys:
"""

from netCDF4 import Dataset


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




def load_era_netcdf(path):
    pass
