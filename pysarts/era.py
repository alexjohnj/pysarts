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
    pass



def load_era_netcdf(path):
    pass
