"""Module implementing the GeoGrid class representing a grid of geospatial
data."""

import numpy as np
from scipy.interpolate import RectBivariateSpline, griddata
from netCDF4 import Dataset


class GeoGrid(object):
    """A class encapsulating a grid of geospatial data.

    The grid's origin lies at `data[0,0]`. This corresponds to lon[0], lat[0].

           lon[0]   lon[1]   lon[2]   ... | lon[n]
    lat[0]   0    |   1    |   2    | ... |  9
    lat[1]   10   |   11   |   12   | ... |  19
    lat[2]   20   |   21   |   22   | ... |  29
    ...      ...  |   ...  |   ...  | ... |  ...
    lat[n]   90   |   91   |   92   | ... |  99

    Attributes
    ----------
    lons : ndarray (m,)
      An ordered 1D array of floats specifying the longitude of points on the
      grid.
    lats : ndarray (n,)
      An ordered 1D array of floats specifying the latitude of points on the
      grid.
    data : ndarray (n,m)
      A 2D array containing a value for each combination of `lat` and `lon`.

    """

    def __init__(self, lons, lats, data):
        if data.shape != (lats.size, lons.size):
            raise ValueError("Dimension mismatch between data, lons and lats")

        self.lons = lons
        self.lats = lats
        self.data = data

    @classmethod
    def from_netcdf(cls, path):
        with Dataset(path) as df:
            if 'Band1' in df.variables:
                # Loading a generic GDAL file
                return cls(df.variables['lon'][:],
                           df.variables['lat'][:],
                           df.variables['Band1'][:, :])
            else:
                # GMT grd file
                return cls(df.variables['x'][:],
                           df.variables['y'][:],
                           df.variables['z'][:, :])

    def clip(self, lon_bounds, lat_bounds):
        """Clip the grid to a bounding box.

        Arguments
        ---------
        lon_bounds : 2-tuple
          (lower, upper) bounds for longitude.
        lat_bounds : 2-tuple
          (lower, upper) bounds for latitude.

        Returns
        -------
        None

        """
        lon_min, lon_max = min(lon_bounds), max(lon_bounds)
        lat_min, lat_max = min(lat_bounds), max(lat_bounds)

        if (not self.point_in_grid(lon_min, lat_min)
            or not self.point_in_grid(lon_max, lat_max)):
            raise IndexError('Bounding box is outside of the grid.')

        lon_min_idx = np.argmin(np.abs(self.lons - lon_min))
        lon_max_idx = np.argmin(np.abs(self.lons - lon_max))

        lat_min_idx = np.argmin(np.abs(self.lats - lat_min))
        lat_max_idx = np.argmin(np.abs(self.lats - lat_max))

        self.data = self.data[lat_min_idx:lat_max_idx+1,
                              lon_min_idx:lon_max_idx+1]
        self.lons = self.lons[lon_min_idx:lon_max_idx+1]
        self.lats = self.lats[lat_min_idx:lat_max_idx+1]

    def point_in_grid(self, lon, lat):
        """Returns true of a point lies within the grid."""
        return (np.amin(self.lons) <= lon <= np.amax(self.lons)
                and
                np.amin(self.lats) <= lat <= np.amax(self.lats))

    def interp_at_res(self, deltax, deltay, method='bivariate'):
        """Interpolate at a new resolution on a regular grid.

        Arguments
        ---------
        deltax : float
          Resolution along the x axis in metres.
        deltay : float
          Resolution along the y axis in metres.
        method : str, opt
          The method to use for interpolation. One of 'bivariate'
          (RectBivariateSpline, default) or 'nearest' (nearest neighbour).
        """
        def metre2deg(x):
            return x / 111110

        deltax_deg = metre2deg(deltax)
        deltay_deg = metre2deg(deltay)

        new_lons = np.arange(self.lons[0], self.lons[-1], deltax_deg)
        new_lats = np.arange(self.lats[0], self.lats[-1], deltay_deg)

        self.interp(new_lons, new_lats, method)

        return None

    def interp(self, ilons, ilats, method='bivariate'):
        """Interpolate onto a new regular grid.

        Arguments
        ---------
        ilons : (m,) ndarray
          New longitudes to interpolate the grid to.
        ilats : (n,) ndarray
          New latitudes to interpolate the grid to.
        method : str, opt
          The method to use for interpolation. One of 'bivariate'
          (RectBivariateSpline, default) or 'nearest' (nearest neighbour).

        Returns
        -------
        None

        See Also
        --------
        `geogrid.interpolated` for a non-mutating version.

        """
        if method == 'bivariate':
            self._interp_bivariate(ilons, ilats)
        elif method == 'nearest':
            self._interp_nearest(ilons, ilats)
        else:
            raise ValueError('Unknown interpolation method {}'.format(method))

        return None

    def _interp_bivariate(self, ilons, ilats):
        """Interpolation using a RectBivariateSpline"""
        splinef = RectBivariateSpline(self.lons,
                                      self.lats,
                                      self.data.T)

        self.data = splinef(ilons, ilats, grid=True).T
        self.lons = ilons
        self.lats = ilats

        return None

    def _interp_nearest(self, ilons, ilats):
        """Interpolation using a nearest neighbour algorithm."""
        grid = np.array(np.meshgrid(self.lats, self.lons)).T.reshape(-1, 2)
        igrid = np.array(np.meshgrid(ilats, ilons)).T.reshape(-1, 2)

        self.data = griddata(grid, self.data.ravel(), igrid, method='nearest')
        self.data.shape = (ilats.size, ilons.size)
        self.lats = ilats
        self.lons = ilons

        return None

    def std(self):
        """Returns the standard deviation of `self.data` with df=0."""
        return self.data.std(ddof=0)
