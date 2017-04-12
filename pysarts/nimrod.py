"""
Module dealing with weather radar images.
"""
from datetime import datetime
import os.path

from netCDF4 import Dataset
import numpy as np
from scipy.stats import pearsonr

from .geogrid import GeoGrid


class Nimrod(GeoGrid):
    def __init__(self, lons, lats, data, date, interpolated=False):
        super().__init__(lons, lats, data)
        self.date = date
        self.interpolated = interpolated

        if isinstance(self.data, np.ma.MaskedArray):
            self.data.fill_value = 0
            self.data = self.data.filled()

    @classmethod
    def from_netcdf(cls, path):
        date = datetime.strptime(os.path.splitext(os.path.basename(path))[0],
                                 '%Y%m%d%H%M')

        with Dataset(path) as df:
            return cls(df.variables['lon'][:], df.variables['lat'][:],
                       df.variables['z'][:, :], date)

    @classmethod
    def interp_radar(cls, wr_before, wr_after, idate):
        """Apply time interpolation to two weather radar images.

        Returns a new instance of `Nimrod` interpolated between the two values.
        """
        if wr_before.data.size != wr_after.data.size:
            raise ValueError('Dimensions of weather radar images do not match')

        if not wr_before.date <= idate <= wr_after.date:
            raise ValueError(('Interpolation date does not lie between known '
                              'dates'))

        if wr_before.date == wr_after.date:
            return wr_before

        time_delta = (wr_after.date - wr_before.date).total_seconds()
        before_delta = (idate - wr_before.date).total_seconds()
        before_factor = 1 - (before_delta / time_delta)

        new_data = before_factor * wr_before.data + (1 - before_factor) * wr_after.data

        return cls(wr_before.lons, wr_before.lats, new_data, idate, True)

    def lwc(self):
        """Estimate the liquid water content from rainfall intensity.

        Returns
        -------
        `Nimrod` instance containing the LWC in g/m^3.

        Notes
        -----
        Liquid water content is estimated using the Marshall-Palmer (1948)
        droplet size distribution. This may not always be appropriate.

        """
        lwc = 8.89 * (10**-2) * (self.data**0.84)
        return Nimrod(self.lons, self.lats, lwc, self.date)


def calc_wr_ifg_correlation(wr, ifg, rain_tol=0):
    """Calculates the correlation coefficient between a weather radar image and an
    interferogram.

    Arguments
    ---------
    wr : Nimrod
      An instance of `Nimrod`.
    ifg : insar.InSAR
      An instance of `insar.InSAR` covering the same area as `wr` and at the
      same resolution.
    rain_tol : float
      Pixels with rainfall less than or equal to rain_tol won't be included in
      the calculation.

    Returns
    -------
    r : float
      The correlation coefficient
    p : float
      2-tailed p-value.
    n : int
      The number of data points used in the calculation.

    """
    if wr.data.size != ifg.data.size:
        raise ValueError(('Size of weather radar image does not match '
                          'interferogram.'))

    # Filter pixels with rainfall below tolerance
    wr_below_tol_idxs = np.where(wr.data.ravel() > rain_tol)
    wr_data = wr.data.ravel()[wr_below_tol_idxs]
    ifg_data = ifg.data.ravel()[wr_below_tol_idxs]

    # Filter pixels outside of interferogram
    ifg_zero_idxs = np.where(np.logical_not(np.isclose(0, ifg_data)))
    ifg_data = ifg_data[ifg_zero_idxs]
    wr_data = wr_data[ifg_zero_idxs]

    return pearsonr(wr_data, ifg_data) + (wr_data.size,)
