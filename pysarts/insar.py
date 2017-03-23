import glob
import os.path

import numpy as np
from scipy.interpolate import RectBivariateSpline
from netCDF4 import Dataset

from .geogrid import GeoGrid
from . import util


class SAR(GeoGrid):
    """Class encapsulating a single SAR image."""
    def __init__(self, lons, lats, data, date):
        super().__init__(lons, lats, data)
        self.date = date

    @classmethod
    def interpolate(cls, sar_before, sar_after, idate):
        """Time interpolate a SAR image to a specific date."""
        if sar_before.data.size != sar_after.data.size:
            raise ValueError('Dimensions of weather radar images do not match')

        if not sar_before.date <= idate <= sar_after.date:
            raise ValueError(('Interpolation date does not lie between known '
                              'dates'))

        if sar_before.date == sar_after.date:
            return sar_before

        time_delta = (sar_after.date - sar_before.date).total_seconds()
        before_delta = (idate - sar_before.date).total_seconds()
        before_factor = 1 - (before_delta / time_delta)

        new_data = before_factor * sar_before.data + (1 - before_factor) * sar_after.data

        return cls(sar_before.lons, sar_before.lats, new_data, idate)

    def zenith2slant(self, angle):
        """Returns a new SAR instance containing the slant delay calculated
        from a cosine mapping with `angle`. `angle` can be an float or a
        matrix."""
        return SAR(self.lons, self.lats, self.data / np.cos(angle), self.date)


class InSAR(GeoGrid):
    """Class encapsulating an interferogram."""
    def __init__(self, lons, lats, data, master_date, slave_date):
        super().__init__(lons, lats, data)
        self.master_date = master_date
        self.slave_date = slave_date

    @classmethod
    def from_netcdf(cls, path):
        master_date, slave_date = util.extract_timestamp_from_ifg_name(path)
        with Dataset(path) as df:
            if 'Band1' in df.variables:
                # Read ISCE GDAL converted NetCDF
                return InSAR(df.variables['lon'][:],
                             df.variables['lat'][:],
                             df.variables['Band1'][:, :],
                             master_date,
                             slave_date)
            else:
                # Try reading a generic NetCDF
                return InSAR(df.variables['x'][:],
                             df.variables['y'][:],
                             df.variables['z'][:, :],
                             master_date,
                             slave_date)


def find_ifgs_for_dates(ifg_dir, master_date, slc_dates=None):
    """Find all the interferograms for a set of SLC dates and a given master
    date.

    Arguments
    ---------
    ifg_dir : str
      The directory to search for interferograms. Interferograms should be
      named as SLAVE_MASTER.nc where SLAVE and MASTER are datestamps in the
      format YYYYMMDD.
    master_date : date
      The master date.
    slc_dates : list(date), opt
      SLC dates to consider when selecting interferograms. A value of `None`
      (default) means use all the files in ifg_dir.

    Returns
    -------
    A list of files that are made up of images from `master_date` or
    `slc_dates`.
    """
    ifg_files = glob.glob(os.path.join(ifg_dir, '**/*.nc'), recursive=True)

    if not slc_dates:
        return ifg_files
    else:
        slc_dates.append(master_date)

    accepted_files = []
    for file in ifg_files:
        ifg_master_date, ifg_slave_date = util.extract_timestamp_from_ifg_name(file)
        if ifg_master_date in slc_dates and ifg_slave_date in slc_dates:
            accepted_files.append(file)

    return accepted_files
