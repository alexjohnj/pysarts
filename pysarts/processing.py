import glob
import os.path

from . import util

def find_ifgs_for_dates(ifg_dir, master_date, slc_dates=None):
    """Find all the interferograms for a set of SLC dates and a given master date.

    Arguments
    ---------
    ifg_dir : str
      The directory to search for interferograms. Interferograms should be named
      as SLAVE_MASTER.nc where SLAVE and MASTER are datestamps in the format
      YYYYMMDD.
    master_date : date
      The master date.
    slc_dates : list(date), opt
      SLC dates to consider when selecting interferograms. A value of `None`
      (default) means use all the files in ifg_dir.

    Returns
    -------
    A list of files that are made up of images from `master_date` or `slc_dates`.
    """
    ifg_files = glob.glob(os.path.join(ifg_dir, '*.nc'))

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
