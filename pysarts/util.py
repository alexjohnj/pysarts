import os.path
from datetime import datetime
def extract_timestamp_from_ifg_name(file_name):
    """Extract master and slave date objects from the name of an interferogram.

    Expects files to be called */YYYYMMDD_YYYYMMDD.xyz. The first date is
    interpreted as the slave date, the second the master.

    Arguments
    ---------
    file_name : str
      The name of the file (can be a path) to extract the dates from.

    Returns
    -------
    A 2-tuple containing (master, slave) date objects.
    """
    base_name = os.path.basename(file_name)
    base_name, _ = os.path.splitext(base_name)
    name_parts = base_name.split('_')
    slave_date = datetime.strptime(name_parts[0], "%Y%m%d")
    master_date = datetime.strptime(name_parts[1], "%Y%m%d")

    return (master_date.date(), slave_date.date())
