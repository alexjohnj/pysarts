"""Module for interacting with TRAIN output.

"""

from datetime import datetime
from math import pi

import scipy.io as scio
import numpy as np


def load_train_slant_delay(path):
    """Loads the slant atmospheric delays calculated by TRAIN.

    Arguments
    ---------
    path : str
      Path to a '.mat' file containing the variables 'lats', 'lons',
      'wet_delay', 'hydro_delay'

    Returns
    -------
    A dictionary with the same keys as the variables in `path`.
    """
    return scio.loadmat(path, variable_names=('lats', 'lons', 'wet_delay',
                                              'hydro_delay'))


def load_train_ifg_delay(delay_path, grid_path, dates_path, master_date, slave_date):
    """Load atmospheric interferometric delays computed by TRAIN.

    Arguments
    ---------
    delay_path : str
      Path to a TRAIN era atmospheric delay output file.
    grid_path : str
      Path to a '.mat' file containing the grid for the computed delays.
    dates_path : str
      Path to a '.mat' file containing the master and slave dates used by
      TRAIN.
    master_date, slave_date : date
      Master-slave date pairing to extract the correction for.

    Notes (IMPORTANT)
    -----------------
    Delays will be converted from phase to LOS delay with an *assumed
    wavelength* of 0.0562 cm (Sentinel-1's wavelength).

    TODO: MAKE THE WAVELENGTH CONFIGURABLE

    Returns
    -------
    A dictionary with the keys 'lats', 'lons', 'master_date', 'slave_date',
    'hydro_delay', 'wet_delay' and 'total_delay'.
    """
    # Coerce datetimes into dates
    if isinstance(master_date, datetime):
        master_date = master_date.date()

    if isinstance(slave_date, datetime):
        slave_date = slave_date.date()

    # Load data
    corrections = scio.loadmat(delay_path,
                               variable_names=('ph_tropo_era',
                                               'ph_tropo_era_wet',
                                               'ph_tropo_era_hydro'))
    grid = scio.loadmat(grid_path)
    dates = scio.loadmat(dates_path)

    # Convert grid to lons and lats
    lons = np.unique(grid['lonlat'][:, 0])
    lats = np.unique(grid['lonlat'][:, 1])

    # Find index of master-slave pairing in dates. This corresponds to the
    # column containing the interferogram pixels in the correction
    # dictionary's matrices.
    #
    # Could probably do this neater but there's only five and a half weeks left
    # on this project and oh my god why am I wasting time writing this comment?
    date_idx = None
    for (idx, row) in enumerate(dates['ifgday']):
        row_master = datetime.strptime(str(row[0]), '%Y%m%d').date()
        row_slave = datetime.strptime(str(row[1]), '%Y%m%d').date()
        if row_master == master_date and row_slave == slave_date:
            date_idx = idx

    if date_idx is None:
        raise KeyError('Master-Slave Date not found in dates file')

    # Pull out the data for these dates
    total_delay = corrections['ph_tropo_era'][:, date_idx]
    wet_delay = corrections['ph_tropo_era_wet'][:, date_idx]
    hydro_delay = corrections['ph_tropo_era_hydro'][:, date_idx]

    # Reshape the data so it matches the grid
    new_shape = (lats.size, lons.size)
    total_delay = np.reshape(total_delay, new_shape)
    wet_delay = np.reshape(wet_delay, new_shape)
    hydro_delay = np.reshape(hydro_delay, new_shape)

    # Convert phase to LOS delay in cm.
    def phase2delay(data):
        wavelength = 0.0562  # cm
        return data * wavelength / (4 * pi)

    total_delay = phase2delay(total_delay)
    wet_delay = phase2delay(wet_delay)
    hydro_delay = phase2delay(hydro_delay)

    # Build the output dictionary
    df = {
        'lons': lons,
        'lats': lats,
        'master_date': master_date,
        'slave_date': slave_date,
        'hydro_delay': hydro_delay,
        'wet_delay': wet_delay,
        'total_delay': total_delay,
    }

    return df
