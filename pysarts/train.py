"""Module for interacting with TRAIN output.

"""

import scipy.io as scio


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
    return scio.loadmat(path, variable_names=('lats', 'lons', 'wet_delay', 'hydro_delay'))
