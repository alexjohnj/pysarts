import logging
from math import floor
import tempfile

import numpy as np

from . import util

def construct_kernel(ifg_dates, master_date):
    """Constructs the kernel matrix for an unwrapped phase inverse problem.

    Arguments
    ---------
    ifg_dates : list(2-tuple(date))
      A list of 2-tuples containing the (master, slave) dates for each
      interferogram used in the inversion.
    master_date : date
      The date of the master interferogram. The kernel will be constructed so
      the inversion produces results relative to it.

    Returns
    -------
    A 2D `ndarray`.

    """
    # Create a lookup dictionary for dates to index into the kernel
    slc_dates = sorted(list(set([date for pairs in ifg_dates for date in pairs])))
    date_lookup = {date: idx for (idx, date) in enumerate(slc_dates)}

    kernel = np.zeros((len(ifg_dates), len(slc_dates)))
    for idx, (ifg_master, ifg_slave) in enumerate(ifg_dates):
        kernel[idx, date_lookup[ifg_master]] = -1
        kernel[idx, date_lookup[ifg_slave]] = 1

    kernel[:, date_lookup[master_date]] = 0 # Make inversion relative to master date

    return kernel

def calculate_inverse(ifg_paths, master_date, grid_shape, output_model):
    """Solves for the unwrapped phase of a time series of IFGs relative to a
    master date.

    Arguments
    ---------
    ifg_paths : list
      A list of paths to `npy` files containing the ifgs to be used.
    master_date : date
      The date to make the inversion relative to.
    grid_shape : tuple
      The size of the grid the interferograms at `ifg_paths` are on.
    output_model : ndarray
      A 3D matrix to save the inverted time series to. Basically this exists so
      you can pass a memory mapped matrix in. If you do use a memory mapped
      matrix, *make sure it's opened in 'w+' mode!*

    Returns
    -------
    A sorted list of dates whose indices can be used to find interferograms in
    `output_model`.
    """
    ifg_date_pairs = list(map(util.extract_timestamp_from_ifg_name, ifg_paths))
    kernel = construct_kernel(ifg_date_pairs, master_date)

    # Construct the data matrix
    with tempfile.TemporaryFile() as data_memmap_file:
        logging.debug('Creating data matrix memory map')
        data = np.memmap(data_memmap_file, np.float,
                         mode='w+',
                         shape=(grid_shape + (kernel.shape[0],)))
        logging.debug('Created data matrix memory map')

        # Load interferograms
        logging.debug('Loading interferograms')
        for idx, path in enumerate(ifg_paths):
            data[:, :, idx] = np.load(path)

        # Reshape matrices so the inversion can be done for all pixels in one function
        #
        # Each column is a pixel starting from the top-left corner. Each row is
        # a SLC time in ascending order.
        data = data.transpose(2, 0, 1).reshape(kernel.shape[0], -1)

        # Solve the inverse problem
        logging.info('Solving inverse problem')
        # A note on the reshaping. The output of lstsq is a 2D matrix. Each
        # column is a pixel in an interferogram starting from the top-left. Each
        # row is an SLC time in ascending order. We have to transpose the output
        # first before reshaping it back into a 3D array.
        output_model[:] = np.linalg.lstsq(kernel, data)[0].T.reshape(grid_shape + (-1,))

    new_ifg_dates = sorted(list(set([date for pairs in ifg_date_pairs for date in pairs])))
    return new_ifg_dates
