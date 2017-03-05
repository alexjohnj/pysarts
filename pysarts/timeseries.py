import logging
import numpy as np

def calculate_master_atmosphere(ts_ifgs, baselines, dates, master_date):
    """Calculate the master atmosphere for a timeseries.

    Arguments
    ---------
    ts_ifgs : ndarray
      A 3D matrix of interferograms that form a timeseries along the third
      dimension of the matrix.
    baselines : array_like
      An array of perpendicular baselines for each ifg date relative to the
      master date. This should be in the same order as `ts_ifgs`.
    dates : list
      A list of date objects in the same order as the timeseries in `ts_ifgs`.
    master_date : date
      The master date in `ts_ifgs`

    Returns
    -------
    - A 2D ndarray containing the master atmosphere.
    - A 2D ndarray containing the DEM error.

    """
    # Make dates relative to the master_date
    relative_dates = [(date - master_date).days for date in dates]
    grid_shape = ts_ifgs.shape[0:2]
    master_atmosphere = np.zeros(grid_shape)
    dem_error = np.zeros(grid_shape)

    kernel = np.ones((ts_ifgs.shape[2], 3))
    kernel[:, 1] = np.array(baselines)
    kernel[:, 2] = np.array(relative_dates)

    # Reshape matrices so the inversion can be run in one function call
    data = ts_ifgs.transpose(2, 0, 1).reshape(kernel.shape[0], -1)
    logging.info('Solving for %d unknowns using %d knowns',
                 3 * grid_shape[0] * grid_shape[1],
                 data.size)
    model = np.linalg.lstsq(kernel, data)[0]
    master_atmosphere = model[0, :].reshape(grid_shape)
    dem_error = model[1, :].reshape(grid_shape)

    return (master_atmosphere, dem_error)
