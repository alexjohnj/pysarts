import logging
import numpy as np

def calculate_master_atmosphere(ts_ifgs, dates, master_date):
    """Calculate the master atmosphere for a timeseries.

    Arguments
    ---------
    ts_ifgs : ndarray
      A 3D matrix of interferograms that form a timeseries along the third
      dimension of the matrix.
    dates : list
      A list of date objects in the same order as the timeseries in `ts_ifgs`.
    master_date : date
      The master date in `ts_ifgs`

    Returns
    -------
    A 2D ndarray containing the master atmosphere.

    """
    # Make dates relative to the master_date
    relative_dates = [(date - master_date).days for date in dates]
    grid_shape = ts_ifgs.shape[0:2]
    master_atmosphere = np.zeros(grid_shape)

    kernel = np.ones((ts_ifgs.shape[2], 2))
    kernel[:, 1] = np.array(relative_dates)

    logging.info('Starting master atmosphere calculation')
    for pixel in np.ndindex(grid_shape):
        print("Calculating pixel {:5d}, {:5d} of {:5d}, {:5d}".format(pixel[0]+1,
                                                                      pixel[1]+1,
                                                                      grid_shape[0],
                                                                      grid_shape[1],),
              end='\r')
        model, _, _, _ = np.linalg.lstsq(kernel, ts_ifgs[pixel[0], pixel[1], :])
        res = ts_ifgs[pixel[0], pixel[1], :] - kernel @ model
        master_atmosphere[pixel[0], pixel[1]] = np.mean(res)

    print("")
    return master_atmosphere
