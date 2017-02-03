"""Module for interactive plotting of interferograms.

"""
import os
from datetime import date, datetime

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

from . import processing
from . import workflow

if __name__ == '__main__':
    from . import config
    import argparse

def _parse_unwrapped_ifg_args(args):
    plot_unwrapped_ifg(args.master, args.slave, args.output, args.resampled)

def plot_unwrapped_ifg(master_date, slave_date, fname=None, resampled=False):
    """
    master_date, slc_date : str or date
      If a string, should be in the format '%Y%m%d'.
    fname : str, opt
      The path to save the image to or `None`. If set, interactive plotting will
      be disabled.

    Returns
    -------
    None
    """
    if isinstance(master_date, date):
        master_date = date.strftime("%Y%m%d")
    if isinstance(slave_date, date):
        slave_date = date.strftime("%Y%m%d")

    ifg_name = '{}_{}'.format(slave_date, master_date)
    ifg_path = ''
    ifg = {}
    if resampled:
        ifg_path = os.path.join(config.SCRATCH_DIR, 'uifg_resampled', ifg_name + '.npy')
        data = np.load(ifg_path)
        lons, lats = workflow.read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                               'grid.txt'))
        ifg['data'] = data
        ifg['lons'] = lons
        ifg['lats'] = lats
        ifg['master_date'] = datetime.strptime(master_date, '%Y%m%d').date()
        ifg['slave_date'] = datetime.strptime(slave_date, '%Y%m%d').date()
    else:
        ifg_path = os.path.join(config.UIFG_DIR, ifg_name + '.nc')
        ifg = processing.open_ifg_netcdf(ifg_path)

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    bmap = Basemap(llcrnrlon=ifg['lons'][0],
                   llcrnrlat=ifg['lats'][0],
                   urcrnrlon=ifg['lons'][-1],
                   urcrnrlat=ifg['lats'][-1],
                   resolution='i',
                   projection='merc',
                   ax=axes)
    parallels = np.linspace(ifg['lats'][0], ifg['lats'][-1], 5)
    meridians = np.linspace(ifg['lons'][0], ifg['lons'][-1], 5)

    bmap.drawcoastlines()
    bmap.drawparallels(parallels, labels=[True, False, False, False], fmt="%.2f", fontsize=9)
    bmap.drawmeridians(meridians, labels=[False, False, False, True], fmt="%.2f", fontsize=9)
    bmap.drawmapboundary()

    vmax = (np.absolute(ifg['data']).max())
    vmin = vmax * -1

    lon_mesh, lat_mesh = np.meshgrid(ifg['lons'], ifg['lats'])
    image = bmap.pcolormesh(lon_mesh,
                            lat_mesh,
                            ifg['data'],
                            latlon=True,
                            cmap=cm.RdBu_r,
                            vmin=vmin,
                            vmax=vmax)

    cbar = fig.colorbar(image, pad=0.07)
    cbar.set_label('LOS Delay / cm')

    title = 'Unwrapped Interferogram\nMaster: {0}\nSlave: {1}'.format(
        ifg['master_date'].strftime('%Y-%m-%d'),
        ifg['slave_date'].strftime('%Y-%m-%d'))
    axes.set_title(title)
    fig.tight_layout()

    if fname:
        fig.savefig(fname)
    else:
        plt.show()

    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='pysarts.plot')
    parser.add_argument('-d', action='store', default='.', help='The project directory')
    subparsers = parser.add_subparsers()

    # Plot unwrapped interferogram parser
    plot_uifg_subparser = subparsers.add_parser('uifg',
                                                help='Plot an unwrapped interferogram')
    plot_uifg_subparser.set_defaults(func=_parse_unwrapped_ifg_args)
    plot_uifg_subparser.add_argument('-m', '--master', action='store',
                                     help='Master date (YYYYMMDD)', required=True)
    plot_uifg_subparser.add_argument('-s', '--slave', action='store',
                                     help='Slave date (YYYYMMDD)', required=True)
    plot_uifg_subparser.add_argument('-o', '--output', action='store', default=None,
                                     help='Output filename')
    plot_uifg_subparser.add_argument('-r', '--resampled', action='store_true',
                                     help='Plot the resampled interferogram')

    args = parser.parse_args()
    os.chdir(args.d)
    config.load_from_yaml('config.yml')

    args.func(args)
