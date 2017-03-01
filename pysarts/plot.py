"""Module for interactive plotting of interferograms.

"""
import os
from datetime import date, datetime
import logging

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np
import yaml

from . import processing
from . import workflow
from . import nimrod
from . import util

if __name__ == '__main__':
    from . import config
    import argparse


def _parse_unwrapped_ifg_args(args):
    if args.time_series:
        plot_time_series_ifg(args.master, args.slave, args.output)
    else:
        plot_unwrapped_ifg(args.master, args.slave, args.output, args.resampled)


def _plot_all_uifgs(args):
    ifg_paths = workflow.find_ifgs()
    date_pairings = [util.extract_timestamp_from_ifg_name(path) for path in ifg_paths]
    for (master_date, slave_date) in date_pairings:
        logging.info('Plotting master/slave pairing: {} / {}'.format(master_date, slave_date))
        if args.output:
            output_file = os.path.join(args.output,
                                       (slave_date.strftime('%Y%m%d')
                                        + '_'
                                        + master_date.strftime('%Y%m%d')
                                        + '.png'))
            plot_unwrapped_ifg(master_date, slave_date, output_file, args.resampled)
        else:
            plot_unwrapped_ifg(master_date, slave_date, None, args.resampled)


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
        master_date = master_date.strftime("%Y%m%d")
    if isinstance(slave_date, date):
        slave_date = slave_date.strftime("%Y%m%d")

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

    fig, _ = plot_ifg(ifg)
    if fname:
        fig.savefig(fname, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()

    return None

def plot_ifg(ifg, axes=None):
    """Plot an ifg dictionary. Returns a figure handle"""
    if axes:
        fig = axes.get_figure()
    else:
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)

    bmap = Basemap(llcrnrlon=ifg['lons'][0],
                   llcrnrlat=ifg['lats'][0],
                   urcrnrlon=ifg['lons'][-1],
                   urcrnrlat=ifg['lats'][-1],
                   resolution='f',
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

    cbar = fig.colorbar(image, pad=0.07, ax=axes)
    cbar.set_label('LOS Delay / cm')

    title = 'Unwrapped Interferogram\nMaster: {0}\nSlave: {1}'.format(
        ifg['master_date'].strftime('%Y-%m-%d'),
        ifg['slave_date'].strftime('%Y-%m-%d'))
    axes.set_title(title)
    fig.tight_layout()

    return fig, bmap

def plot_time_series_ifg(master_date, slave_date, fname=None):
    if isinstance(master_date, date):
        master_date = master_date.strftime('%Y%m%d')

    if isinstance(slave_date, date):
        slave_date = slave_date.strftime('%Y%m%d')

    lons, lats = workflow.read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                           'grid.txt'))
    ifg_ts = np.load(os.path.join(config.SCRATCH_DIR,
                                  'uifg_ts',
                                  master_date + '.npy'),
                     mmap_mode='r')
    slave_date_idx = 0
    with open(os.path.join(config.SCRATCH_DIR,
                           'uifg_ts',
                           master_date + '.yml')) as f:
        ts_date_indexes = yaml.safe_load(f)
        slave_date_date = datetime.strptime(slave_date, '%Y%m%d').date()
        slave_date_idx = ts_date_indexes.index(slave_date_date)

    ifg = {
        'lons': lons,
        'lats': lats,
        'data': ifg_ts[:, :, slave_date_idx],
        'master_date': datetime.strptime(master_date, '%Y%m%d').date(),
        'slave_date': datetime.strptime(slave_date, '%Y%m%d').date()
    }

    fig, _ = plot_ifg(ifg)
    if fname:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
        plt.close()

def _plot_master_atmosphere(args):
    """Plot the master atmosphere from the command line."""
    if args.master_date:
        plot_master_atmosphere(args.master_date, args.output)
    else:
        plot_master_atmosphere(config.MASTER_DATE.date(), args.output)

def plot_master_atmosphere(master_date, fname=None):
    """Plot the master atmosphere for a given date."""
    if isinstance(master_date, date):
        master_date = master_date.strftime('%Y%m%d')

    # Load the master atmosphere
    master_atmosphere = np.load(os.path.join(config.SCRATCH_DIR,
                                             'master_atmosphere',
                                             master_date + '.npy'),
                                mmap_mode='r')
    lons, lats = workflow.read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                           'grid.txt'))

    ifg = {
        'lons': lons,
        'lats': lats,
        'data': master_atmosphere,
        'master_date': datetime.strptime(master_date, '%Y%m%d').date(),
        'slave_date': datetime.today().date() # Dummy value
    }

    fig, _ = plot_ifg(ifg)
    axes = fig.get_axes()[0]
    axes.set_title('Master Atmosphere\n{}'.format(ifg['master_date'].strftime('%Y-%m-%d')))
    if fname:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
def _plot_master_atmosphere_vs_rainfall(args):
    """Interactively plot master atmosphere vs. rainfall rate."""
    if args.master_date:
        plot_master_atmosphere_vs_rainfall(args.master_date, args.rain_tol, args.output)
    else:
        plot_master_atmosphere_vs_rainfall(config.MASTER_DATE, args.rain_tol, args.output)

def plot_master_atmosphere_vs_rainfall(master_date, rain_tol=0, fname=None):
    if isinstance(master_date, str):
        master_date = datetime.strptime(master_date, '%Y%m%dT%H%M')

    # Load master atmosphere for the date
    master_atmosphere = np.load(os.path.join(config.SCRATCH_DIR,
                                             'master_atmosphere',
                                             master_date.strftime('%Y%m%d') + '.npy'),
                                mmap_mode='r')
    lons, lats = workflow.read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                           'grid.txt'))
    ifg = {
        'lons': lons,
        'lats': lats,
        'data': master_atmosphere,
        'master_date': master_date
    }

    # Load weather radar for date
    wr = nimrod.load_from_netcdf(os.path.join(config.WEATHER_RADAR_DIR,
                                              master_date.strftime('%Y'),
                                              master_date.strftime('%m'),
                                              master_date.strftime('%Y%m%d%H%M') + '.nc'))
    lon_bounds = (np.amin(lons), np.amax(lons))
    lat_bounds = (np.amin(lats), np.amax(lats))
    nimrod.clip_wr(wr, lon_bounds, lat_bounds)
    wr = nimrod.resample_wr(wr, lons, lats)
    wr_above_tol_idxs = np.where(wr['data'].ravel() > rain_tol)

    fig = plt.figure()
    axes = fig.add_subplot(1, 1, 1)
    axes.scatter(wr['data'].ravel()[wr_above_tol_idxs],
                 ifg['data'].ravel()[wr_above_tol_idxs]*10**16,
                 s=1)
    axes.set_title('Rainfall Scatter ({})'.format(master_date.strftime('%Y-%m-%d')))
    axes.set_xlabel(r'Rainfall / mm hr$^{-1}$')
    axes.set_ylabel(r'Residual LOS Delay / cm $\left(\times 10^{-16}\right)$')

    if fname:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()


def plot_wr(wr, axes=None):
    """Plot a weather radar dictionary. Returns a figure handle and a basemap object."""
    if axes:
        fig = axes.get_figure()
    else:
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)

    bmap = Basemap(llcrnrlon=wr['lons'][0],
                   llcrnrlat=wr['lats'][0],
                   urcrnrlon=wr['lons'][-1],
                   urcrnrlat=wr['lats'][-1],
                   resolution='h',
                   projection='merc',
                   ax=axes)

    parallels = np.linspace(wr['lats'][0], wr['lats'][-1], 5)
    meridians = np.linspace(wr['lons'][0], wr['lons'][-1], 5)

    bmap.drawcoastlines()
    bmap.drawparallels(parallels, labels=[True, False, False, False],
                       fmt="%.2f", fontsize=9)
    bmap.drawmeridians(meridians, labels=[False, False, False, True],
                       fmt="%.2f", fontsize=9)
    bmap.drawmapboundary()

    lon_mesh, lat_mesh = np.meshgrid(wr['lons'], wr['lats'])
    image = bmap.pcolormesh(lon_mesh,
                            lat_mesh,
                            np.ma.masked_values(wr['data'], 0),
                            latlon=True,
                            cmap=cm.Spectral_r,
                            vmin=0)

    cbar = fig.colorbar(image, pad=0.07, ax=axes)
    cbar.set_label(r'Rainfall / mm hr$^{-1}$')

    title = ('Rainfall Radar Image\n({0})'
             .format(wr['date'].strftime('%Y-%m-%dT%H:%M')))

    axes.set_title(title)
    fig.tight_layout()

    return (fig, bmap)


def _plot_weather(args):
    if args.date:
        plot_weather(args.date, args.full, args.output)
    else:
        plot_weather(config.MASTER_DATE, args.full, args.output)


def plot_weather(wr_date, full=False, fname=None):
    """Plot a weather radar image.
    """
    if isinstance(wr_date, str):
        wr_date = datetime.strptime(wr_date, '%Y%m%dT%H%M')

    # Load the weather radar image
    wr_path = workflow.find_closest_weather_radar_file(wr_date)
    wr = nimrod.load_from_netcdf(wr_path)

    if not full:
        # Clip image to target region
        lon_bounds = (config.REGION['lon_min'], config.REGION['lon_max'])
        lat_bounds = (config.REGION['lat_min'], config.REGION['lat_max'])
        nimrod.clip_wr(wr, lon_bounds, lat_bounds)

    fig, _ = plot_wr(wr)

    if fname:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
        plt.close()


def _plot_profile(args):
    plot_profile(config.MASTER_DATE, args.longitude, args.output)

def plot_profile(master_date, longitude, fname=None):
    if isinstance(master_date, str):
        master_date = datetime.strptime(master_date, '%Y%m%dT%H%M')

    fig = plt.figure()
    ifg_ax = plt.subplot2grid((2, 2), (0, 0))
    wr_ax = plt.subplot2grid((2, 2), (0, 1))
    profile_ax = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    # Load master atmosphere for master date.
    master_atmosphere = np.load(os.path.join(config.SCRATCH_DIR,
                                             'master_atmosphere',
                                             master_date.strftime('%Y%m%d') + '.npy'),
                                mmap_mode='r')
    lons, lats = workflow.read_grid_from_file(os.path.join(config.SCRATCH_DIR,
                                                           'grid.txt'))

    ifg = {
        'lons': lons,
        'lats': lats,
        'data': master_atmosphere,
        'master_date': master_date,
        'slave_date': datetime.today().date()
    }

    # Load weather radar image and clip.
    lon_bounds = (np.amin(ifg['lons']), np.amax(ifg['lons']))
    lat_bounds = (np.amin(ifg['lats']), np.amax(ifg['lats']))
    wr = nimrod.load_from_netcdf(workflow.find_closest_weather_radar_file(master_date))
    nimrod.clip_wr(wr, lon_bounds, lat_bounds)

    # Plot and configure IFG
    _, bmap_ifg = plot_ifg(ifg, axes=ifg_ax)
    ifg_ax.set_title('Master Atmosphere\n({})'.format(master_date.strftime('%Y-%m-%dT%H:%M')))
    bmap_ifg.plot([longitude, longitude], lat_bounds, latlon=True, linewidth=2, color='white')
    bmap_ifg.plot([longitude, longitude], lat_bounds, latlon=True, linewidth=1)

    # Plot and configure weather radar image
    _, bmap_wr = plot_wr(wr, axes=wr_ax)
    bmap_wr.plot([longitude, longitude], lat_bounds, latlon=True, linewidth=2, color='white')
    bmap_wr.plot([longitude, longitude], lat_bounds, latlon=True, linewidth=1)

    # Plot the profile
    ## LOS Delay
    ifg_lon_idx = np.argmin(np.absolute(ifg['lons'] - longitude))
    wr_lon_idx = np.argmin(np.absolute(wr['lons'] - longitude))
    profile_ax.plot(ifg['lats'], ifg['data'][:, ifg_lon_idx])
    profile_ax.set_ylabel('LOS Delay / cm')
    profile_ax.set_xlabel(r'Latitude / $\degree$')

    ## Rainfall
    profile_ax_rain = profile_ax.twinx()
    profile_ax_rain.plot(wr['lats'], wr['data'][:, wr_lon_idx], color='orange')
    profile_ax_rain.tick_params('y', colors='orange')
    profile_ax_rain.set_ylabel(r'Rainfall / mm hr$^{-1}$')

    if fname:
        fig.savefig(fname, bbox_inches='tight')
    else:
        plt.show()
        plt.close()


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
    plot_uifg_subparser.add_argument('-t', '--time-series', action='store_true',
                                     help='Plot inverted time series interferogram')

    plot_uifg_all_subparser = subparsers.add_parser('uifg-all',
                                                    help='Plot all unwrapped interferograms')
    plot_uifg_all_subparser.set_defaults(func=_plot_all_uifgs)
    plot_uifg_all_subparser.add_argument('-r', '--resampled', action='store_true',
                                         help='Plot resampled/clipped interferograms')
    plot_uifg_all_subparser.add_argument('-o', '--output', action='store', default=None,
                                         help='Directory to save plots to')

    plot_master_atmosphere_subparser = subparsers.add_parser('master-atmos',
                                                             help='Plot master atmosphere for a date')
    plot_master_atmosphere_subparser.set_defaults(func=_plot_master_atmosphere)
    plot_master_atmosphere_subparser.add_argument('master_date', default=None, nargs='?')
    plot_master_atmosphere_subparser.add_argument('-o', '--output', action='store', default=None,
                                                  help='Output file name')

    plot_rainfall_correlation_subparser = subparsers.add_parser('radar-correlation',
                                                                help='Plot radar rainfall correlation for a date')
    plot_rainfall_correlation_subparser.set_defaults(func=_plot_master_atmosphere_vs_rainfall)
    plot_rainfall_correlation_subparser.add_argument('master_date', default=None, nargs='?')
    plot_rainfall_correlation_subparser.add_argument('-o', '--output', action='store', default=None,
                                                     help='Output file name')
    plot_rainfall_correlation_subparser.add_argument('-r', '--rain-tol', default=0, type=float, action='store',
                                                     help='Minimum rainfall level to plot in scatter chart.')

    plot_radar_rainfall_subparser = subparsers.add_parser('weather',
                                                          help='Plot a rainfall radar image.')
    plot_radar_rainfall_subparser.set_defaults(func=_plot_weather)
    plot_radar_rainfall_subparser.add_argument('date', default=None, nargs='?',
                                               help=('The date and time (HH:MM) to plot'
                                                     + ' the weather for in ISO8601 format'))
    plot_radar_rainfall_subparser.add_argument('-o', '--output',
                                               action='store',
                                               default=None,
                                               help='Output file name')
    plot_radar_rainfall_subparser.add_argument('-f', '--full',
                                               action='store_true',
                                               help='Plot the entire radar image instead of just the project region.')

    plot_profile_subparser = subparsers.add_parser('profile',
                                                  help='Plot atmosphere LOS delay and rainfall along a profile')
    plot_profile_subparser.set_defaults(func=_plot_profile)
    plot_profile_subparser.add_argument('--longitude',
                                        action='store',
                                        default=None,
                                        help='The line of longitude to plot along.',
                                        nargs=1,
                                        type=float,
                                        required=True)
    plot_profile_subparser.add_argument('-o', '--output',
                                        action='store',
                                        default=None,
                                        help='Output file name')


    args = parser.parse_args()
    os.chdir(args.d)
    config.load_from_yaml('config.yml')
    logging.basicConfig(level=config.LOG_LEVEL)

    args.func(args)
