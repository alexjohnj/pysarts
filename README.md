# PySARTS

PySARTS is a Python package I developed during my undergraduate dissertation
while working with interferometric synthetic aperture radar (InSAR) data. The
package implements methods for working with timeseries of interferograms and
making atmospheric corrections. There's also a variety of plotting methods.

The atmospheric corrections were the focus of my dissertation. I investigated
the use of weather radar for atmospheric corrections. Two of the implemented
corrections were developed during my dissertation: a correction for the liquid
delay using radar data and a correction for the wet and dry delays using the
ERA-interim weather model and radar data. An additional "pure" weather model
correction is also implemented with an implementation based off [TRAIN][train]
(with some differences).

[train]: http://www.davidbekaert.com/

# Intended Audience

I don't expect anyone to use the project for anything professional in its
current form. The documentation is seriously lacking, there's no formal tests
and there's lots of hacks put in place to meet deadlines. If I get time, I'll
try to document the package more thoroughly but otherwise, I don't foresee any
further development.

# Installation

Clone the repository and put it with your other Python packages. Or update your
`PYTHONPATH` environment variable to point at another location where you've
stored the repository.

The file `requirements.txt` lists the project's requirements. You can run `pip
install -r requirements.txt` from the repository to install everything. The main
requirements are the [SciPy stack][scipy], [Numba][numba] and [netCDF][netcdf]
bindings. PySARTS has only been tested with Python 3 but might work with Python
2 too.

[scipy]: https://www.scipy.org/
[numba]: http://numba.pydata.org/
[netcdf]: https://unidata.github.io/netcdf4-python/

# Usage

## The Project Directory

PySARTS works with a project directory. This is a directory containing a file
called `config.yml` that tells PySARTS:

1. The study region.
2. The resolution to work at.
3. The location of your data (if it isn't in the project directory).
4. The master study date.

Check out the `example_config.yml` file in the repository to see what the
configuration looks like. The data paths in the example file are the default
paths. If you put those directories/files in the project directory, you don't
need to configure anything.

## Required Files

As a bare minimum, you'll need the following files:

- A directory containing unwrapped interferograms named
  `SLAVEDATE_MASTERDATE.nc` where dates are formatted as `%Y%m%d`.
- A text file containing the perpendicular baselines for the
  interferograms. This should have three columns: the slave date, master date
  and the perpendicular baseline.

To do anything with weather radar data you'll need

- A directory containing weather radar images (Cartesian, not radial) named
  `%Y%m%d%H%M.nc`.

For any atmospheric corrections, you're going to need

- A DEM of the target region.
- A directory containing weather model output named `%Y%m%d%H%M.nc`.

## Data Formats

PySARTS works with NetCDF 4 files. Everything in the previous section needs to
be in NetCDF format (except the baselines file). 2D data (interferograms, radar
images) needs three variables in the NetCDF:

- Two 1D arrays containing longitudes and latitudes.
- A 2D array containing the data.

These variables can be called either `x`, `y` and `z` OR `lon`, `lat` and
`Band1`. If those variables aren't found, you'll get some cryptic error from
NetCDF.

Weather model output should contain 7 variables. Three of those variables should
be the modelled temperature, relative humidity and geopotential. The variable
names should be the same as those produced by ERA-interim. If you download
output from ERA-interim containing the aforementioned modelled variables, you
should be good to go. **Note**: Only the first time in the model output will be
read. You should store separate model times in separate files.

## Command Line Interface

PySARTS exposes a command line interface from the `pysarts` and `pysarts.plot`
modules. Assuming your `PYTHONPATH` is set correctly, you can run

``` sh
python -m pysarts -h
python -m pysarts.plot -h
```

to get help with the command line interfaces. The help isn't great but it should
be enough to get you going. For more details on how things work, check out the
documentation strings in the source.

# API

All modules under `pysarts` except the `workflow`, `plot` and `config` modules
are designed to be usable from other programs or scripts. They make no
assumptions about the existence of a project directory or any files. You should
read the docstrings of the methods to see how to use them.

# License

MIT. Go wild.
