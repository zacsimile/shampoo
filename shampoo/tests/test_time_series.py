# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os.path
import tempfile

import numpy as np

from ..reconstruction import RANDOM_SEED, Hologram, ReconstructedWave
from ..time_series import TimeSeries

np.random.seed(RANDOM_SEED)

def _example_hologram(dim = 512):
    """ Generate example Hologram object """
    im = 1000*np.ones((dim, dim), dtype = np.float) + np.random.random(size = (dim, dim))
    return Hologram(im)

def test_time_series_metadata_defaults():
    name = os.path.join(tempfile.gettempdir(), 'test_time_series.hdf5')
    with TimeSeries(filename = name, mode = 'w') as time_series:

        # Check default values when newly-created object
        assert time_series.time_points == tuple()
        assert time_series.wavelengths == tuple()

def test_time_series_storing_hologram_single_wavelength():
    """ Test storage of holograms with a single wavelength """
    name = os.path.join(tempfile.gettempdir(), 'test_time_series.hdf5')
    hologram = _example_hologram()
    with TimeSeries(filename = name, mode = 'w') as time_series:
        time_series.add_hologram(hologram, time_point = 0)
        time_series.add_hologram(hologram, time_point = 1)

        assert time_series.time_points == (0,1)

        retrieved = time_series.hologram(0)
        assert isinstance(retrieved, Hologram)
        assert np.allclose(hologram.hologram, retrieved.hologram)
        assert np.allclose(time_series.wavelengths, hologram.wavelength)

def test_time_series_storing_hologram_three_wavelength():
    """ Test storage of holograms with three wavelengths """
    name = os.path.join(tempfile.gettempdir(), 'test_time_series.hdf5')
    hologram = Hologram(np.zeros(shape = (512, 512, 3), dtype = np.float), 
                        wavelength = [1,2,3])
    with TimeSeries(filename = name, mode = 'w') as time_series:
        time_series.add_hologram(hologram, time_point = 0)
        time_series.add_hologram(hologram, time_point = 1)

        assert np.allclose(time_series.time_points, (0,1))
        assert np.allclose(time_series.wavelengths, (1,2,3))

        assert np.allclose(hologram.hologram, time_series.hologram(0).hologram)
        assert np.allclose(hologram.hologram, time_series.hologram(1).hologram)

def test_time_series_reconstruct():
    #raise AssertionError('Skipping the test due to unknown bug')
    name = os.path.join(tempfile.gettempdir(), 'test_time_series.hdf5')
    hologram = _example_hologram()
    with TimeSeries(filename = name, mode = 'w') as time_series:
        time_series.add_hologram(hologram, time_point = 0)
        ts_reconw = time_series.reconstruct(time_point = 0,
                                            propagation_distance = 1) 
        assert isinstance(ts_reconw, ReconstructedWave) 
        
        # Retrieve reconstructed wave from archive
        archived_reconw = time_series.reconstructed_wave(time_point = 0)
        
        assert isinstance(archived_reconw, ReconstructedWave)

        assert np.allclose(ts_reconw.reconstructed_wave, 
                           archived_reconw.reconstructed_wave)
