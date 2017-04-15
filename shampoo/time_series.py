# -*- coding: utf-8 -*-
"""
This module implements storage of holographic time-series
via an HDF5 file.
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from collections import Iterable
import h5py
import numpy as np
import os
from .reconstruction import Hologram, ReconstructedWave

class TimeSeries(h5py.File):
    """
    Holographic time-series as an HDF5 archive.

    Attributes
    ----------
    time_points : tuple of floats
        Time-points in seconds
    wavelengths : tuple of floats
        Wavelengths in nm.
    """
    _default_ckwargs = {'chunks': True, 
                        'compression':'lzf', 
                        'shuffle': True}
    
    @property
    def time_points(self):
        return tuple(self.attrs.get('time_points', default = tuple()))
    
    @property
    def wavelengths(self):
        return tuple(self.attrs.get('wavelengths', default = tuple()))

    @property
    def hologram_group(self):
        return self.require_group('holograms')
    
    @property
    def reconstructed_group(self):
        return self.require_group('reconstructed')

    def add_hologram(self, hologram, time_point = 0):
        """
        Add a hologram to the time-series.

        Parameters
        ----------
        hologram : Hologram

        time_point : float, optional

        Raises
        ------
        ValueError
            If the hologram is not compatible with the current TimeSeries,
            e.g. the wavelengths do not match.
        """
        holo_wavelengths = tuple(np.atleast_1d(hologram.wavelength))

        if len(self.time_points) == 0:
            # This is the first hologram. Reshape all dataset to fit the resolution
            # and number of wavelengths
            # Extend to one time-point at axis 3
            self.attrs['wavelengths'] = holo_wavelengths
            self.attrs['time_points'] = (time_point,)
            self.hologram_group.create_dataset('holograms', 
                                               data = np.expand_dims(np.atleast_3d(hologram.hologram), axis = 3),
                                               dtype = np.float, maxshape = (None, None, 3, None),
                                               **self._default_ckwargs)
            self.reconstructed_group.create_dataset('reconstructed_wave', 
                                                    shape = self.hologram_group['holograms'].shape, 
                                                    dtype = np.complex,  maxshape = (None, None, 3, None),
                                                    **self._default_ckwargs)
            self.reconstructed_group.create_dataset('fourier_mask', 
                                                    shape = self.hologram_group['holograms'].shape,
                                                    maxshape = (None, None, 3, None),
                                                    dtype = np.bool, **self._default_ckwargs)
        
        # The entire TimeSeries has the uniform wavelengths
        if not np.allclose(holo_wavelengths, self.wavelengths):
            raise ValueError('Wavelengths of this hologram ({}) do not match the TimeSeries \
                              wavelengths ({})'.format(holo_wavelengths, self.wavelengths))

        # Find time-point index if it exists
        # Otherwise, insert the time_point at the end
        # WARNING: this means that time_points are not sorted
        # TODO: make sure holograms are always sorted along time-axis?
        if time_point in self.time_points:
            i = self._time_index(time_point)
        else:
            # Resize all datasets along time-axis and include new time-point
            self.attrs['time_points'] = self.time_points + (time_point,)
            self.hologram_group['holograms'].resize(size = len(self.attrs['time_points']), axis = 3)
            self.reconstructed_group['reconstructed_wave'].resize(self.hologram_group['holograms'].shape)
            self.reconstructed_group['fourier_mask'].resize(self.hologram_group['holograms'].shape)

            i = len(self.time_points) - 1

        self.hologram_group['holograms'][:,:,:,i] = np.atleast_3d(hologram.hologram)
    
    def hologram(self, time_point, **kwargs):
        """
        Return Hologram object from archive. Keyword arguments are
        passed to the Hologram constructor.
        
        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        
        Returns
        -------
        out : Hologram
        """
        # Use np.squeeze to remove dimensions of size 1, 
        # i.e. axis 2 for a single wavelength
        dset = self.hologram_group['holograms']
        wavelength = self.wavelengths[0] if len(self.wavelengths) == 1 else self.wavelengths
        arr = np.array(dset[:,:,:,self._time_index(time_point)])
        return Hologram(np.squeeze(arr), wavelength = wavelength, **kwargs)
    
    def reconstructed_wave(self, time_point):
        """
        Returns the ReconstructedWave object from archive. 

        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        
        Returns
        -------
        out : ReconstructedWave object
        """
        # Use np.squeeze to remove dimensions of size 1, 
        # i.e. axis 2 for a single wavelength
        time_index = self._time_index(time_point)
        gp = self.reconstructed_group
        wave = np.array(gp['reconstructed_wave'][:,:,:,time_index])
        mask = np.array(gp['fourier_mask'][:,:,:,time_index])
        return ReconstructedWave(np.squeeze(wave), fourier_mask = np.squeeze(mask))
    
    def reconstruct(self, time_point, propagation_distance, 
                    fourier_mask = None, **kwargs):
        """
        Hologram reconstruction from Hologram.reconstruct(). Keyword arguments
        are also passed to Hologram.reconstruct()
        
        Parameters
        ----------
        time_point : float
            Time-point in seconds.
        propagation_distance : float
            Propagation distance in meters.
        
        Returns
        -------
        out : ReconstructedWave object
            The ReconstructedWave is both stored in the TimeSeries HDF5 file
            and returned to the user. 
        """
        
        # TODO: extend to multiple propagation distances
        hologram = self.hologram(time_point)
        recon_wave = hologram.reconstruct(propagation_distance, 
                                          fourier_mask = fourier_mask)
        
        # TODO: store propagation distance(s)?
        time_index = self._time_index(time_point)
        gp = self.reconstructed_group
        gp['reconstructed_wave'][:,:,:,time_index] = np.atleast_3d(recon_wave.reconstructed_wave)
        gp['fourier_mask'][:,:,:,time_index] = np.atleast_3d(recon_wave.fourier_mask)
        return recon_wave
    
    def batch_reconstruct(self, propagation_distance, fourier_mask = None, **kwargs):
        """ Reconstruct all the holograms stored in the TimeSeries """
        raise NotImplementedError
    
    def _time_index(self, time_point):
        """ Determine the index of the time_point within the TimeSeries time_points"""
        return np.argmin(np.abs(np.array(self.time_points) - time_point))