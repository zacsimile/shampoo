"""
This module handles reconstruction of phase and intensity images from raw
holograms using "the convolution approach": see Section 3.3 of Schnars & Juptner
(2002) Meas. Sci. Technol. 13 R85-R101 [1]_.

Aberration corrections from Colomb et al., Appl Opt. 2006 Feb 10;45(5):851-63
are applied [2]_.

    .. [1] http://x-ray.ucsd.edu/mediawiki/images/d/df/Digital_recording_numerical_reconstruction.pdf
    .. [2] http://www.ncbi.nlm.nih.gov/pubmed/16512526

"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import Iterable, deque

import warnings
from multiprocessing.dummy import Pool as ThreadPool

from .vis import save_scaled_image

import h5py
import numpy as np
from numpy.compat import integer_types

from scipy.ndimage import gaussian_filter
from scipy.signal import tukey

from skimage.restoration import unwrap_phase as skimage_unwrap_phase
from skimage.io import imread
from skimage.feature import blob_doh

from astropy.utils.exceptions import AstropyUserWarning
from astropy.convolution import convolve_fft, MexicanHat2DKernel

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Try importing optional dependency PyFFTW for Fourier transforms. If the import
# fails, import scipy's FFT module instead
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
except ImportError:
    from scipy.fftpack import fft2, ifft2

__all__ = ['Hologram', 'ReconstructedWave', 'unwrap_phase']
RANDOM_SEED = 42
TWO_TO_N = [2**i for i in range(13)]


def rebin_image(a, binning_factor):
    # Courtesy of J.F. Sebastian: http://stackoverflow.com/a/8090605
    if binning_factor == 1:
        return a

    new_shape = (a.shape[0]/binning_factor, a.shape[1]/binning_factor)
    sh = (new_shape[0], a.shape[0]//new_shape[0], new_shape[1],
          a.shape[1]//new_shape[1])
    return a.reshape(map(int, sh)).mean(-1).mean(1)

def fftshift(x, additional_shift=None, axes=None):
    """
    Shift the zero-frequency component to the center of the spectrum, or with
    some additional offset from the center.

    This is a more generic fork of `~numpy.fft.fftshift`, which doesn't support
    additional shifts.


    Parameters
    ----------
    x : array_like
        Input array.
    additional_shift : list of length ``M``
        Desired additional shifts in ``x`` and ``y`` directions respectively
    axes : int or shape tuple, optional
        Axes over which to shift.  Default is None, which shifts all axes.

    Returns
    -------
    y : `~numpy.ndarray`
        The shifted array.
    """
    tmp = np.asarray(x)
    ndim = len(tmp.shape)
    if axes is None:
        axes = list(range(ndim))
    elif isinstance(axes, integer_types):
        axes = (axes,)

    # If no additional shift is supplied, reproduce `numpy.fft.fftshift` result
    if additional_shift is None:
        additional_shift = [0, 0]

    y = tmp
    for k, extra_shift in zip(axes, additional_shift):
        n = tmp.shape[k]
        if (n+1)//2 - extra_shift < n:
            p2 = (n+1)//2 - extra_shift
        else:
            p2 = abs(extra_shift) - (n+1)//2
        mylist = np.concatenate((np.arange(p2, n), np.arange(0, p2)))
        y = np.take(y, mylist, k)
    return y

def _load_hologram(hologram_path):
    """
    Load a hologram from path ``hologram_path`` using scikit-image and numpy.
    """
    return np.array(imread(hologram_path, plugin = 'tifffile'), dtype=np.float64)


def _find_peak_centroid(image, gaussian_width=10):
    """
    Smooth the image, find centroid of peak in the image.
    """
    image = np.atleast_3d(image)

    # Do not filter along axis 2, i.e. across different wavelengths.
    smoothed_image = gaussian_filter(image, [gaussian_width, gaussian_width, 0])
    smoothed_image = np.reshape(smoothed_image, newshape = (-1, smoothed_image.shape[2]) )
    return np.array(np.unravel_index(smoothed_image.argmax(axis = 0), image.shape[0:2]))


def _crop_image(image, crop_fraction):
    """
    Crop an image by a factor of ``crop_fraction``.
    """
    if crop_fraction == 0:
        return image

    crop_length = int(image.shape[0] * crop_fraction)

    if crop_length not in TWO_TO_N:
        message = ("Final dimensions after crop should be a power of 2^N. "
                   "Crop fraction of {0} yields dimensions ({1}, {1})"
                   .format(crop_fraction, crop_length))
        warnings.warn(message, CropEfficiencyWarning)

    cropped_image = image[crop_length//2:crop_length//2 + crop_length,
                          crop_length//2:crop_length//2 + crop_length]
    return cropped_image


def _crop_to_square(image):
    """
    Ensure that hologram is square.
    """
    sh = image.shape
    if sh[0] != sh[1]:
        square_image = image[:min(sh), :min(sh)]
    else:
        square_image = image

    return square_image


class CropEfficiencyWarning(AstropyUserWarning):
    pass


class Hologram(object):
    """
    Container for holograms and methods to reconstruct them.
    """
    def __init__(self, hologram, crop_fraction=None, wavelength=405e-9,
                 rebin_factor=1, dx=3.45e-6, dy=3.45e-6):
        """
        Parameters
        ----------
        hologram : `~numpy.ndarray`
            Input hologram. If the hologram was taken with multiple wavelengths,
            the array should be a stack of single-wavelength hologram along axis 2.
        crop_fraction : float
            Fraction of the image to crop for analysis
        wavelength : float [meters] or iterable
            Wavelength of laser. Multiple wavelengths can be given as well.
        rebin_factor : int
            Rebin the image by factor ``rebin_factor``. Must be an even integer.
        dx : float [meters]
            Pixel width in x-direction (unbinned)
        dy : float [meters]
            Pixel width in y-direction (unbinned)

        Notes
        -----
        Non-square holograms will be cropped to a square with the dimensions of
        the smallest dimension.

        Raises
        ------
        ValueError
            Raised if the number of wavelengths does not match the input hologram 
            array's third dimension
        """
        wavelength = np.atleast_1d(wavelength)

        if len(wavelength) != np.atleast_3d(hologram).shape[2]:
            raise ValueError('Number of wavelengths {} does not match the dimensions of the  \
                              input hologram {}'.format(len(wavelength), hologram.shape))

        self.crop_fraction = crop_fraction
        self.rebin_factor = rebin_factor

        # Rebin the hologram
        square_hologram = _crop_to_square(np.float64(hologram))
        binned_hologram = rebin_image(square_hologram, self.rebin_factor)

        # Crop the hologram by factor crop_factor, centered on original center
        if crop_fraction is not None:
            self.hologram = _crop_image(binned_hologram, crop_fraction)
        else:
            self.hologram = binned_hologram
        
        # To generalize to multiple wavelengths,
        # wavelengths are stored with shape (1,1,N)
        self.n = self.hologram.shape[0]
        self.wavelength = wavelength.reshape((1,1,-1))
        self.wavenumber = 2*np.pi/self.wavelength
        self.reconstructions = dict()
        self.dx = dx*rebin_factor
        self.dy = dy*rebin_factor
        self.mgrid = np.mgrid[0:self.n, 0:self.n]
        self.random_seed = RANDOM_SEED
        self.apodization_window_function = None

    @classmethod
    def from_tif(cls, hologram_path, **kwargs):
        """
        Load a hologram from a TIF file.

        This class method takes the path to the TIF file as the first argument.
        All other arguments are the same as `~shampoo.Hologram`.

        Parameters
        ----------
        hologram_path : str
            Path to the hologram to load
        """
        hologram = _load_hologram(hologram_path)
        return cls(hologram, **kwargs)

    def reconstruct(self, propagation_distance, fourier_mask=None, cache=True):
        """
        Reconstruct the wave at ``propagation_distance``.

        If ``cache`` is `True`, the reconstructed wave will be cached onto the
        `~shampoo.reconstruction.Hologram` object for quick retrieval.

        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]
        fourier_mask : array_like or None, optional
            Fourier-domain mask. If None (default), a mask is determined from the position of the main
            spectral peak.
        cache : bool, optional
            Cache reconstructions onto the hologram object? Default is False. NOTE: This argument
            has not effect for now.

        Returns
        -------
        reconstructed_wave : `~shampoo.reconstruction.ReconstructedWave`
            The reconstructed wave.
        """
        #######
        # TODO: different fourier masks might be in use; disabled cache until figured out
        cache = False
        #######

        if cache:
            
            # Cache dictionary is accessible by keys = propagation distances
            cache_key = propagation_distance

            # If this reconstruction is cached, get it.
            if cache_key in self.reconstructions:
                reconstructed_wave = self.reconstructions[cache_key]

            # If this reconstruction is not cached, calculate it and cache it
            else:
                reconstructed_wave, mask = self._reconstruct(propagation_distance,
                                                             fourier_mask=fourier_mask)
                self.reconstructions[cache_key] = reconstructed_wave

        else:
            reconstructed_wave, mask = self._reconstruct(propagation_distance,
                                                         fourier_mask=fourier_mask)

        return ReconstructedWave(reconstructed_wave, fourier_mask = mask)

    def _reconstruct(self, propagation_distance, fourier_mask=None):
        """
        Reconstruct wave from hologram stored in file ``hologram_path`` at
        propagation distance ``propagation_distance``.

        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]
        fourier_mask : array_like or None, optional
            Fourier-domain mask. If None (default), a mask is determined from the position of the main
            spectral peak. If array_like, the array will be cast to boolean.

        Returns
        -------
        reconstructed_wave : `~numpy.ndarray` (complex)
            Reconstructed wave from hologram
        mask : `~numpy.ndarray` (bool)
            Fourier-domain mask used in the reconstruction.
        """
        # Read input image
        apodized_hologram = self.apodize(self.hologram)

        # Isolate the real image in Fourier space, find spectral peak
        # Treat multiple wavelengths individually
        ft_hologram = fft2(apodized_hologram, axes = (0, 1))

        # Determine location of spectral peak
        if self.rebin_factor != 1:
            mask_radius = 150./self.rebin_factor
        elif self.crop_fraction is not None and self.crop_fraction != 0:
            mask_radius = 150./abs(np.log(self.crop_fraction)/np.log(2))
        else:
            mask_radius = 150.
        
        # Due to fourier peaks being different for each wavelength,
        # x_peak and y_peaks are arrays in general
        x_peak, y_peak = self.fourier_peak_centroid(ft_hologram, mask_radius)
        
        # Either use a Fourier-domain mask based on coords of spectral peak,
        # or a user-specified mask
        if fourier_mask is None:
            mask = self.real_image_mask(x_peak, y_peak, mask_radius)
        else:
            mask = np.asarray(fourier_mask, dtype=np.bool)
        mask = np.atleast_3d(mask)

        # Calculate Fourier transform of impulse response function
        G = self.fourier_trans_of_impulse_resp_func(propagation_distance)

        # Now calculate digital phase mask. First center the spectral peak for each channel
        x_peak, y_peak = x_peak.reshape(-1), y_peak.reshape(-1)
        shifted_ft_hologram = np.empty_like(ft_hologram)
        for channel in range(shifted_ft_hologram.shape[2]):
            shifted_ft_hologram[:,:,channel] = fftshift(ft_hologram[:,:,channel] * mask[:,:,channel],
                                                        additional_shift=[-x_peak[channel], 
                                                                          -y_peak[channel]],
                                                        axes = (0,1))

        # Apodize the result
        psi = self.apodize(shifted_ft_hologram * G)
        digital_phase_mask = self.get_digital_phase_mask(psi)

        # Reconstruct the image
        # fftshift is independent of channel
        psi = np.empty_like(shifted_ft_hologram)
        _ft = fft2(apodized_hologram * digital_phase_mask, axes = (0,1)) * mask
        for channel in range(psi.shape[2]):
            psi[:,:,channel] = fftshift(_ft[:,:,channel], 
                                        additional_shift=[-x_peak[channel], 
                                                          -y_peak[channel]],
                                        axes = (0,1))
        psi *= G
        
        reconstructed_wave = fftshift(ifft2(psi, axes = (0,1)), axes = (0,1))
        return reconstructed_wave, mask

    def get_digital_phase_mask(self, psi):
        """
        Calculate the digital phase mask (i.e. reference wave), as in Colomb et
        al. 2006, Eqn. 26 [1]_.

        Fit for a second order polynomial, numerical parametric lens with least
        squares to remove tilt, spherical aberration.

        .. [1] http://www.ncbi.nlm.nih.gov/pubmed/16512526

        Parameters
        ----------
        psi : `~numpy.ndarray`
            The product of the Fourier transform of the hologram and the Fourier
            transform of impulse response function

        Returns
        -------
        phase_mask : `~numpy.ndarray`
            Digital phase mask, used for correcting phase aberrations.
        """
        inverse_psi = fftshift(ifft2(psi, axes = (0 ,1)), axes = (0, 1))

        unwrapped_phase_image = unwrap_phase(inverse_psi)/2/self.wavenumber
        smooth_phase_image = gaussian_filter(unwrapped_phase_image, [50, 50, 0]) # do not filter along axis 2

        high = np.percentile(unwrapped_phase_image, 99)
        low = np.percentile(unwrapped_phase_image, 1)

        smooth_phase_image[high < unwrapped_phase_image] = high
        smooth_phase_image[low > unwrapped_phase_image] = low

        # Fit the smoothed phase image with a 2nd order polynomial surface with
        # mixed terms using least-squares.
        # This is iterated over all wavelength channels separately
        # TODO: can this be done on the smooth_phase_image along axis 2 instead
        # of direct iteration?
        channels = np.split(smooth_phase_image, smooth_phase_image.shape[2], axis = 2)
        fits = list()

        # Need to flip mgrid indices for this least squares solution
        y, x = self.mgrid - self.n/2
        x, y = np.squeeze(x), np.squeeze(y)

        for channel in channels:
            v = np.array([np.ones(len(x[0, :])), x[0, :], y[:, 0], x[0, :]**2,
                        x[0, :] * y[:, 0], y[:, 0]**2])
            coefficients = np.linalg.lstsq(v.T, np.squeeze(channel))[0]
            fits.append(np.dot(v.T, coefficients))
        
        field_curvature_mask = np.stack(fits, axis = 2)
        digital_phase_mask = np.exp(-1j*self.wavenumber * field_curvature_mask)

        return digital_phase_mask

    def apodize(self, array, alpha=0.075):
        """
        Force the magnitude of an array to go to zero at the boundaries.

        Parameters
        ----------
        array : `~numpy.ndarray`
            Array to apodize
        alpha : float between zero and one
            Alpha parameter for the Tukey window function. For best results,
            keep between 0.075 and 0.2.

        Returns
        -------
        apodized_arr : `~numpy.ndarray`
            Apodized array
        """
        if self.apodization_window_function is None:
            x, y = self.mgrid
            n = len(x[0])
            tukey_window = tukey(n, alpha)
            self.apodization_window_function = np.atleast_3d(tukey_window[:, np.newaxis] * tukey_window)
        
        # In the most general case, array might represent a multi-wavelength hologram
        apodized_array = np.atleast_3d(array) * self.apodization_window_function
        return apodized_array

    def fourier_trans_of_impulse_resp_func(self, propagation_distance):
        """
        Calculate the Fourier transform of impulse response function, sometimes
        represented as ``G`` in the literature.

        For reference, see Eqn 3.22 of Schnars & Juptner (2002) Meas. Sci.
        Technol. 13 R85-R101 [1]_,

        .. [1] http://x-ray.ucsd.edu/mediawiki/images/d/df/Digital_recording_numerical_reconstruction.pdf

        Parameters
        ----------
        propagation_distance : float
            Propagation distance [m]

        Returns
        -------
        G : `~numpy.ndarray`
            Fourier transform of impulse response function
        """
        x, y = self.mgrid - self.n/2
        x, y = np.atleast_3d(x), np.atleast_3d(y)
        first_term = (self.wavelength**2 * (x + self.n**2 * self.dx**2 /
                      (2.0 * propagation_distance * self.wavelength))**2 /
                      (self.n**2 * self.dx**2))
        second_term = (self.wavelength**2 * (y + self.n**2 * self.dy**2 /
                       (2.0 * propagation_distance * self.wavelength))**2 /
                       (self.n**2 * self.dy**2))
        G = np.exp(-1j * self.wavenumber * propagation_distance *
                   np.sqrt(1.0 - first_term - second_term))
        return G

    def real_image_mask(self, center_x, center_y, radius):
        """
        Calculate the Fourier-space mask to isolate the real image
    
        Parameters
        ----------
        center_x : `~numpy.ndarray`
            ``x`` centroid [pixels] of real image in Fourier space for each 
            image in a stack.
        center_y : `~numpy.ndarray`
            ``y`` centroid [pixels] of real image in Fourier space for each
            image in a stack.
        radius : float
            Radial width of mask [pixels] to apply to the real image in Fourier
            space
    
        Returns
        -------
        mask : `~numpy.ndarray`
            Binary-valued mask centered on the real-image peak in the Fourier
            transform of the hologram.
        """
        center_x, center_y = np.reshape(center_x, (1, 1, -1)), np.reshape(center_y, (1, 1, -1))
        x, y = self.mgrid
        x, y = x[:,:,None], y[:,:,None]
        mask = np.zeros_like(np.atleast_3d(self.hologram), dtype = np.bool)
        mask[(x-center_x)**2 + (y-center_y)**2 < radius**2] = True

        # exclude corners
        buffer = 20
        mask[(x < buffer) | (y < buffer) |
             (x > len(x[0]) - buffer) | (y > len(y[1]) - buffer)] = 0.0

        return mask
    
    def fourier_peak_centroid(self, fourier_arr, mask_radius=None,
                              margin_factor=0.1, plot=False):
        """
        Calculate the centroid of the signal spike in Fourier space near the
        frequencies of the real image.

        Parameters
        ----------
        fourier_arr : `~numpy.ndarray`
            Fourier-transform of the hologram
        margin_factor : int
            Fraction of the length of the Fourier-transform of the hologram
            to ignore near the edges, where spurious peaks occur there.

        Returns
        -------
        pixel : `~numpy.ndarray`
            Pixel at the centroid of the spike in Fourier transform of the
            hologram near the real image.
        """
        margin = int(self.n*margin_factor)
        #abs_fourier_arr = np.abs(fourier_arr)[margin:-margin, margin:-margin]
        abs_fourier_arr = np.abs(fourier_arr)[margin:self.n//2, margin:-margin, :]
        return _find_peak_centroid(abs_fourier_arr, gaussian_width=10) + margin

    def reconstruct_multithread(self, propagation_distances, threads=4, fourier_mask=None):
        """
        Reconstruct phase or intensity for multiple distances, for one hologram.

        Parameters
        ----------
        propagation_distances : `~numpy.ndarray` or list
            Propagation distances to reconstruct
        threads : int
            Number of threads to use via `~multiprocessing`
        fourier_mask : array_like or None, optional
            Fourier-domain mask. If None (default), a mask is determined from the position of the main
            spectral peak. If array_like, the array will be cast to boolean.

        Returns
        -------
        reconstructed : ReconstructedWave
        """

        n_z_slices = len(propagation_distances)

        wave_shape = self.hologram.shape
        wave_cube = np.zeros((n_z_slices, wave_shape[0], wave_shape[1]),
                               dtype=np.complex128)
        mask_cube = np.empty_like(wave_cube, dtype = np.bool)

        def _reconstruct(index):
            # Reconstruct image, add to data cube
            wave = self.reconstruct(propagation_distances[index], 
                                    fourier_mask = fourier_mask)
            wave_cube[index, ...] = wave._reconstructed_wave
            mask_cube[index, ...] = wave.fourier_mask

        # Make the Pool of workers
        pool = ThreadPool(threads)
        pool.map(_reconstruct, range(n_z_slices))

        # close the pool and wait for the work to finish
        pool.close()
        pool.join()

        return ReconstructedWave(wave_cube, fourier_mask = mask_cube)


def unwrap_phase(reconstructed_wave, seed=RANDOM_SEED):
    """
    2D phase unwrap a complex reconstructed wave.

    Essentially a wrapper around the `~skimage.restoration.unwrap_phase`
    function.

    Parameters
    ----------
    reconstructed_wave : `~numpy.ndarray`
        Complex reconstructed wave
    seed : float (optional)
        Random seed, optional.

    Returns
    -------
    `~numpy.ndarray`
        Unwrapped phase image
    """   
    phase = 2 * np.arctan(reconstructed_wave.imag / reconstructed_wave.real)

    # No channel, no need for shenanigans
    if phase.ndim < 3:
        return skimage_unwrap_phase(phase, seed=seed)

    # Each wavelength channel must be done separately
    unwrapped = np.empty_like(reconstructed_wave)
    unwrapped_channels = list()
    for phase_channel in np.dsplit(phase, phase.shape[2]):
        unwrapped_channels.append( skimage_unwrap_phase(phase_channel,seed=seed) )
    return np.dstack(unwrapped_channels)

class ReconstructedWave(object):
    """
    Container for reconstructed waves and their intensity and phase
    arrays.
    """
    def __init__(self, reconstructed_wave, fourier_mask):
        """
        Parameters
        ----------
        reconstructed_wave : array_like, complex
            Reconstructed wave in 2- or 3- dimensions.
        fourier_mask : array_like
            Reconstruction Fourier mask, in 2- or 3- dimensions.
        """
        self.reconstructed_wave = np.squeeze(reconstructed_wave)
        self._intensity_image = None
        self._phase_image = None
        self.fourier_mask = np.squeeze(np.asarray(fourier_mask, dtype = np.bool))
        self.random_seed = RANDOM_SEED
    
    @property
    def intensity(self):
        """
        `~numpy.ndarray` of the reconstructed intensity
        """
        if self._intensity_image is None:
            self._intensity_image = np.abs(self.reconstructed_wave)
        return self._intensity_image

    @property
    def phase(self):
        """
        `~numpy.ndarray` of the reconstructed, unwrapped phase.

        Returns the unwrapped phase using `~skimage.restoration.unwrap_phase`.
        """
        if self._phase_image is None:
            self._phase_image = unwrap_phase(self.reconstructed_wave)

        return self._phase_image