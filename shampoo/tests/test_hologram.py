from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from ..reconstruction import (Hologram, rebin_image, _find_peak_centroid,
                              RANDOM_SEED, _crop_image, CropEfficiencyWarning)

import numpy as np
np.random.seed(RANDOM_SEED)


def _example_hologram(dim=256):
    """
    Generate example hologram.

    Parameters
    ----------
    dim : int
        Dimensions of image. Default is 2048.
    """
    return 1000*np.ones((dim, dim)) + np.random.randn(dim, dim)


def test_load_hologram():
    holo = Hologram(_example_hologram())
    assert holo is not None


def test_rebin_image():
    dim = 2048
    full_res = _example_hologram(dim=dim)
    assert (dim//2, dim//2) == rebin_image(full_res, 2).shape

def test_nondefault_fourier_mask():
    im = _example_hologram()
    holo = Hologram(im)
    mask = np.random.randint(0, 2, size = im.shape).astype(np.bool)
    w = holo.reconstruct(0.5, fourier_mask = mask)

    assert np.allclose(np.squeeze(w.fourier_mask), np.squeeze(mask))

def test_reconstruction_multiwavelength():
    im = np.dstack([_example_hologram() for _ in range(3)])
    holo = Hologram(im, wavelength = [450e-9, 550e-9, 650e-9])

    w = holo.reconstruct(0.2)
    assert im.shape == w.reconstructed_wave.shape

def _gaussian2d(amplitude, width, centroid, dim):
    x, y = np.mgrid[0:dim, 0:dim]
    x_centroid, y_centroid = centroid
    return amplitude*np.exp(-0.5 * ((x - x_centroid)**2/width**2 +
                                    (y - y_centroid)**2/width**2))


def test_centroid():
    centroid = (265, 435)
    test_image = _gaussian2d(amplitude=10, width=5, centroid=centroid, dim=1024)
    assert np.all(np.squeeze(_find_peak_centroid(image=test_image)) == centroid)
    assert np.all(test_image[centroid] == np.max(test_image))

def test_centroid_multichannel():
    """ Test _find_peak_centroid for inputs with multiple channels (wavelengths) """
    centroids = (102, 304), (405, 312)
    test_image = np.dstack([_gaussian2d(amplitude=20, width=4, centroid=centroids[0], dim=512),
                            _gaussian2d(amplitude=10, width=7, centroid=centroids[1], dim=512)])
    computed_centroids = np.squeeze(_find_peak_centroid(image=test_image))
    assert np.all(computed_centroids[:,0] == centroids[0])
    assert np.all(computed_centroids[:,1] == centroids[1])

def test_crop_image():
    # Even number rows/cols
    image1 = np.arange(1024).reshape((32, 32))
    new_shape1 = (image1.shape[0]//2, image1.shape[1]//2)
    cropped_image1 = _crop_image(image1, 0.5)
    assert new_shape1 == cropped_image1.shape

    # Odd number rows/cols
    image2 = np.arange(121).reshape((11, 11))
    new_shape2 = (image2.shape[0]//2, image2.shape[1]//2)
    cropped_image2 = _crop_image(image2, 0.5)
    assert new_shape2 == cropped_image2.shape


def test_multiple_reconstructions():
    """
    At commit cc730bd and earlier, the Hologram.apodize function modified
    the Hologram.hologram array every time Hologram.reconstruct was called.
    This tests that that should not happen anymore.
    """

    propagation_distances = [0.5, 0.8]
    holo = Hologram(_example_hologram())
    h_raw = holo.hologram.copy()
    holograms = []

    for d in propagation_distances:
        w = holo.reconstruct(d)
        holograms.append(holo.hologram)

    # check hologram doesn't get modified in place first time
    assert np.all(h_raw == holograms[0])

    # check hologram doesn't get modified again
    assert np.all(holograms[0] == holograms[1])


def test_nonsquare_hologram():
    sq_holo = _example_hologram()
    nonsq_holo = sq_holo[:-10, :]

    holo = Hologram(nonsq_holo)
    w = holo.reconstruct(0.5)

    phase_shape = w.phase.shape

    assert phase_shape[0] == min(nonsq_holo.shape)
    assert phase_shape[1] == min(nonsq_holo.shape)
