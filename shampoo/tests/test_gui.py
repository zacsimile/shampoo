from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
from os.path import join, dirname
from time import sleep

from ..reconstruction import Hologram
from .test_hologram import _example_hologram
from ..gui.reactor import Reconstructor, ProcessSafeQueue
from ..gui.gui import ShampooController

def test_reconstructor_startup():
    """ Tests whether the event loop is running right after start(). """
    reactor = Reconstructor(callback = lambda x: None)
    reactor.start()
    assert reactor.is_alive()
    reactor.stop()

def test_reactor_start_and_stop():
    """ Tests whether the reactor can start, stop, and start again """
    reactor = Reconstructor(callback = lambda x: None)
    reactor.start()
    assert reactor.is_alive()
    reactor.stop()
    assert not reactor.is_alive()
    reactor.start()
    assert reactor.is_alive()
    reactor.stop()

def test_reconstructor_handling_of_fourier_mask():
    reconstructed = list()
    reactor = Reconstructor(callback = reconstructed.append)
    reactor.start()

    # Reconstruct with a specific mask
    hologram = _example_hologram()
    mask = np.zeros_like(hologram)
    reactor.send_item( ([0], hologram, mask) )

    # reconstruct without a mask
    reactor.send_item( ([0], hologram, None ))

def test_controller():
    """ Test the heart of SHAMPOO's GUI """
    controller = ShampooController()
    controller.stop()