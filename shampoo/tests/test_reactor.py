from __future__ import (absolute_import, division, print_function, unicode_literals)
import numpy as np
from os.path import join, dirname
from time import sleep

from ..reconstruction import Hologram
from .test_hologram import _example_hologram
from ..gui.reactor import Reconstructor, ProcessSafeQueue

def test_reconstructor_startup():
    """ Tests whether the event loop is running right after start(). """
    reactor = Reconstructor(callback = lambda x: None)
    reactor.start()
    assert reactor.is_alive()

def test_reactor_start_and_stop():
    """ Tests whether the reactor can start, stop, and start again """
    reactor = Reconstructor(callback = lambda x: None)
    reactor.start()
    assert reactor.is_alive()
    reactor.stop()
    assert not reactor.is_alive()
    reactor.start()
    assert reactor.is_alive()