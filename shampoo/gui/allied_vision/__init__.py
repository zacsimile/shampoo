# -*- coding: utf-8 -*-
"""
Wrapper to the Vimba C API.

The code in this package is a modification of the Pymba package, 
available at https://github.com/morefigs/pymba.

Any connection to an Allied Vision camera must go through a Vimba object.
"""
from __future__ import absolute_import

from .vimba import Vimba

STRING_ENCODING = 'utf-8'