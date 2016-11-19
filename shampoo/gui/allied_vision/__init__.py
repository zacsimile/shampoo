# -*- coding: utf-8 -*-
"""
Wrapper to the Vimba C API.

The code in this package is a modification of the Pymba package, 
available at https://github.com/morefigs/pymba.

Any connection to an Allied Vision camera must go through a Vimba object.

Example
-------
>>> from shampoo.camera.allied_vision import Vimba
>>> 
>>> with Vimba() as vimba:
>>> 
>>>     cameraIds = vimba.getCameraIds()
>>>     for cameraId in cameraIds:
>>>         print('Camera ID:', cameraId)
>>>     
>>>     camera0 = vimba.getCamera(cameraIds[0])
>>>     camera0.openCamera()
>>>
>>>     cameraFeatureNames = camera0.getFeatureNames()
>>>     for name in cameraFeatureNames:
>>>         print('    Camera feature:', name)
>>>    
>>>     camera0.closeCamera()

"""

from .vimba import Vimba

STRING_ENCODING = 'utf-8'