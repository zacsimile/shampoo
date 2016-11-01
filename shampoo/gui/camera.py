
"""
Camera object that abstracts away the underlying API.

Functions
---------
available_cameras
    Prints the available, connected cameras as well as a list of their features.
"""
from .allied_vision import Vimba
from collections import namedtuple
import numpy as np
from threading import Thread

def available_cameras():
    """
    Prints the make/model of available cameras, as well as available features.

    Returns
    -------
    ids : list of strings
        Camera IDs
    """
    # Allied Vision cameras
    with Vimba() as vimba:
        
        vimba.getSystem().runFeatureCommand('GeVDiscoveryAllOnce')  # Enable gigabit-ethernet discovery
        cameraIds = vimba.getCameraIds()
        
        return list(map(str, cameraIds)) # In py3+, map() returns an iterable map object

class Feature(object):
    """
    Simple object containing camera feature name, value, and access mode.
    """
    def __init__(self, name, access_mode, value = None):
        self.name = name
        self.access_mode = access_mode
        self.value = value

class AlliedVisionCamera(object):
    """
    Camera object from manufacturer Allied Vision.

    In order to discover available cameras, consider using available_cameras()

    Attributes
    --------
    exposure : float
        Integration time in microsecond.
    exposure_increment : float
        Minimum integration time increment in microsecond.
    resolution : (int, int)
        Sensor resolution.
    """
    features = ('exposure', 'exposure_increment', 'resolution')
    features_access_modes = {'exposure':'RW', 'exposure_increment':'R', 'resolution':'R'}

    def __init__(self, ID = None):
        """
        Parameters
        ----------
        ID : str
            Camera identifier.
        """
        super(AlliedVisionCamera, self).__init__()

        self._api = Vimba()
        self._camera = None
        self._frame = None
        self._keep_acquiring = True
        self.ID = ID

        # Startup
        self._api.startup()
        self._api.getSystem().runFeatureCommand('GeVDiscoveryAllOnce')
        self._camera = self._api.getCamera(ID)
        self.connect() # Open camera, instantiate self._frame
    
    def __repr__(self):
        return '< Allied Vision Camera, ID = {}>'.format(self.ID)
    
    def __del__(self):
        self.disconnect()
    
    # Camera features
    
    @property
    def exposure(self):
        """ Sensor integration time """
        return self._camera.ExposureTimeAbs
    
    @exposure.setter
    def exposure(self, value_us):
        """ Integration time in microseconds """
        # Make sure the exposure is allowable
        if not (value_us % self.exposure_increment) == 0:
            value_us = value_us + (value_us % self.exposure_increment)

        self._camera.ExposureTimeAbs = value_us
    
    @property
    def exposure_increment(self):
        return self._camera.ExposureTimeIncrement
    
    @property
    def resolution(self):
        return (self._frame.height, self._frame.width)
    
    def connect(self):

        self._camera.openCamera()
        
        # Set some feature values
        self._camera.PixelFormat = 'Mono8'
        try:
            self._camera.StreamBytesPerSecond = 100000000 # Only valid for Gigabit-Ethernet
        except:
            pass

        self._frame = self._camera.getFrame()
        self._frame.announceFrame()        
        self._camera.startCapture()
       
    def snapshot(self):
        """
        Instantaneous snapshot.

        Returns
        -------
        img : ndarray
        """
        self._frame.queueFrameCapture()
        self._camera.runFeatureCommand('AcquisitionStart')
        self._camera.runFeatureCommand('AcquisitionStop')
        self._frame.waitFrameCapture(1000)
        frame_data = self._frame.getFrameBufferData()
        return ndarray(buffer = frame_data, dtype = n.uint8, shape = self.resolution)
        
    def start_acquisition(self, image_queue):

        #TODO: run in a separate process?
        self._live_acquisition_thread = Thread(target = self._live_acquisition, args = (image_queue,))
        self._live_acquisition_thread.start()

    def _live_acquisition(self, image_queue):
        """ 
        Live acquisition of the camera in a separate thread. 
        
        image_queue : Queue instance
            Thread-safe Queue object
        callback : callable, optional
        """
        while self._keep_acquiring:
            image_queue.put(self.snapshot())

        # Prepare for next time self.start_acquisition() is called
        self._keep_acquiring = True
        
    def stop_acquisition(self):
        # Stop the thread
        self._keep_acquiring = False
        self._live_acquisition_thread.join()
    
    def disconnect(self):
        
        try:
            self.stop_acquisition()
        except:
            pass
        
        self._camera.endCapture()
        self._camera.revokeAllFrames()
        self._camera.closeCamera()
        self._api.shutdown()