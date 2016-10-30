
"""
Camera object that abstracts away the underlying API.

Functions
---------
available_cameras
    Prints the available, connected cameras as well as a list of their features.
"""
from .allied_vision import Vimba
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
    with Vimba() as vimba:
        
        vimba.getSystem().runFeatureCommand('GeVDiscoveryAllOnce')  # Enable gigabit-ethernet discovery
        cameraIds = vimba.getCameraIds()
        
        return list(map(str, cameraIds)) # In py3+, map() returns an iterable map object

class Camera(object):
    """ Template object for cameras that can interact with shampoo.gui """

    @property
    def resolution(self):
        """ Shape of the image data (height, width) """
        raise NotImplementedError
        
    def snapshot(self):
        """
        Instantaneous snapshot.

        Returns
        -------
        img : ndarray
        """
        raise NotImplementedError

    def start_acquisition(self, image_queue = None):
        """
        Parameters
        ----------
        image_queue : Queue instance or None, optional
            Thread-safe queue in which frame data is deposited as NumPy Arrays. 
            If None (default), a callback is used on every frame.
        """
        raise NotImplementedError
    
    def stop_acquisition(self):
        raise NotImplementedError
       
    def connect(self):
        raise NotImplementedError
    
    def disconnect(self):
        raise NotImplementedError
    
    def __del__(self):
        self.disconnect()

class AlliedVisionCamera(Camera):
    """
    Camera object from manufacturer Allied Vision.

    In order to discover available cameras, consider using available_cameras()
    """

    def __init__(self, ID = None):
        """
        Parameters
        ----------
        ID : str
            Camera identifier. Ignored until implemented.
        """
        # TODO: select from list of cameras
        super(AlliedVisionCamera, self).__init__()
        self._api = Vimba()
        self._camera = None
        self._frame = None
        self._keep_acquiring = True

        self.connect() # Instantiates self._camera, self._frame
    
    @property
    def resolution(self):
        return (self._frame.height, self._frame.width)
    
    def connect(self):
        self._api.startup()
        self._api.getSystem().runFeatureCommand('GeVDiscoveryAllOnce')
        
        # Get camera
        camera_ids = self._api.getCameraIds()
        self._camera = self._api.getCamera(camera_ids[0])
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