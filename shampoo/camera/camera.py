
"""
Camera object that abstracts away the underlying API.

Functions
---------
available_cameras
    Prints the available, connected cameras as well as a list of their features.
"""
from .allied_vision import Vimba
from ..gui.reactor import Reactor, ThreadSafeQueue
import numpy as np
from threading import Thread
import time

def available_cameras():
    """
    Prints the make/model of available cameras, as well as available features.
    """
    with Vimba() as vimba:
            
        cameraIds = vimba.getCameraIds()

        if not cameraIds:
            print('No cameras are available.')
            return
        
        for cameraId in cameraIds:
            print('Allied Vision Camera ID: ', cameraId)
            camera = vimba.getCamera(cameraId)
            camera.openCamera()
            for name in camera.getFeatureNames():
                print('    Feature: ', name)
            camera.closeCamera()

class Camera(object):
    """ Template object for cameras that can interact with shampoo.gui """

    @property
    def resolution(self):
        """ Shape of the image data (height, width) """
        raise NotImplementedError

    def start_acquisition(self, image_queue = None, callback = None):
        """
        Parameters
        ----------
        image_queue : Queue instance or None, optional
            Thread-safe queue in which frame data is deposited as NumPy Arrays. 
            If None (default), a callback is used on every frame.
        
        Raises
        ------
        ValueError
            If image_queue and callback are both None.
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
        super(Camera, self).__del__()


class AlliedVisionCamera(Camera):
    """
    Camera object from manufacturer Allied Vision.

    In order to discover available cameras, consider using available_cameras()
    """

    def __init__(self):
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
        # TODO: select from list of cameras
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
    
    def start_acquisition(self, image_queue = None):

        self._frame.announceFrame()
        self._camera.startCapture()

        #TODO: run in a separate process?
        self._live_acquisition_thread = Thread(target = self._live_acquisition, args = (image_queue,))
        self._live_acquisition_thread.start()
    
    def snapshot(self):
        """
        Returns an image from the camera as soon as possible.

        Returns
        -------
        img : ndarray
        """
        self._frame.queueFrameCapture()
        self._frame.queueFrameCapture()
        self._camera.runFeatureCommand('AcquisitionStart')
        self._camera.runFeatureCommand('AcquisitionStop')
        self._frame.waitFrameCapture(1000)
        return self._frame.getImage()
    
    def live_view(self):
        """ 
        Presents a live view from the camera. For testing purposes only.
        """
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.show()

        image_queue = ThreadSafeQueue()
        display_reactor = Reactor(input_queue = image_queue, callback = plt.imshow)
        display_reactor.start()
        self.start_acquisition(image_queue = image_queue)

    def _live_acquisition(self, image_queue):
        """ 
        Live acquisition of the camera in a separate thread. 
        
        image_queue : Queue instance
            Thread-safe Queue object
        callback : callable, optional
        """
        while self._keep_acquiring:
            img = self.snapshot()
            self.image_queue.put(img)

        # Prepare for next time self.start_acquisition() is called
        self._keep_acquiring = True
    
    def stop_acquisition(self):
        # Stop the thread
        self._keep_acquiring = False
        self._live_acquisition_thread.join()

        self._camera.endCapture()
        self._camera.revokeAllFrames()
    
    def disconnect(self):
        try:
            self.stop_acquisition()
        except:
            pass

        self._camera.closeCamera()
        self._api.shutdown()