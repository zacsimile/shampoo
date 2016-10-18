
"""
Camera object that abstracts away the underlying API
"""
from .allied_vision.vimba import Vimba
import numpy as np
from threading import Thread

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
    
    def start_acquisition(self, image_queue = None, callback = None):
        if not any( (image_queue, callback) ):
            raise ValueError('image_queue and callback cannot be both None')

        #TODO: deal with possibly image_queue = None or callback = None
        
        self._frame.announceFrame()
        self._camera.startCapture()

        #TODO: run in a separate process?
        self._live_acquisition_thread = Thread(target = self._live_acquisition, args = (image_queue, callback))
        self._live_acquisition_thread.start()

    def _live_acquisition(self, image_queue, callback):
        """ Live acquisition of the camera in a separate thread. """
        while self._keep_acquiring:
            self._frame.queueFrameCapture()
            img = np.ndarray(buffer = self._frame.getBufferByteData(), dtype = n.uint8, shape = self.resolution)
            self.image_queue.put(img)
            callback(img)

        # Prepare for next time self.start_acquisition() is called
        self._keep_acquiring = True
    
    def stop_acquisition(self):
        self._camera.endCapture()
        self._camera.revokeAllFrames()

    
    def disconnect(self):
        try:
            self.stop_acquisition()
        except:
            pass

        self._camera.closeCamera()
        self._api.shutdown()