
"""
Camera object that abstracts away the underlying API
"""
from .allied_vision.vimba import Vimba
class Camera(object):
    """ Template object for cameras that can interact with shampoo.gui """

    @property
    def resolution(self):
        """ Shape of the image data (height, width) """
        raise NotImplementedError

    def start_acquisition(self, image_queue):
        """
        Parameters
        ----------
        image_queue : Queue instance
            Thread-safe queue in which frame data is deposited as NumPy Arrays
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
    
    def start_acquisition(self, image_queue):
        self._frame.announceFrame()
        self._camera.startCapture()
    
    def _live_acquisition(self, image_queue):
        """ """
    
    def stop_acquisition(self):
        self._camera.endCapture()
        self._camera.revokeAllFrames()
        self._camera.closeCamera()

    
    def disconnect(self):
        try:
            self.stop_acquisition()
        except:
            pass

        self._api.shutdown()