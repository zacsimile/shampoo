"""
Debug classes to ease development
"""

from .camera import Camera
import numpy as np
from time import sleep

class DebugCamera(Camera):
    """
    Debug version of a camera object
    """

    @property
    def features(self):
        """
        Returns a list of strings representing features that can be changed by the user.
        """
        return ['Resolution', 'Framerate', 'Manufacturer', 'Model']
    
    @property
    def resolution(self):
        """ Shape of the image data (height, width) """
        return (1024, 1024)
        
    def snapshot(self):
        """
        Instantaneous snapshot.

        Returns
        -------
        img : ndarray
        """
        return np.random.random(size = self.resolution)

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
            sleep(0.1)

        # Prepare for next time self.start_acquisition() is called
        self._keep_acquiring = True
        
    def stop_acquisition(self):
        # Stop the thread
        self._keep_acquiring = False
        self._live_acquisition_thread.join()
       
    def connect(self): pass
    
    def disconnect(self): pass