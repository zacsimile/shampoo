"""
Debug classes to ease development
"""

from .camera import AlliedVisionCamera
import numpy as np
from time import sleep

class DebugCamera(AlliedVisionCamera):
    """
    Debug version of a camera object
    """
    def __init__(self, *args, **kwargs): 
        self._exposure = 16000
    

    @property
    def resolution(self):
        """ Shape of the image data (height, width) """
        return (1024, 1024)
    
    @property
    def exposure_increment(self):
        return 10
    
    @property
    def exposure(self):
        return self._exposure
    
    @exposure.setter
    def exposure(self, value_us):
        if not (value_us % self.exposure_increment) == 0:
            value_us = value_us + (value_us % self.exposure_increment)
        self._exposure = value_us

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