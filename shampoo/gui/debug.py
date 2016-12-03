"""
Debug classes to ease development
"""
from __future__ import absolute_import
from .camera import AlliedVisionCamera
import numpy as np
from time import sleep

class DebugCamera(AlliedVisionCamera):
    """
    Debug version of a camera object
    """
    def __init__(self, *args, **kwargs): 
        self._exposure = 16000
        self._bit_depth = 8
    

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
        print('Exposure changed to {}'.format(value_us))
        self._exposure = value_us
    
    @property
    def bit_depth(self):
        """ Returns the bit depth: (8, 10, 12, 14) bits"""
        return self._bit_depth
    
    @bit_depth.setter
    def bit_depth(self, depth):
        """ Bit depth : 8, 10, 12, 14 bits. """ 
        print('bit depth changed to {}'.format(depth))
        self._bit_depth = depth

    def snapshot(self):
        """
        Instantaneous snapshot.

        Returns
        -------
        img : ndarray
        """
        arr = 256*np.random.random(size = self.resolution)
        return arr.astype(np.uint8)

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
    
    def disconnect(self):
        self.stop_acquisition()