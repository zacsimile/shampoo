"""
Underlying Objects abstracting concurrent operations in SHAMPOO's GUI.

Classes
-------
Reactor

ProcessReactor
"""
from __future__ import absolute_import

from multiprocessing import Process, Pipe
from multiprocessing import Queue as ProcessSafeQueue
import numpy as np
from threading import Thread
from time import sleep

try:
    from queue import Queue as ThreadSafeQueue  # Python 3
    from queue import Empty
except ImportError:
    from Queue import Queue as ThreadSafeQueue  # Python 2
    from Queue import Empty

class Reconstructor(object):
    """
    Object in charge of reconstructing holograms as they come in.

    Methods
    -------
    start
        Start the reactor. Input can be queued before this method is called.
    stop
        Stop the reactor. Can be restarted using start().
    send_item
        Add an item to reactor, adding it to the input queue.
    is_alive
        Check whether the reactor is still running.
    """

    # This object requires two event loops because processes cannot do callbacks
    # Therefore, there is one loop for reconstruction, and one loop for emission
    # of reconstructed holograms

    def __init__(self, callback):
        """
        Parameters
        ----------
        parent : QtCore.QObject

        output_signal : QtCore.pyqtSignal
            Signal emitted with the result of a reconstruction 
        """

        self.input_queue = ProcessSafeQueue()
        self.output_queue = ProcessSafeQueue()
        self.callback = callback
        self.worker = None 
        self.emitter = None
    
    def start(self):
        """ Start the event loop in a separate process. """
        self._keep_running = True
        self.worker = Process(target = _reconstruction_loop, args = (self.input_queue, self.output_queue))
        self.emitter = Thread(target = self._emission_loop)
        
        self.worker.start()
        self.emitter.start()
    
    def stop(self):
        """ Stop the event loop. This method will block until the event loop process is joined. """
        self._keep_running = False
        self.worker.terminate()
        self.worker.join()
        self.emitter.join()

    def is_alive(self):
        """ Returns True if the event loop is running. Otherwise, returns False. """
        try:
            return self.worker.is_alive() and self.emitter.is_alive()
        except: # Worker might not have been created yet
            return False
    
    def send_item(self, item):
        """ Adds an item to the input queue. """
        self.input_queue.put(item)
    
    def _emission_loop(self):
        while self._keep_running:
            try:
                # Don't wait to long here, as 
                # _keep_running might have changed
                item = self.output_queue.get(timeout = 1)
            except:
                pass
            else:
                self.callback(item)
    
def _reconstruction_loop(input_queue, output_queue):
    while True:
        try: 
            item = input_queue.get(timeout = 1)
        except Empty: 
            pass
        else:
            propagation_distance, hologram = item
            if len(propagation_distance) == 1:
                output_queue.put( (propagation_distance, 
                                   hologram.reconstruct(propagation_distance = propagation_distance[0])) )
            else:
                output_queue.put( (propagation_distance, 
                                   hologram.reconstruct_multithread(propagation_distances = propagation_distance)) )