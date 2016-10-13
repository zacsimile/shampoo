
from multiprocessing import Process, Queue
import numpy as np
from threading import Thread
from time import sleep
from ..reconstruction import Hologram, ReconstructedWave

class ShampooController(object):
    """
    Underlying controller to SHAMPOO's Graphical User Interface
    """
    def __init__(self, out_queue):
        """
        Parameters
        ----------
        output_function: callable
        """
        self.input_queue = Queue()
        self.output_queue = out_queue

        self.reconstruction_reactor = ReconstructionReactor(in_queue = self.input_queue, out_queue = self.output_queue)

    def send_data(self, data):
        """ 
        Send holographic data to the reconstruction reactor.

        Parameters
        ----------
        data : ndarray or Hologram object
            Can be any type that can is accepted by the Hologram() constructor.
        """
        if not isinstance(data, Hologram):
            data = Hologram(data)
        self.input_queue.put(data)
    
    # Proxy property to propagation distance from reconstruction reactor
    @property
    def propagation_distance(self):
        return self.reconstruction_reactor.propagation_distance
    
    @propagation_distance.setter
    def propagation_distance(self, prop_distance):
        self.reconstruction_reactor.propagation_distance = prop_distance

### REACTORS ###

class Reactor(object):
    """
    Reactor template class. A reactor is an object that waits for input and reacts accordingly when
    an input is sent to it. To subclass, must at least override reaction(self, item). Can also be initialized with
    a function argument.

    Methods
    -------
    send_item
        Add an item to the queue.
    is_alive
        Check whether the reactor is still running.
    reaction
        Method that is called on every item in the input queue.

    Example
    -------
    Via constructor: for simple, one-argument functions like print
    >>> from queue import Queue
    >>> 
    >>> print_queue = Queue()
    >>> test = Reactor(in_queue = print_queue, function = print)
    >>> print_queue.put('foobar')

    Subclassing: for more complicated, dynamic argument functions
    Print incoming items with the creating time of the reactor
    >>> from queue import Queue
    >>> from datetime.datetime import now
    >>>
    >>> class PrintReactor(Reactor):
    >>>     def __init__(self, in_queue, **kwargs):
    >>>         super(PrintReactor, self).__init__(in_queue = in_queue, function = None, **kwargs)
    >>>         self.creation_time = now()
    >>> 
    >>>     def reaction(self, item):
    >>>         print(item, self.creation_time)
    >>> 
    >>> print_queue = Queue()
    >>> test = PrintReactor(in_queue = print_queue)
    >>> print_queue.put('foobar')
    """
    def __init__(self, in_queue, function = None, **kwargs):
        """
        Parameters
        ----------
        in_queue : Queue instance
            Thread-safe Queue object. Can be from the queue (py3) or Queue (py2) module, or multiprocessing.
        function : callable or None, optional
            Function of one argument. Will be applied to all items in in_queue. If None, Reactor must be subclassed.
        """
        super(Reactor, self).__init__(**kwargs)
        self.input_queue = in_queue
        self._function = function

        self.worker = Thread(target = self._event_loop)
        self.worker.daemon = True
        self.worker.start()
    
    def is_alive(self):
        return self.worker.is_alive()
    
    def send_item(self, item):
        """ Send item to the input queue of the reactor. """
        self.input_queue.put(item)
    
    def reaction(self, item):
        if self._function is not None:
            return self._function(item)
        raise NotImplementedError('Either override the Reactor.reaction method, or provide a function in the constructor.')
    
    def _event_loop(self):
        while True:
            item = self.input_queue.get()   # Reactor waits indefinitely here
            self.reaction(item)

def _reconstruct_hologram(hologram, propagation_distance, output_queue):
    """
    Function wrapper to Hologram.reconstruct

    Parameters
    ----------
    hologram: Hologram instance

    propagation_distance : ndarray, shape (N,) 

    output_queue : multiprocessing.Queue instance

    """
    print(propagation_distance)
    if len(propagation_distance) == 1:
        # Keep the propagation distance in the queue, for correct plotting
        output_queue.put( (propagation_distance, hologram.reconstruct(propagation_distance = propagation_distance[0])) )
    else:
        output_queue.put( (propagation_distance, hologram.reconstruct_multithread(propagation_distances = propagation_distance)) )

class ReconstructionReactor(Reactor):
    """    
    """
    def __init__(self, in_queue, out_queue, **kwargs):
        """
        Parameters
        ----------
        in_queue, out_queue : Queue instances
            Thread-safe and Process-safe Queue objects.
        """
        super(ReconstructionReactor, self).__init__(in_queue = in_queue, function = None, **kwargs)
        self.output_queue = out_queue
        self._propagation_distance = np.array([0.03685])

    # Propagation distance property ensures that propagation distance is always an array
    @property
    def propagation_distance(self):
        return self._propagation_distance
    
    @propagation_distance.setter
    def propagation_distance(self, value):
        self._propagation_distance = np.array(value).tolist()
    
    def reaction(self, hologram):
        self.sub_worker = Process(target = _reconstruct_hologram, args = (hologram, self.propagation_distance, self.output_queue))
        self.sub_worker.start()