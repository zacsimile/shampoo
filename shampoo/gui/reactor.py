
from multiprocessing import Process
from multiprocessing import Queue as ProcessSafeQueue
import numpy as np
from threading import Thread
from time import sleep
from ..reconstruction import Hologram, ReconstructedWave

try:
    from queue import Queue as ThreadSafeQueue  # Python 3
except ImportError:
    from Queue import Queue as ThreadSafeQueue  # Python 2

def _trivial_function(item):
    return item

def _trivial_callback(item, *args, **kwargs):
    pass

class DummyQueue(object):
    """ Dummy queue that does not respond to the put() method """
    def __init__(self, *args, **kwargs):
        pass

    def put(self, *args, **kwargs):
        pass

class Reactor(object):
    """
    Reactor template class. A reactor is an object that reacts accordingly when
    an input is sent to it. To subclass, must at least override reaction(self, item). Can also be initialized with
    a function argument.

    Methods
    -------
    send_item
        Add an item to reactor, adding it to the input queue.
    is_alive
        Check whether the reactor is still running.
    reaction
        Method that is called on every item in the input queue.

    Example
    -------
    Via constructor: for simple, one-argument functions like print
    >>> from queue import Queue     # Python 3
    >>> 
    >>> def func(item):
    ...    print(item)
    ...    return item
    ...
    >>> messages = Queue()
    >>> test = Reactor(out_queue = messages, function = func)
    >>> test.send_item('foobar')
    >>> messages.get() # 'foobar'

    Subclassing: for more complicated, dynamic argument functions
    Print incoming items with the creating time of the reactor
    >>> from queue import Queue     # Python 3
    >>> from datetime.datetime import now
    >>>
    >>> class PrintReactor(Reactor):
    >>>     def __init__(self, out_queue, **kwargs):
    >>>         super(PrintReactor, self).__init__(**kwargs)
    >>>         self.creation_time = now()
    >>> 
    >>>     def reaction(self, item):
    >>>         print(item, self.creation_time)
    >>> 
    >>> messages = Queue()
    >>> test = PrintReactor(out_queue = messages)
    >>> test.send_item('foobar')
    """
    def __init__(self, in_queue = None, out_queue = None, function = None, callback = None, **kwargs):
        """
        Parameters
        ----------
        in_queue: Queue instance or None, optional
            Thread-safe Queue object. If None (default), a local Queue is created. In this case, items can be sent to
            the reactir using the send_item() method.
        out_queue : Queue instance or None, optional
            Thread-safe Queue object, or process-safe Queue object. If None (default), processed items are not passed
            to this output queue.
        function : callable or None, optional
            Function of one argument which will be applied to all items in in_queue. If None (default), 
            a trivial function that returns the input is used.
        callback : callable or None, optional
            Called on each item, after being stored in the output queue. Ideal for emitting Qt signals or printing.
        
        Raises
        ------
        ValueError
            If callback and out_queue are both None.
        """
        if not any((out_queue, callback)):      # out_queue and callback are None
            raise ValueError('out_queue and callback cannot be both None.')

        self.input_queue = in_queue if in_queue is not None else ThreadSafeQueue()
        self.output_queue = out_queue if out_queue is not None else DummyQueue()
        self.function = function if function is not None else _trivial_function
        self.callback = callback if callback is not None else _trivial_callback
        self.worker = None
    
    def start(self):
        # Start of the reactor is in a separate method to allow control by subclasses.
        self.worker = Thread(target = self._event_loop, daemon = True)
        self.worker.start()
    
    def is_alive(self):
        return self.worker.is_alive()
    
    def send_item(self, item):
        self.input_queue.put(item)
    
    def reaction(self, item):
        return self.function(item)
    
    def _event_loop(self):
        while True:
            item = self.input_queue.get()   # Reactor waits indefinitely here
            reacted = self.reaction(item)
            self.callback(reacted)
            self.output_queue.put(reacted)