
from multiprocessing import Process, Pipe
from multiprocessing import Queue as ProcessSafeQueue
import numpy as np
from threading import Thread
from time import sleep
from ..reconstruction import Hologram, ReconstructedWave

try:
    from queue import Queue as ThreadSafeQueue  # Python 3
    from queue import Empty
except ImportError:
    from Queue import Queue as ThreadSafeQueue  # Python 2
    from Queue import Empty

def _trivial_function(item):
    return item

def _trivial_callback(item, *args, **kwargs):
    pass

class VoidQueue(object):
    """ Dummy queue that does not respond to the put() method """
    def __init__(self, *args, **kwargs):
        pass

    def put(self, *args, **kwargs):
        pass

class Reactor(object):
    """
    Reactor template class. A reactor is an object that reacts accordingly when
    an input is sent to it. To subclass, must at least override reaction(self, item).

    Methods
    -------
    start
        Start the reactor. Input can be queued before this method is called.
    send_item
        Add an item to reactor, adding it to the input queue.
    is_alive
        Check whether the reactor is still running.
    reaction
        Method that is called on every item in the input queue. User-facing for subclassing purposes.

    Example
    -------
    Simple callback:
    >>> from __future__ import print_function
    >>> test = Reactor(callback = print)
    >>> test.send_item('foobar')

    Chaining reactors:
    >>> from queue import Queue   # Python 3
    >>>
    >>> messages = Queue()
    >>> results = Queue()
    >>>
    >>> def some_func_1(item): return item
    >>> def some_func_2(item): return item
    >>>
    >>> reactor1 = Reactor(output_queue = messages, function = some_func_1, callback = print)
    >>> reactor2 = Reactor(input_queue = messages, output_queue = results, function = some_func_2, callback = print)
    >>> reactor1.start(), reactor2.start()
    >>> reactor1.send_item('foobar')
    """
    def __init__(self, input_queue = None, output_queue = None, function = None, callback = None, **kwargs):
        """
        Parameters
        ----------
        input_queue: Queue instance or None, optional
            Thread-safe Queue object. If None (default), a local Queue is created. In this case, items can be sent to
            the reactor using the send_item() method.
        output_queue : Queue instance or None, optional
            Thread-safe Queue object, or process-safe Queue object. If None (default), processed items are not stored in any queue.
        function : callable or None, optional
            Function of one argument which will be applied to all items in input_queue. If None (default), 
            a trivial function that returns the input is used.
        callback : callable or None, optional
            Called on each item before being stored in the output queue. Ideal for emitting Qt signals or printing.
        
        Raises
        ------
        ValueError
            If callback and output_queue are both None.
        """
        if not any((output_queue, callback)):      # output_queue and callback are None
            raise ValueError('output_queue and callback cannot be both None.')

        self.input_queue = input_queue if input_queue is not None else ThreadSafeQueue()
        self.output_queue = output_queue if output_queue is not None else VoidQueue()
        self.function = function if function is not None else _trivial_function
        self.callback = callback if callback is not None else _trivial_callback
        self.worker = None

        self._keep_running = True
    
    def start(self):
        """ Start the event loop in a separate thread. """
        # Start of the reactor is in a separate method to allow control by subclasses.
        self.worker = Thread(target = self._event_loop, daemon = False)
        self.worker.start()
    
    def stop(self):
        """ 
        Stop the event loop. This method will block until the event loop
        thread is joined. 
        """
        self._keep_running = False
        self.worker.join()

    def is_alive(self):
        """ Returns True if the event loop is running. Otherwise, returns False. """
        try:
            return self.worker.is_alive()
        except: # Worker might not have been created yet
            return False
    
    def send_item(self, item):
        """ Adds an item to the input queue. """
        self.input_queue.put(item)
    
    def reaction(self, item):
        """ Method applied to every item in the input queue. Provided as an easy mean to subclass. """
        return self.function(item)
    
    def _event_loop(self):
        while self._keep_running:
            # Don't wait indefinitely here, in case the stop() method has been called
            try: 
                item = self.input_queue.get(timeout = 1)   # Reactor waits at most 1 second here
            except Empty: 
                pass
            else:
                reacted = self.function(item)
                self.callback(reacted)
                self.output_queue.put(reacted)
        
        # Prepare for next time start() is called
        self._keep_running = True

class ProcessReactor(Reactor):
    """
    Reactor running an event loop in a separate process.
    """

    def __init__(self, input_queue = None, output_queue = None, function = None, callback = None, **kwargs):
        """
        Parameters
        ----------
        input_queue: Queue instance or None, optional
            Process-safe Queue object. If None (default), a local Queue is created. In this case, items can be sent to
            the reactor using the send_item() method.
        output_queue : Queue instance or None, optional
            Process-safe Queue object, or process-safe Queue object. If None (default), processed items are not stored in any queue.
        function : callable or None, optional
            Function of one argument which will be applied to all items in input_queue. If None (default), 
            a trivial function that returns the input is used.
        callback : callable or None, optional
            Called on each item before being stored in the output queue. Ideal for emitting Qt signals or printing.
        
        Raises
        ------
        ValueError
            If callback and output_queue are both None.
        """
        if not any((output_queue, callback)):      # output_queue and callback are None
            raise ValueError('output_queue and callback cannot be both None.')
        
        self.input_queue = input_queue if input_queue is not None else ProcessSafeQueue()
        self.output_queue = output_queue if output_queue is not None else VoidQueue()
        self.function = function if function is not None else _trivial_function
        self.callback = callback if callback is not None else _trivial_callback
        self.worker = None
    
    def start(self):
        """ Start the event loop in a separate thread. """
        # Start of the reactor is in a separate method to allow control by subclasses.
        self.worker = Process(target = _func_event_loop, args = (self.input_queue, self.output_queue, self.reaction, self.callback))
        self.worker.start()
    
    def stop(self):
        """ 
        Stop the event loop. This method will block until the event loop
        thread is joined. 
        """
        self.worker.terminate()
        self.worker.join()

def _func_event_loop(input_queue, output_queue, function, callback):
    """ Functional form of the reactor event loop. """
    while True:
        try:
            item = input_queue.get(timeout = 1)
        except Empty:
            pass
        else:
            reacted = function(item)
            callback(item)
            output_queue.put(reacted)