
from multiprocessing import Process, Queue
from threading import Thread, Lock
from .reconstruction import Hologram, ReconstructedWave

class ShampooController(object):
    """
    Underlying controller to SHAMPOO's Graphical User Interface

    Attributes
    ----------
    propagation_distance : float
        Propagation distance at which to reconstruct holograms. Will be extended
        to arrays soon.
    input_queue : multiprocessing.Queue
        Input holograms queue.
    output_queue : multiprocessing.Queue
        Output ReconstructedWave queue.
    reactor : ReconstructionReactor instance
        Reconstruction reactor
    
    Methods
    -------
    start_real_time_reconstruction

    stop_real_time_reconstruction

    send_data

    """
    DEFAULT_PROPAGATION_DISTANCE = 0.03658

    def __init__(self, output_function):
        """
        Parameters
        ----------
        output_function: callable
        """
        self.input_queue = Queue()
        self.output_queue = Queue()

        self.reconstruction_reactor = ReconstructionReactor(in_queue = self.input_queue, out_queue = self.output_queue, prop_distance = self.DEFAULT_PROPAGATION_DISTANCE)
        self.output_reactor = OutputReactor(in_queue = self.output_queue, output_function = output_function)

        self.start_real_time_reconstruction()

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
        # TODO: add type checks?
        self.reconstruction_reactor.propagation_distance = prop_distance
    
    # Real-time control
    def start_real_time_reconstruction(self):
        """ Start the real-time reconstruction of holograms. """
        self.reconstruction_reactor.start()
        self.output_reactor.start()

    def stop_real_time_reconstruction(self):
        """ Halts the real-time reconstruction of holograms. """
        self.reconstruction_reactor.stop()
        self.output_reactor.stop()
    
    def __del__(self):
        # Is this necessary?
        self.stop_real_time_reconstruction()
        super(ReconstructionController, self).__del__()

### REACTORS ###

class OutputReactor(object):
    """ Reactor that reacts to the output. For simplicity reasons, this is done in a Thread """
    def __init__(self, in_queue, output_function):

        self.input_queue = in_queue
        self.function = output_function
        self.worker = None

        self._isrunning = True

    def start(self):        
        self.worker = Thread(target = self._main_loop)
        self.worker.start()
    
    def stop(self):
        self._isrunning = False

    def _main_loop(self):
        while True:
            if not self._isrunning:
                break
            
            item = self.input_queue.get()
            self.function(item)
        
        # Prepare next loop start()
        self._isrunning = True

def _reactor_main_loop(reactor):
    return reactor._main_loop()

class ReconstructionReactor(object):
    """
    Reactor that reconstructs holograms as they come in.

    Attributes
    ----------
    propagation_distance : float
        Propagation distance at which to reconstruct holograms. Will be extended
        to arrays soon.
    input_queue : multiprocessing.Queue
        Input holograms queue.
    output_queue : multiprocessing.Queue
        Output ReconstructedWave queue.
    worker : multiprocessing.Process instance or None

    Methods
    -------
    start
        Start reconstructing inputs as they come in.
    stop
        Stop the main loop. Can be restarted at any time.
    """
    def __init__(self, in_queue, out_queue, prop_distance = 0.03685):
        """
        Parameters
        ----------
        input_queue : multiprocessing.Queue
            Input holograms queue.
        output_queue : multiprocessing.Queue
            Output ReconstructedWave queue.
        propagation_distance : float
        """
        self.input_queue = in_queue
        self.output_queue = out_queue
        self.propagation_distance = prop_distance
        self.worker = None

    def start(self):
        # To run in a process, a function must be pickable, and thus not a method
        # Therefore, self._main_loop() is wrapped by an external function
        # A Process object is used (instead of Thread) so that the Hologram.reconstruct
        # method is free to multithread the computation.
        self.worker = Process(target = _reactor_main_loop, args = (self,))
        self.worker.start()

    def stop(self):
        if self.worker.is_alive():
            self.worker.terminate()
    
    def _main_loop(self):
        """ Loop checking for holograms in the input queue and reconstructing them. """

        while True:           
            hologram = self.input_queue.get()
            self.output_queue.put(hologram.reconstruct(propagation_distance = self.propagation_distance))