"""
Graphical User Interface to the SHAMPOO API.

Usage
-----
>>> from shampoo.gui import run
>>> run()
"""
from .camera import available_cameras, AlliedVisionCamera
import numpy as np
import os
from pyqtgraph import QtGui, QtCore
import pyqtgraph as pg
from .reactor import Reactor, ProcessReactor, ThreadSafeQueue, ProcessSafeQueue
from ..reconstruction import Hologram, ReconstructedWave
import sys
from .widgets import ShampooWidget, DataViewer, ReconstructedHologramViewer, PropagationDistanceSelector

DEFAULT_PROPAGATION_DISTANCE = 0.03658

def run():   
    app = QtGui.QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create('cde'))
    gui = App()
    
    sys.exit(app.exec_())

def _reconstruct_hologram(item):
    """ Function wrapper to Hologram.reconstruct and Hologram.reconstruct_multithread. 
    item : 2-tuple
        (propagation_distance, hologram)
    """
    propagation_distance, hologram = item
    if len(propagation_distance) == 1:
        return (propagation_distance, hologram.reconstruct(propagation_distance = propagation_distance[0]))
    else:
        return (propagation_distance, hologram.reconstruct_multithread(propagation_distances = propagation_distance))

class ShampooController(QtCore.QObject):
    """
    Underlying controller to SHAMPOO's Graphical User Interface

    Signals
    -------
    reconstructed_hologram_signal
        Emits a reconstructed hologram whenever one is available.
    
    raw_data_signal
        Emits holographic data whenever one is loaded into memory.
    
    camera_connected_signal
        Emits True when a camera has been successfully connected, and False otherwise.
    
    Slots
    -------
    send_data
        Send raw holographic data, to be reconstructed.
    
    send_snapshot_data
        Send raw holographic data to be reconstructed, from a camera snapshot.
    
    update_propagation_distance
        Change the propagation distance(s) used in the holographic reconstruction process.
    
    connect_camera
        Connect a camera by ID. Check for available cameras using available_cameras()
    
    Methods
    -------
    choose_camera
        Choose camera from a list of ID. Not implemented.
    """
    raw_data_signal = QtCore.pyqtSignal(object, name = 'raw_data_signal')
    reconstructed_hologram_signal = QtCore.pyqtSignal(object, name = 'reconstructed_hologram_signal')

    camera_connected_signal = QtCore.pyqtSignal(bool, name = 'camera_connected_signal')

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        output_function: callable
        """
        super(ShampooController, self).__init__(**kwargs)
        self.reconstructed_queue = ProcessSafeQueue()
        self.propagation_distance = [DEFAULT_PROPAGATION_DISTANCE]
        
        self.camera = None
        self.camera_connected_signal.emit(False)

        # Wire up reactors
        self.reconstruction_reactor = ProcessReactor(function = _reconstruct_hologram, output_queue = self.reconstructed_queue)
        self.display_reactor = Reactor(input_queue = self.reconstructed_queue, callback = self.reconstructed_hologram_signal.emit)
        self.reconstruction_reactor.start(), self.display_reactor.start()
    
    def __del__(self):
        self.reconstruction_reactor.stop()
        self.display_reactor.stop()
    
    @QtCore.pyqtSlot(object)
    def send_snapshot_data(self):
        """
        Send holographic data from the camera to the reconstruction reactor.
        """
        data = self.camera.snapshot()
        self.send_data(data)
    
    @QtCore.pyqtSlot(object)
    def send_data(self, data):
        """ 
        Send holographic data to the reconstruction reactor.

        Parameters
        ----------
        tup : ndarray or Hologram object
            Can be any type that can is accepted by the Hologram() constructor.
        """
        if not isinstance(data, Hologram):
            data = Hologram(data)
        self.raw_data_signal.emit(data)
        self.reconstruction_reactor.send_item( (self.propagation_distance, data) )
    
    @QtCore.pyqtSlot(object)
    def update_propagation_distance(self, item):
        """
        Thread-safe PyQt slot API to updating the propagation distance. 

        Parameters
        ----------
        item : array-like
            Propagation distances in meters.
        """
        self.propagation_distance = item
    
    @QtCore.pyqtSlot(object)
    def connect_camera(self, ID):
        """ Connect camera by ID. """
        # TODO: generalize to other manufacturers
        # This method should never fail. available_cameras() must have been called
        # before so that connecting is always successful.
        self.camera = AlliedVisionCamera(ID)
        self.camera_connected_signal.emit(True)

class App(ShampooWidget, QtGui.QMainWindow):
    """
    GUI shell to the ShampooController object.

    Widgets
    -------
    data_viewer
        View raw holographic data
    
    reconstructed_viewer
        View reconstructed holographic data

    propagation_distance_selector
        Select the propagation distance(s) with which to reconstruct
        holograms.
    """
    connect_camera_signal = QtCore.pyqtSignal(object, name = 'connect_camera_signal')
    
    def __init__(self):

        self.data_viewer = None
        self.reconstructed_viewer = None
        self.propagation_distance_selector = None
        self.controller = ShampooController()

        super(App, self).__init__()

    @QtCore.pyqtSlot()
    def load_data(self):
        """ Load a hologram into memory and displays it. """
        path = self.file_dialog.getOpenFileName(self, 'Load holographic data', filter = '*tif')
        hologram = Hologram.from_tif(os.path.abspath(path))
        self.controller.send_data(data = hologram)
    
    @QtCore.pyqtSlot()
    def connect_camera(self):
        """ Bring up a modal dialog to choose amongst available cameras. """
        cameras = available_cameras()
        if not cameras:
            error_window = QtGui.QErrorMessage(self)
            return error_window.showMessage('No cameras available. ')
        
        camera_ID, ok = QtGui.QInputDialog.getItem(self, 'Select camera', 'List of cameras', 
                                                   cameras, 0, False)
        
        if ok and camera_ID:
            self.connect_camera_signal.emit(camera_ID)

    def _init_ui(self):
        """
        Method initializing UI components.
        """
        self.data_viewer = DataViewer(parent = self)
        self.reconstructed_viewer = ReconstructedHologramViewer(parent = self)
        self.propagation_distance_selector = PropagationDistanceSelector(parent = self)

        self.file_dialog = QtGui.QFileDialog(parent = self)
        self.menubar = self.menuBar()
        self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)

        # Assemble menu from previously-defined actions
        self.file_menu = self.menubar.addMenu('&File')
        self.camera_menu = self.menubar.addMenu('&Camera')
        self.main_splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.right_splitter = QtGui.QSplitter(QtCore.Qt.Vertical)

        # Assemble window
        self.right_splitter.addWidget(self.propagation_distance_selector)
        self.right_splitter.addWidget(self.reconstructed_viewer)
        self.main_splitter.addWidget(self.data_viewer)
        self.main_splitter.addWidget(self.right_splitter)

        self.layout = QtGui.QHBoxLayout()
        self.layout.addWidget(self.main_splitter)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        self.setGeometry(500, 500, 800, 800)
        self.setWindowTitle('SHAMPOO')
        self._center_window()
        self.showMaximized()
    
    def _init_actions(self):
        self.load_data_action = QtGui.QAction('&Load raw data', self)
        self.load_data_action.triggered.connect(self.load_data)
        self.file_menu.addAction(self.load_data_action)

        self.connect_camera_action = QtGui.QAction('&Connect a camera', self)
        self.connect_camera_action.triggered.connect(self.connect_camera)
        self.camera_menu.addAction(self.connect_camera_action)

        self.camera_snapshot_action = QtGui.QAction('&Take camera snapshot', self)
        self.camera_snapshot_action.triggered.connect(self.controller.send_snapshot_data)
        self.camera_menu.addAction(self.camera_snapshot_action)
        self.camera_snapshot_action.setEnabled(False)
    
    def _connect_signals(self):
        self.propagation_distance_selector.propagation_distance_signal.connect(self.controller.update_propagation_distance)
        self.controller.reconstructed_hologram_signal.connect(self.reconstructed_viewer.display)
        self.controller.raw_data_signal.connect(self.data_viewer.display)

        # Signify to the controller to connect to a new camera
        self.connect_camera_signal.connect(self.controller.connect_camera)
        self.controller.camera_connected_signal.connect(self.camera_snapshot_action.setEnabled)
    
    def _center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())