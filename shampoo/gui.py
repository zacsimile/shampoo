"""
Graphical User Interface to the SHAMPOO API.

Author: Laurent P. Rene de Cotret
"""

from multiprocessing import Queue
import numpy as np
import os
from pyqtgraph import QtGui, QtCore
import pyqtgraph as pg
from .reactor import ShampooController, Reactor
from .reconstruction import Hologram, ReconstructedWave
import sys
from .widgets import DataViewer, ReconstructedHologramViewer, PropagationDistanceSelector

DEFAULT_PROPAGATION_DISTANCE = 0.03658

class App(QtGui.QMainWindow):
    """
    
    Attributes
    ----------
    """
    def __init__(self):
        super(App, self).__init__()

        self.data_viewer = None
        self.reconstructed_viewer = None
        self.propagation_distance_selector = None

        self._init_ui()
        self._init_actions()
        self._connect_signals()

        self.controller = ShampooController(out_queue = self.reconstructed_viewer.display_queue)
    
    def load_data(self):
        """ Load a hologram into memory and displays it. """
        path = self.file_dialog.getOpenFileName(self, 'Load holographic data', filter = '*tif')
        hologram = Hologram.from_tif(os.path.abspath(path))
        self.data_viewer.display_data(hologram)
        self.controller.send_data(data = hologram)

    @QtCore.pyqtSlot(object)
    def update_propagation_distance(self, value):
        """ 
        Updates the Controller with a new propagation distance value.
        
        Parameters
        ----------
        value : float or iterable
            Propagation distance(s) at which to reconstruct holograms.
        """
        self.controller.propagation_distance = value

    ### Boilerplate ###

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

        # Assemble window
        self.splitter.addWidget(self.data_viewer)
        self.splitter.addWidget(self.propagation_distance_selector)
        self.splitter.addWidget(self.reconstructed_viewer)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.splitter)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)
        
        self.setGeometry(500, 500, 800, 800)
        self.setWindowTitle('SHAMPOO')
        self._center_window()
        self.show()
    
    def _init_actions(self):
        self.load_data_action = QtGui.QAction('&Load raw data', self)
        self.load_data_action.triggered.connect(self.load_data)
        self.file_menu.addAction(self.load_data_action)
    
    def _connect_signals(self):
        self.propagation_distance_selector.propagation_distance_signal.connect(self.update_propagation_distance)
    
    def _center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

def run():   
    app = QtGui.QApplication(sys.argv)
    gui = App()
    
    sys.exit(app.exec_())