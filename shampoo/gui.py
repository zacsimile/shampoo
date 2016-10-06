"""
Graphical User Interface to the SHAMPOO API.

Author: Laurent P. Rene de Cotret
"""

from .gui_utils import ComputationThread, InProgressWidget
import numpy as np
import os
from PyQt4 import QtGui, QtCore
import pyqtgraph as pg
from .reconstruction import Hologram, ReconstructedWave
import sys

class DataViewer(pg.ImageView):
    """
    QWidget displaying the raw holograms, before reconstruction.

    Attributes
    ----------
    parent : QObject displaying the viewer

    Methods
    -------
    display_data
    """

    def __init__(self, parent):
        """
        Parameters
        ----------
        parent : QObject
        """
        super(DataViewer, self).__init__()
        self.parent = parent

        self._init_ui()
        self._connect_signals()
    
    @QtCore.pyqtSlot(object)
    def display_data(self, data):
        """
        Displays a NumPy array or arrays.

        Parameters
        ----------
        data : ndarray or shampoo.Hologram
        """
        if isinstance(data, Hologram):
            data = data.hologram
        self.setImage(data)

    ### Boilerplate ###

    def _init_ui(self):

        self.progress_widget = InProgressWidget(parent = self)
        self.progress_widget.hide()
    
    def _connect_signals(self):
        pass
    
    def resizeEvent(self, event):
        self.progress_widget.resize(event.size())
        event.accept()

class ReconstructedHologramViewer(QtGui.QWidget):
    """
    QWidget displaying the reconstructed wave holographic data, in two plots: phase and amplitude.
    """

    def __init__(self, parent):
        """
        Parameters
        ----------
        parent : QObject
        """
        super(ReconstructedHologramViewer, self).__init__()
        self.parent = parent

        self._init_ui()
        self._connect_signals()
    
    @QtCore.pyqtSlot(object)
    def display_reconstructed(self, data):
        """
        Dsplays the amplitude and phase information of a reconstructed hologram.

        Parameters
        ----------
        array : ndarray, dtype complex or shampoo.ReconstructedWave instance
        """
        if isinstance(data, ReconstructedWave):
            intensity, phase = data.intensity, data.phase
        else:
            intensity, phase = np.absolute(data), np.angle(data, deg = False)
        
        self.amplitude_viewer.setImage(img = intensity)
        self.phase_viewer.setImage(img = phase)
    
    def clear_view(self):
        self.display_reconstructed(np.zeros(shape = (100, 100), dtype = np.complex))

    ### Boilerplate ###

    def _init_ui(self):

        self.amplitude_viewer = pg.ImageView(parent = self, name = 'Reconstructed amplitude')
        self.phase_viewer = pg.ImageView(parent = self, name = 'Reconstructed phase')

        self.progress_widget = InProgressWidget(parent = self)
        self.progress_widget.hide()

        #Assemble window
        amplitude_layout = QtGui.QVBoxLayout()
        amplitude_layout.addWidget(QtGui.QLabel('Reconstructed amplitude', parent = self))
        amplitude_layout.addWidget(self.amplitude_viewer)

        phase_layout = QtGui.QVBoxLayout()
        phase_layout.addWidget(QtGui.QLabel('Reconstructed phase', parent = self))
        phase_layout.addWidget(self.phase_viewer)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(amplitude_layout)
        self.layout.addLayout(phase_layout)
        self.setLayout(self.layout)
    
    def _connect_signals(self):
        pass
    
    def resizeEvent(self, event):
        self.progress_widget.resize(event.size())
        event.accept()

class App(QtGui.QMainWindow):
    """
    
    Attributes
    ----------
    """
    def __init__(self):
        super(App, self).__init__()

        self.hologram = None

        self._init_ui()
        self._init_actions()
        self._connect_signals()
    
    def load_data(self):
        """ Load a hologram into memory and displays it. """
        path = self.file_dialog.getOpenFileName(self, 'Load holographic data', filter = '*tif')
        self.hologram = Hologram.from_tif(os.path.abspath(path))
        self.reconstructed_viewer.clear_view()
        self.data_viewer.display_data(data = self.hologram)
    
    def reconstruct_hologram(self):
        """ Reconstruct hologram. """
        prop_distance, ok = self.reconstruction_parameters_dialog.getText(self, 'Reconstruction Parameters', 'Propagation distance (m)', text = str(0.03685))
        if ok:
            self.worker = ComputationThread(self.hologram.reconstruct, float(prop_distance))
            self.worker.results_signal.connect(self.reconstructed_viewer.display_reconstructed)

            # Optional: progress widget
            self.worker.in_progress_signal.connect(self.data_viewer.progress_widget.show)
            self.worker.done_signal.connect(self.data_viewer.progress_widget.hide)
            self.worker.start()

    ### Boilerplate ###

    def _init_actions(self):
        """ 
        Connects the menubar actions with other methods. 
        """
        self.load_data_action = QtGui.QAction('&Load raw data', self)
        self.load_data_action.triggered.connect(self.load_data)
        self.file_menu.addAction(self.load_data_action)

        self.reconstruct_hologram_action = QtGui.QAction('&Reconstruct hologram', self)
        self.reconstruct_hologram_action.triggered.connect(self.reconstruct_hologram)
        self.hologram_menu.addAction(self.reconstruct_hologram_action)

    def _init_ui(self):
        """
        Method initializing UI components.
        """
        self.data_viewer = DataViewer(parent = self)
        self.reconstructed_viewer = ReconstructedHologramViewer(parent = self)

        self.reconstruction_parameters_dialog = QtGui.QInputDialog(parent = self)
        self.file_dialog = QtGui.QFileDialog(parent = self)
        self.menubar = self.menuBar()
        self.splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)

        # Assemble menu from previously-defined actions
        self.file_menu = self.menubar.addMenu('&File')
        self.hologram_menu = self.menubar.addMenu('&Hologram')

        # Assemble window
        self.splitter.addWidget(self.data_viewer)
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
    
    def _connect_signals(self):
        pass
    
    def _center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

def run():   
    app = QtGui.QApplication(sys.argv)
    gui = App()
    
    sys.exit(app.exec_())