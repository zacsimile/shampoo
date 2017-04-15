from __future__ import absolute_import

from multiprocessing import Queue
import numpy as np
import os
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from ..reconstruction import Hologram, ReconstructedWave

# Try importing optional dependency PyFFTW for Fourier transforms. If the import
# fails, import scipy's FFT module instead
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
    from scipy.fftpack import fftshift
except ImportError:
    from scipy.fftpack import fftshift, fft2, ifft2

ICONS_FOLDER = os.path.join(os.path.dirname(__file__), 'icons')
DEFAULT_PROPAGATION_DISTANCE = 0.03658

#########################################################################################
###             GUI COMPONENTS

class ShampooStatusBar(QtGui.QStatusBar):
    """
    QStatusBar subclass with a simplied API to update the status.
    """
    def __init__(self, *args, **kwargs):
        super(ShampooStatusBar, self).__init__(*args, **kwargs)
        self.status_label = QtGui.QLabel()
        self.addPermanentWidget(self.status_label)
    
    @QtCore.pyqtSlot(str)
    def update_status(self, message):
        self.status_label.setText(message)

class CameraFeatureDialog(QtGui.QDialog):
    """
    Modal dialog used to set camera features within SHAMPOO
    """
    camera_features_update_signal = QtCore.pyqtSignal(dict, name = 'camera_features_update_signal')

    def __init__(self, camera, parent = None):
        """
        Parameters
        ----------
        camera : shampoo.gui.Camera instance

        parent : QWidget or None, optional
        """
        self.camera = camera
        super(CameraFeatureDialog, self).__init__()

        self.setModal(True)
        self.save_btn = QtGui.QPushButton('Update values', self)
        self.save_btn.clicked.connect(self.accept)

        self.cancel_btn = QtGui.QPushButton('Cancel changes', self)
        self.cancel_btn.clicked.connect(self.reject)
        self.cancel_btn.setDefault(True)

        # Order in which features are enumerated must be the same as ordered in self.camera.features
        # The order is assumed here
        feature_labels = [QtGui.QLabel(label) for label in ('Exposure (us)', 'Resolution (px, px)', 'Bit depth')]
        self.exposure_edit = QtGui.QLineEdit(str(self.camera.exposure), parent = self)
        self.resolution_edit = QtGui.QLineEdit(str(self.camera.resolution), parent = self)
        self.bit_depth_edit = QtGui.QLineEdit(str(self.camera.bit_depth), parent = self) 
        feature_edit = [self.exposure_edit, self.resolution_edit, self.bit_depth_edit]

        #Set access mode
        # TODO: set automatically?
        self.resolution_edit.setReadOnly(True)
        self.bit_depth_edit.setReadOnly(True)

        self.labels_layout = QtGui.QVBoxLayout()
        self.values_layout = QtGui.QVBoxLayout()
        for label, edit in zip( feature_labels, feature_edit ):
            self.labels_layout.addWidget(label)
            self.values_layout.addWidget(edit)

        self.buttons = QtGui.QHBoxLayout()
        self.buttons.addWidget(self.save_btn)
        self.buttons.addWidget(self.cancel_btn)
        
        self.list_layout = QtGui.QHBoxLayout()
        self.list_layout.addLayout(self.labels_layout)
        self.list_layout.addLayout(self.values_layout)

        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(self.list_layout)
        self.layout.addLayout(self.buttons)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot()
    def accept(self):
        feature_dict = {'exposure': int(self.exposure_edit.text()), 'bit_depth': int(self.bit_depth_edit.text())}
        self.camera_features_update_signal.emit(feature_dict)
        super(CameraFeatureDialog, self).accept()
    
class RawDataViewer(QtGui.QWidget):
    """
    QWidget displaying raw holograms, as well as related information
    such as Fourier decomposition.
    """
    def __init__(self, *args, **kwargs):
        super(RawDataViewer, self).__init__(*args, **kwargs)

        self.raw_data_viewer = pg.ImageView(parent = self, name = 'Raw data')
        self.fourier_plane_viewer = pg.ImageView(parent = self, name = 'Fourier plane')

        tabs = QtGui.QTabWidget(parent = self)
        tabs.addTab(self.raw_data_viewer, 'Raw hologram')
        tabs.addTab(self.fourier_plane_viewer, 'Fourier plane')

        layout = QtGui.QHBoxLayout()
        layout.addWidget(tabs)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot(object)
    def display(self, data):
        """
        Display raw hologram and associated Fourier plane information.

        Parameters
        ----------
        data : Hologram or ndarray
        """
        if not isinstance(data, Hologram):
            data = Hologram(data)
        
        self.raw_data_viewer.setImage(data.hologram)
        ft = fftshift(fft2(data.hologram, axes = (0, 1)), axes = (0,1))
        self.fourier_plane_viewer.setImage(ft.real)

class ReconstructedHologramViewer(QtGui.QWidget):
    """
    QWidget displaying the reconstructed wave holographic data, in two plots: phase and amplitude.

    Slots
    -----
    display
        Display the phase and amplitude information of a reconstructed hologram.
    
    clear
        Clear view.
    """

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        parent : QObject
        """
        super(ReconstructedHologramViewer, self).__init__(*args, **kwargs)

        self.amplitude_viewer = pg.ImageView(parent = self, name = 'Reconstructed amplitude')
        self.phase_viewer = pg.ImageView(parent = self, name = 'Reconstructed phase')
        self.fourier_mask_viewer = pg.ImageView(parent = self, name = 'Reconstruction Fourier mask')

        self.tabs = QtGui.QTabWidget()
        self.tabs.addTab(self.amplitude_viewer, 'Amplitude')
        self.tabs.addTab(self.phase_viewer, 'Phase')
        self.tabs.addTab(self.fourier_mask_viewer, 'Fourier mask')

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # Set to maximal size, since this is the star of the show
        self.resize(self.maximumSize())
    
    @QtCore.pyqtSlot(object)
    def display(self, data_tup):
        """
        Dsplays the amplitude and phase information of a reconstructed hologram.

        Parameters
        ----------
        data_tup : tuple of ndarrays,
            Contains the propagation distance information, and a ReconstructedWave instance
        """
        xvals, reconstructed = data_tup
        xvals = np.array(xvals)
        fourier_mask = reconstructed.fourier_mask

        self.amplitude_viewer.setImage(img = reconstructed.intensity, xvals = xvals)
        self.phase_viewer.setImage(img = reconstructed.phase, xvals = xvals)
        self.fourier_mask_viewer.setImage(img = fourier_mask, xvals = xvals)
        
    @QtCore.pyqtSlot()
    def clear(self):
        self.amplitude_viewer.clear()
        self.phase_viewer.clear()