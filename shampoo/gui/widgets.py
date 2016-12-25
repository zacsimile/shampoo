from __future__ import absolute_import

from multiprocessing import Queue
import numpy as np
import os
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from ..reconstruction import Hologram, ReconstructedWave
from ..fftutils import fftshift

# Try importing optional dependency PyFFTW for Fourier transforms. If the import
# fails, import scipy's FFT module instead
try:
    from pyfftw.interfaces.scipy_fftpack import fft2, ifft2
except ImportError:
    from scipy.fftpack import fft2, ifft2

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
    
    @QtCore.pyqtSlot()
    def reject(self):
        super(CameraFeatureDialog, self).reject()
    
class RawDataViewer(QtGui.QWidget):
    """
    QWidget displaying raw holograms, as well as related information
    such as Fourier decomposition.
    """
    def __init__(self, *args, **kwargs):
        super(RawDataViewer, self).__init__(*args, **kwargs)

        self.raw_data_viewer = pg.ImageView(parent = self, name = 'Raw data')
        self.fourier_plane_viewer = pg.ImageView(parent = self, name = 'Fourier plane')
        self.fourier_mask_viewer = pg.ImageView(parent = self, name = 'Fourier mask viewer')

        tabs = QtGui.QTabWidget(parent = self)
        tabs.addTab(self.raw_data_viewer, 'Raw hologram')
        tabs.addTab(self.fourier_plane_viewer, 'Fourier plane')
        tabs.addTab(self.fourier_mask_viewer, 'Fourier mask')

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
        self.fourier_plane_viewer.setImage(np.real(fftshift(fft2(data.hologram))))
        #self.fourier_mask_viewer.setImage(data.fourier_mask)

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

        self.tabs = QtGui.QTabWidget()
        self.tabs.addTab(self.amplitude_viewer, 'Amplitude')
        self.tabs.addTab(self.phase_viewer, 'Phase')

        self.layout = QtGui.QVBoxLayout()
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot(object)
    def display(self, data_tup):
        """
        Dsplays the amplitude and phase information of a reconstructed hologram.

        Parameters
        ----------
        data_tup : tuple of ndarrays,
            Contains the propagation distance information, and an array of dtype complex or ReconstructedWave instance
        """
        xvals, data = data_tup
        xvals = np.array(xvals)

        if not isinstance(data, ReconstructedWave):
            data = ReconstructedWave(data)
        self.amplitude_viewer.setImage(img = data.intensity, xvals = xvals)
        self.phase_viewer.setImage(img = data.phase, xvals = xvals)
        
    @QtCore.pyqtSlot()
    def clear(self):
        self.amplitude_viewer.clear()
        self.phase_viewer.clear()

class PropagationDistanceSelector(QtGui.QWidget):
    """
    QWidget allowing the user to select an array of propagation distances

    Signals
    -------
    propagation_distance_signal
        Emitted when propagation distances are modified.
    """
    propagation_distance_signal = QtCore.pyqtSignal(object, name = 'propagation_distance')

    def __init__(self, *args, **kwargs):

        super(PropagationDistanceSelector, self).__init__(*args, **kwargs)

        self.start_value_widget = None
        self.stop_value_widget = None
        self.step_value_widget = None

        # Widgets
        self.start_value_widget = QtGui.QLineEdit(parent = self)
        self.stop_value_widget = QtGui.QLineEdit(parent = self)
        self.step_value_widget = QtGui.QLineEdit(parent = self)

        # Initial values
        self.start_value_widget.setText(str(DEFAULT_PROPAGATION_DISTANCE))
        self.stop_value_widget.setText('0.0')
        self.step_value_widget.setText('0.0')

        self.start_value_widget.editingFinished.connect(self.update_propagation_distance)
        self.stop_value_widget.editingFinished.connect(self.update_propagation_distance)
        self.step_value_widget.editingFinished.connect(self.update_propagation_distance)

        # Final layout
        self.layout = QtGui.QVBoxLayout()

        self.start_layout = QtGui.QVBoxLayout()
        self.start_layout.addWidget(QtGui.QLabel('Start (m)'))
        self.start_layout.addWidget(self.start_value_widget)

        self.stop_layout = QtGui.QVBoxLayout()
        self.stop_layout.addWidget(QtGui.QLabel('Stop (m)'))
        self.stop_layout.addWidget(self.stop_value_widget)

        self.step_layout = QtGui.QVBoxLayout()
        self.step_layout.addWidget(QtGui.QLabel('Step (m)'))
        self.step_layout.addWidget(self.step_value_widget)

        self.values_layout = QtGui.QHBoxLayout()
        self.values_layout.addLayout(self.start_layout)
        self.values_layout.addLayout(self.stop_layout)
        self.values_layout.addLayout(self.step_layout)

        self._title_layout = QtGui.QHBoxLayout()
        self._title_layout.addWidget(QtGui.QLabel(text = 'Propagation distance', parent = self))
        self.layout.addLayout(self._title_layout)
        self.layout.addLayout(self.values_layout)
        self.setLayout(self.layout)
    
    @QtCore.pyqtSlot()
    def update_propagation_distance(self):
        """ Emits the propagation_distance signal with the sorted propagation distance data parsed from the widget. """

        start, stop, step = [float(widget.text()) for widget in (self.start_value_widget, self.stop_value_widget, self.step_value_widget)]

        if step == 0:
            propagation_distance = [start]
        else:
            # Singe arange does not include the 'stop' value, add another step
            propagation_distance = np.arange(start = start, stop = stop + step, step = step)
        
        self.propagation_distance_signal.emit(propagation_distance)