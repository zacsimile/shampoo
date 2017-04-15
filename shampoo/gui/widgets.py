from __future__ import absolute_import

import os
from multiprocessing import Queue

import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui

from ..reconstruction import Hologram, ReconstructedWave

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

class TimeSeriesControls(QtGui.QWidget):
    """ Control of TimeSeries, as well as some metadata """
    
    time_point_request_signal = QtCore.pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super(TimeSeriesControls, self).__init__(*args, **kwargs)
        self.time_points = None

        self.filename_label = QtGui.QLabel('', self)
        self.filename_label.setAlignment(QtCore.Qt.AlignCenter)

        self.time_point_label = QtGui.QLabel('', self)
        self.time_point_label.setAlignment(QtCore.Qt.AlignLeft)

        self.time_point_slider = QtGui.QSlider(parent = self)
        self.time_point_slider.setTracking(True)
        self.time_point_slider.setOrientation(QtCore.Qt.Horizontal)

        # Update label with time_point
        self.time_point_slider.valueChanged.connect(self.time_point_label.setNum)
        self.time_point_slider.valueChanged.connect(self._update)

        time_label = QtGui.QLabel('Time point: ', self)
        time_label.setAlignment(QtCore.Qt.AlignLeft)
        controls = QtGui.QHBoxLayout()
        controls.addWidget(time_label)
        controls.addWidget(self.time_point_label)
        controls.addWidget(self.time_point_slider)

        title = QtGui.QLabel('Time-series controls')
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout = QtGui.QVBoxLayout()
        layout.addWidget(title)
        layout.addWidget(self.filename_label)
        layout.addLayout(controls)
        self.setLayout(layout)

        self.hide()
    
    @QtCore.pyqtSlot(dict)
    def update_metadata(self, metadata):
        """ Update TimeSeries metadata """
        if not metadata:
            self.hide()
            return
        
        self.time_points = metadata['time_points']
        self.time_point_slider.setRange(0, len(metadata['time_points']) - 1)
        self.time_point_slider.setValue(0)
        self.filename_label.setText('File: {}'.format(metadata['filename']))
        self.show()
    
    @QtCore.pyqtSlot(int)
    def _update(self, index):
        self.time_point_request_signal.emit(self.time_points[index])


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
        self.phase_viewer.setImage(img = np.nan_to_num(reconstructed.phase), xvals = xvals)
        self.fourier_mask_viewer.setImage(img = fourier_mask, xvals = xvals)
        
    @QtCore.pyqtSlot()
    def clear(self):
        self.amplitude_viewer.clear()
        self.phase_viewer.clear()
