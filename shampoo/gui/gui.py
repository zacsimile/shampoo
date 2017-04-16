"""
Graphical User Interface to the SHAMPOO API.
"""
from __future__ import absolute_import

import functools
import os.path
import sys
import traceback

import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from skimage import img_as_bool
from skimage.io import imsave

from . import error_aware
from ..reconstruction import Hologram, ReconstructedWave
from ..time_series import TimeSeries
from .camera import AlliedVisionCamera, available_cameras
from .controller import ShampooController
from .debug import DebugCamera
from .fourier_mask_dialog import FourierMaskDialog
from .hologram_viewer import HologramViewer
from .recon_params_widget import ReconstructionParametersWidget
from .time_series_creator import TimeSeriesCreator
from .widgets import (CameraFeatureDialog, ReconstructedHologramViewer,
                      ShampooStatusBar, TimeSeriesControls)

def run(debug = False):
    """
    
    """
    app = QtGui.QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create('cde'))
    try:
        gui = App(debug = debug)
        sys.exit(app.exec_())
    finally:
        # Reactor might hang due to an exception
        del app

class App(QtGui.QMainWindow):
    """
    GUI shell to the ShampooController object.

    Widgets
    -------
    data_viewer
        View raw holographic data and associated Fourier plane information
    
    reconstructed_viewer
        View reconstructed holographic data

    reconstruction_parameters_widget
        Select the propagation distance(s) with which to reconstruct
        holograms.
    """
    save_latest_hologram_signal = QtCore.pyqtSignal(object, name = 'save_latest_hologram_signal')
    connect_camera_signal = QtCore.pyqtSignal(object, name = 'connect_camera_signal')

    error_message_signal = QtCore.pyqtSignal(str, name = 'error_message_signal')
    
    def __init__(self, debug = False):
        """
        Parameters
        ----------
        debug : bool, optional
            If True, extra options are available as a debug tool. Default is False.
        """
        super(App, self).__init__()

        self.controller = ShampooController()
        self.debug = debug

        self.data_viewer = HologramViewer(parent = self)
        self.reconstructed_viewer = ReconstructedHologramViewer(parent = self)
        self.reconstruction_parameters_widget = ReconstructionParametersWidget(parent = self)

        self.time_series_controls = TimeSeriesControls(parent = self)
        self.controller.time_series_metadata_signal.connect(self.time_series_controls.update_metadata)
        self.time_series_controls.time_point_request_signal.connect(self.controller.data_from_time_series)

        self.file_dialog = QtGui.QFileDialog(parent = self)
        self.menubar = self.menuBar()

        # Assemble menu from previously-defined actions
        self.file_menu = self.menubar.addMenu('&File')
        self.camera_menu = self.menubar.addMenu('&Camera')

        raw_data_layout = QtGui.QVBoxLayout()
        raw_data_layout.addWidget(self.data_viewer)
        raw_data_layout.addWidget(self.time_series_controls)

        reconstructed_layout = QtGui.QVBoxLayout()
        reconstructed_layout.addWidget(self.reconstructed_viewer)
        reconstructed_layout.addWidget(self.reconstruction_parameters_widget)

        self.status_bar = ShampooStatusBar(parent = self)
        self.setStatusBar(self.status_bar)
        self.status_bar.update_status('Ready')

        self.error_window = QtGui.QErrorMessage(parent = self)
        self.error_message_signal.connect(self.error_window.showMessage)
        self.controller.error_message_signal.connect(self.error_window.showMessage)

        self.layout = QtGui.QHBoxLayout()
        self.layout.addLayout(raw_data_layout)
        self.layout.addLayout(reconstructed_layout)

        self.central_widget = QtGui.QWidget()
        self.central_widget.setLayout(self.layout)
        self.setCentralWidget(self.central_widget)

        self.setGeometry(500, 500, 800, 800)
        self.setWindowTitle('SHAMPOO')
        self._center_window()
        self.showMaximized()

        self.load_time_series_action = QtGui.QAction('&Load time-series', self)
        self.load_time_series_action.triggered.connect(self.load_time_series)
        self.file_menu.addAction(self.load_time_series_action)

        self.load_data_action = QtGui.QAction('&Load raw data', self)
        self.load_data_action.triggered.connect(self.load_data)
        self.file_menu.addAction(self.load_data_action)

        self.save_data_action = QtGui.QAction('&Save raw data', self)
        self.save_data_action.triggered.connect(self.save_raw_data)
        self.file_menu.addAction(self.save_data_action)
        self.save_data_action.setEnabled(False)

        self.file_menu.addSeparator()

        self.time_series_creator_action = QtGui.QAction('&Create hologram time series', self)
        self.time_series_creator_action.triggered.connect(self.launch_time_series_creator)
        self.file_menu.addAction(self.time_series_creator_action)

        self.file_menu.addSeparator()

        self.load_fourier_mask_action = QtGui.QAction('&Load Fourier mask', self)
        self.load_fourier_mask_action.triggered.connect(self.load_fourier_mask)
        self.file_menu.addAction(self.load_fourier_mask_action)

        self.connect_camera_action = QtGui.QAction('&Connect a camera', self)
        self.connect_camera_action.triggered.connect(self.connect_camera)
        self.camera_menu.addAction(self.connect_camera_action)

        self.camera_snapshot_action = QtGui.QAction('&Take camera snapshot', self)
        self.camera_snapshot_action.triggered.connect(self.controller.send_snapshot_data)
        self.camera_menu.addAction(self.camera_snapshot_action)
        self.camera_snapshot_action.setEnabled(False)

        self.camera_features_action = QtGui.QAction('&Change camera features', self)
        self.camera_features_action.triggered.connect(self.change_camera_features)
        self.camera_menu.addAction(self.camera_features_action)
        self.camera_features_action.setEnabled(False)

        self.reconstruction_parameters_widget.propagation_distance_signal.connect(self.controller.update_propagation_distance)
        self.controller.reconstructed_hologram_signal.connect(self.reconstructed_viewer.display)
        self.controller.raw_data_signal.connect(self.data_viewer.display)

        # Save and loads
        self.save_latest_hologram_signal.connect(self.controller.save_latest_hologram)

        # Controller status signals
        self.connect_camera_signal.connect(self.controller.connect_camera)
        self.controller.reconstruction_status_signal.connect(self.status_bar.update_status)

        # What actions are available when a camera is made available
        # These actions will become unavailable when a camera is disconnected.
        self.controller.camera_connected_signal.connect(lambda x: self.status_bar.update_status('Camera connected'))
        self.controller.camera_connected_signal.connect(self.camera_snapshot_action.setEnabled)
        self.controller.camera_connected_signal.connect(self.camera_features_action.setEnabled)
        self.controller.raw_data_signal.connect(lambda x: self.save_data_action.setEnabled(True))

        self.reconstruction_parameters_widget.update_propagation_distance()

    @error_aware('Data could not be loaded.')
    @QtCore.pyqtSlot()
    def load_data(self):
        """ Load a hologram into memory and displays it. """
        path = self.file_dialog.getOpenFileName(self, 'Load holographic data', filter = '*tif')[0]
        hologram = Hologram.from_tif(os.path.abspath(path))
        self.controller.reconstruct(data = hologram)
    
    @error_aware('Fourier mask could not be loaded')
    @QtCore.pyqtSlot()
    def load_fourier_mask(self):
        """ Load a user-defined reconstruction Fourier mask """
        fourier_mask_dialog = FourierMaskDialog(initial_mask = self.controller.fourier_mask)
        fourier_mask_dialog.fourier_mask_update_signal.connect(self.controller.set_fourier_mask)
        success = fourier_mask_dialog.exec_()
    
    @error_aware('Time-series could not be loaded.')
    @QtCore.pyqtSlot()
    def load_time_series(self):
        path = self.file_dialog.getOpenFileName(self, 'Load time-series', filter = '*hdf5 *.h5')[0]
        self.controller.load_time_series(path)
    
    @error_aware('The hologram time series could not be created.')
    @QtCore.pyqtSlot()
    def launch_time_series_creator(self):
        time_series_creator = TimeSeriesCreator(parent = self)
        time_series_creator.time_series_path_signal.connect(self.controller.load_time_series)
        success = time_series_creator.exec_()
    
    @error_aware('Raw data could not be saved.')
    @QtCore.pyqtSlot()
    def save_raw_data(self):
        """ Save a raw hologram from the raw data screen """
        path = self.file_dialog.getSaveFileName(self, 'Save holographic data', filter = '*tif')
        if not path.endswith('.tif'):
            path = path + '.tif'
        self.save_latest_hologram_signal.emit(path)
    
    @error_aware('Camera could not be connected')
    @QtCore.pyqtSlot()
    def connect_camera(self):
        """ Bring up a modal dialog to choose amongst available cameras. """
        cameras = available_cameras()

        if self.debug:
            cameras.append('debug')
        
        if not cameras:
            self.error_message_signal.emit('No cameras available.')
            return
        
        camera_ID, ok = QtGui.QInputDialog.getItem(self, 'Select camera', 'List of cameras', 
                                                   cameras, 0, False)
        
        if ok and camera_ID:
            self.connect_camera_signal.emit(camera_ID)
    
    @error_aware('Camera features could not be updated.')
    @QtCore.pyqtSlot()
    def change_camera_features(self):
        self.camera_features_dialog = CameraFeatureDialog(camera = self.controller.camera, parent = self)
        self.camera_features_dialog.camera_features_update_signal.connect(self.controller.update_camera_features)
        success = self.camera_features_dialog.exec_()
        if not success:
            raise RuntimeError
    
    def closeEvent(self, event):
        reply = QtGui.QMessageBox.question(self, 'SHAMPOO', 'Are you sure you want to quit?', 
                                           QtGui.QMessageBox.Yes, QtGui.QMessageBox.No)

        if reply == QtGui.QMessageBox.Yes:
            event.accept()
            self.controller.stop()
        else:
            event.ignore()
    
    def _center_window(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())
