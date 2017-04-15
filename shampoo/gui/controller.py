from __future__ import absolute_import

import numpy as np
from pyqtgraph import QtCore

from . import error_aware
from .. import Hologram, TimeSeries
from .reactor import Reconstructor


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
        Emits True when a camera has been successfully connected, and False when disconnected.
    
    Slots
    -------
    load_time_series[str]
        Load an HDF5-based TimeSeries object

    send_data[object]
        Send raw holographic data, to be reconstructed.

    data_from_time_series[float]
        Load a Hologram from a time-series.
    
    send_snapshot_data
        Send raw holographic data to be reconstructed, from a camera snapshot.
    
    update_propagation_distance
        Change the propagation distance(s) used in the holographic reconstruction process.
    
    update_camera_features
    
    connect_camera
        Connect a camera by ID. Check for available cameras using available_cameras()
    
    Methods
    -------
    choose_camera
        Choose camera from a list of ID. Not implemented.
    """
    raw_data_signal = QtCore.pyqtSignal(object)
    reconstructed_hologram_signal = QtCore.pyqtSignal(object)

    # Time series metadata
    time_series_metadata_signal = QtCore.pyqtSignal(dict)

    # Status signals
    reconstruction_in_progress_signal = QtCore.pyqtSignal(str)
    reconstruction_complete_signal = QtCore.pyqtSignal(str)
    camera_connected_signal = QtCore.pyqtSignal(bool)

    error_message_signal = QtCore.pyqtSignal(str)

    def __init__(self, **kwargs):
        super(ShampooController, self).__init__(**kwargs)

        self.time_series = None

        self.propagation_distance = list()
        self.fourier_mask = None
        
        self.camera = None
        self.camera_connected_signal.emit(False)

        # Hologram reconstruction and display
        def display_callback(item):
            self.reconstructed_hologram_signal.emit(item)
            self.reconstruction_complete_signal.emit('Reconstruction complete') 
        
        self.reconstruction_reactor = Reconstructor(callback = display_callback)
        self.reconstruction_reactor.start()

        # Private attributes
        self._latest_hologram = None
    
    @error_aware('Time series could not be loaded')
    @QtCore.pyqtSlot(str)
    def load_time_series(self, path):
        """
        Load TimeSeries object into the controller

        Parameters
        ----------
        path : str
            Path to the HDF5 file
        """
        if self.time_series is not None:
            self.time_series.close()
        
        self.time_series = TimeSeries(path, mode = 'r+')
        metadata = dict(self.time_series.attrs)
        metadata.update({'filename': path})
        self.time_series_metadata_signal.emit(metadata)

    @QtCore.pyqtSlot()
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
        data : ndarray or Hologram object
            Can be any type that can is accepted by the Hologram() constructor.
        """
        if self.time_series is not None:
            # 'unload' the existing time-series
            self.time_series.close()
            self.time_series_metadata_signal.emit(dict())
            self.time_series = None

        if not isinstance(data, Hologram):
            data = Hologram(data)
        
        self._latest_hologram = data
        self.raw_data_signal.emit(data)

        self.reconstruction_reactor.send_item( (self.propagation_distance, data, self.fourier_mask) )
        self.reconstruction_in_progress_signal.emit('Reconstruction in progress...')
    
    @error_aware('Data could not be loaded from time-series')
    @QtCore.pyqtSlot(float)
    def data_from_time_series(self, time_point):
        """ Display raw data and reconstruction from TimeSeries """
        hologram = self.time_series.hologram(time_point)
        self.send_data(hologram)
        # TODO: see if reconstruction exists, and bypass reactor if possible
    
    @error_aware('Latest hologram could not be saved.')
    @QtCore.pyqtSlot(object)
    def save_latest_hologram(self, path):
        """
        Save latest raw holographic data into a HDF5

        Parameters
        ----------
        path : str or path-like object
        """
        imsave(path, arr = self._latest_hologram.hologram, plugin = 'tifffile')
    
    @error_aware('Fourier mask could not be set.')
    @QtCore.pyqtSlot(object)
    def set_fourier_mask(self, mask):
        self.fourier_mask = img_as_bool(mask)
        # Refresh screen
        if self._latest_hologram:
            self.send_data(self._latest_hologram)
    
    @error_aware('Propagation distance(s) could not be updated.')
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
        # Refresh screen
        if self._latest_hologram:
            self.send_data(self._latest_hologram)
    
    @error_aware('Camera features could not be updated.')
    @QtCore.pyqtSlot(dict)
    def update_camera_features(self, feature_dict):
        """ 
        Update camera features (e.g. exposure, bit depth) according to a dictionary.
        
        Parameters
        ----------
        feature_dict : dict
        """
        if not self.camera:
            return
        
        for feature, value in feature_dict.items():
            setattr(self.camera, feature, value)
    
    @error_aware('Camera could not be connected.')
    @QtCore.pyqtSlot(object)
    def connect_camera(self, ID):
        """ 
        Connect camera by ID. 
        
        Parameters
        ----------
        ID : str
            String identifier to a camera. If 'debug', a dummy DebugCamera
            instance will be connected.
        """
        # TODO: generalize to other manufacturers
        # This method should never fail. available_cameras() must have been called
        # before so that connecting is always successful.
        if ID == 'debug':
            self.camera = DebugCamera()
        else:
            self.camera = AlliedVisionCamera(ID)
        self.camera_connected_signal.emit(True)
    
    def stop(self):
        """ Stop all reactors. """
        self.reconstruction_reactor.stop()
