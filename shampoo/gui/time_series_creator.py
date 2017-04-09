from __future__ import absolute_import

from collections import Iterable
import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore

from ..reconstruction import Hologram
from ..time_series import TimeSeries

class TimeSeriesCreator(QtGui.QDialog):

    time_series_path_signal = QtCore.pyqtSignal(str)
    _assembly_update_signal = QtCore.pyqtSignal(int)

    def __init__(self, **kwargs):
        self.holograms = list()

        super(TimeSeriesCreator, self).__init__(**kwargs)
        self.setModal(True)
        self.setWindowTitle('Create hologram time series')

        # Create reorderable list
        # Example from http://www.walletfox.com/course/qtreorderablelist.php
        self.hologram_table = QtGui.QListWidget(parent = self)
        self.hologram_table.setDragDropMode(QtGui.QAbstractItemView.InternalMove)
        
        self.assembly_progress_bar = QtGui.QProgressBar(parent = self)
        self.assembly_progress_bar.setRange(0, 100)
        self._assembly_update_signal.connect(self.assembly_progress_bar.setValue)
        self.assembly_progress_bar.hide()

        file_search_btn = QtGui.QPushButton('Add hologram', self)
        file_search_btn.clicked.connect(self.add_hologram_file)

        clear_btn = QtGui.QPushButton('Clear holograms', self)
        clear_btn.clicked.connect(self.clear)

        accept_btn = QtGui.QPushButton('Create', self)
        accept_btn.clicked.connect(self.accept)

        reject_btn = QtGui.QPushButton('Cancel', self)
        reject_btn.clicked.connect(self.reject)
        reject_btn.setDefault(True)

        btns = QtGui.QHBoxLayout()
        btns.addWidget(file_search_btn)
        btns.addWidget(clear_btn)
        btns.addWidget(accept_btn)
        btns.addWidget(reject_btn)

        explanation = QtGui.QLabel('Add holograms and order them by drag-and-drop')
        explanation.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(explanation)
        layout.addWidget(self.hologram_table)
        layout.addWidget(self.assembly_progress_bar)
        layout.addLayout(btns)
        self.setLayout(layout)
    
    @QtCore.pyqtSlot()
    def clear(self):
        self.holograms.clear()
        self.hologram_table.clear()
    
    @QtCore.pyqtSlot()
    def add_hologram_file(self):
        """ 
        Add hologram to the holograms dictionary by filepath. 

        Parameters
        ----------
        path : str or iterable of strings
            Path or paths to a hologram.
        """
        paths = QtGui.QFileDialog.getOpenFileNames(self, caption = 'Select one of more holograms', 
                                                   filter = "Images (*.tif)")[0]
        self.hologram_table.addItems(paths)
        self.holograms.extend(paths)
        
    @QtCore.pyqtSlot()
    def accept(self):

        filename = QtGui.QFileDialog.getSaveFileName(self, caption = 'Save time series', 
                                                   filter = "HDF5 (*.hdf5 *.h5)")[0]

        self.assembly_progress_bar.show()
        self._assembly_update_signal.emit(0)

        t = TimeSeries(filename = filename, mode = 'w')
        for index, path in enumerate(self.holograms):
            # TODO: record wavelength somehow
            # TODO: choose time-points instead of index
            holo = Hologram.from_tif(path, wavelength = 800)
            t.add_hologram(holo, time_point = index)
            self._assembly_update_signal.emit(int(100*index / len(self.holograms)))
        self._assembly_update_signal.emit(100)
        
        self.time_series_path_signal.emit(filename)
        self.assembly_progress_bar.hide()

        super(TimeSeriesCreator, self).accept()