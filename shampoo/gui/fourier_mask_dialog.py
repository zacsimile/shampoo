from __future__ import absolute_import

import numpy as np
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from skimage.io import imread
from skimage import img_as_bool

class FourierMaskDialog(QtGui.QDialog):

    fourier_mask_update_signal = QtCore.pyqtSignal(object)

    def __init__(self, initial_mask = None, **kwargs):
        """
        Parameters
        ----------
        initial_mask : ndarray or None, optional
            Dialog will display this array if provided.
        """
        self.mask = None or initial_mask

        super(FourierMaskDialog, self).__init__(**kwargs)
        self.setModal(True)

        clear_prev_mask_btn = QtGui.QPushButton('Clear user-defined mask', self)
        clear_prev_mask_btn.clicked.connect(self.clear_mask)

        file_search_btn = QtGui.QPushButton('Load Fourier mask', self)
        file_search_btn.clicked.connect(self.load_mask)

        accept_btn = QtGui.QPushButton('accept', self)
        accept_btn.clicked.connect(self.accept)

        reject_btn = QtGui.QPushButton('reject', self)
        reject_btn.clicked.connect(self.reject)
        reject_btn.setDefault(True)

        self.viewer = pg.ImageView(parent = self, name = 'Fourier mask')
        if self.mask is not None:
            self.viewer.setImage(self.mask)

        btns = QtGui.QHBoxLayout()
        btns.addWidget(clear_prev_mask_btn)
        btns.addWidget(file_search_btn)
        btns.addWidget(accept_btn)
        btns.addWidget(reject_btn)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.viewer)
        layout.addLayout(btns)
        self.setLayout(layout)

    @QtCore.pyqtSlot()
    def load_mask(self):
        filename = QtGui.QFileDialog.getOpenFileName(self, 'Load Fourier mask')[0]
        self.mask = img_as_bool(imread(filename, as_grey = True))
        self.viewer.setImage(self.mask)
    
    @QtCore.pyqtSlot()
    def clear_mask(self):
        self.mask = None
        self.viewer.clear()
    
    @QtCore.pyqtSlot()
    def accept(self):
        self.fourier_mask_update_signal.emit(self.mask)
        super(FourierMaskDialog, self).accept()