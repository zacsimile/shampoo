
from multiprocessing import Queue
import numpy as np
import os
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from .reactor import Reactor
from ..reconstruction import Hologram, ReconstructedWave

ICONS_FOLDER = os.path.join(os.path.dirname(__file__), 'icons')

#########################################################################################
###             TEMPLATE CLASSES

class ShampooWidget(object):
    """ 
    Template class for SHAMPOO's GUI components. 
    
    Subclasses must minimally override _init_ui(), _init_actions() and _connect_signals().
    """

    def __init__(self, *args, **kwargs):
        super(ShampooWidget, self).__init__()

        self._init_ui()
        self._init_actions()
        self._connect_signals()

    def _init_ui(self):
        raise NotImplementedError

    def _init_actions(self):
        raise NotImplementedError

    def _connect_signals(self):
        raise NotImplementedError

class Viewer(ShampooWidget):
    """
    Template class for SHAMPOO widgets that display image data.

    Subclasses must minimally override display() and clear().
    """
    def __init__(self, parent, **kwargs):
        super(Viewer, self).__init__(parent = parent, **kwargs)
    
    @QtCore.pyqtSlot(object)
    def display(self, item):
        raise NotImplementedError
    
    @QtCore.pyqtSlot()
    def clear(self):
        raise NotImplementedError

#########################################################################################
###             GUI COMPONENTS

class DataViewer(Viewer, pg.ImageView):
    """
    QWidget displaying the raw holograms, before reconstruction.

    Methods
    -------
    display
    """

    def __init__(self, parent, **kwargs):
        """
        Parameters
        ----------
        parent : QObject
        """
        super(DataViewer, self).__init__(parent = parent, **kwargs)
    
    @QtCore.pyqtSlot(object)
    def display(self, item):
        """
        Displays a shampoo.Hologram or NumPy array.

        Parameters
        ----------
        item : ndarray or shampoo.Hologram
        """
        if isinstance(item, Hologram):
            item = item.hologram
        self.setImage(item)
    
    @QtCore.pyqtSlot()
    def clear(self):
        self.clear()
    
    def _init_ui(self): pass

    def _init_actions(self): pass
    
    def _connect_signals(self):pass

class ReconstructedHologramViewer(Viewer, QtGui.QWidget):
    """
    QWidget displaying the reconstructed wave holographic data, in two plots: phase and amplitude.

    Methods
    -------
    display
    """
    # A PyQt Signal is required to transfer data between threads
    # Therefore, the display reactor must emit a signal to be able to plot ano image
    _internal_display_signal = QtCore.pyqtSignal(object, name = 'display_signal')

    def __init__(self, parent, **kwargs):
        """
        Parameters
        ----------
        parent : QObject
        """
        self.amplitude_viewer = None
        self.phase_viewer = None
        
        super(ReconstructedHologramViewer, self).__init__(parent = parent, **kwargs)
    
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

    @QtCore.pyqtSlot(object)
    def _display_from_queue(self, data_tup):
        """ Display a propagation_distance, reconstructed_hologram pair. """

    
    @QtCore.pyqtSlot()
    def clear(self):
        self.amplitude_viewer.clear()
        self.phase_viewer.clear()

    ### Boilerplate ###

    def _init_ui(self):

        self.amplitude_viewer = pg.ImageView(parent = self, name = 'Reconstructed amplitude')
        self.phase_viewer = pg.ImageView(parent = self, name = 'Reconstructed phase')

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
    
    def _init_actions(self):
        pass

    def _connect_signals(self):
        self._internal_display_signal.connect(self._display_from_queue)

class PropagationDistanceSelector(ShampooWidget, QtGui.QWidget):
    """
    QWidget allowing the user to select an array of propagation distances

    Methods
    -------
    update_propagation_distance
    """
    propagation_distance_signal = QtCore.pyqtSignal(object, name = 'propagation_distance')

    def __init__(self, parent):
        super(PropagationDistanceSelector, self).__init__(parent)
    
    @QtCore.pyqtSlot()
    def update_propagation_distance(self):
        """
        Emits the propagation_distance signal with the sorted propagation distance data parsed from the widget.
        """
        propagation_distance = list()
        for item_row in range(0, self.table.rowCount()):
            content = self.table.item(item_row, 0)     #QTableWidgetItem instance or None
            if content is None:
                content = 0.0
            else:
                content = content.text()
            
            # Content might not be castable as a float. In this case, do not append.
            try:
                propagation_distance.append( float(content) )
            except:
                pass
        
        # Remove duplicates and sort
        propagation_distance = list(set(propagation_distance))
        propagation_distance.sort()
        self.propagation_distance_signal.emit(propagation_distance)
    
    @QtCore.pyqtSlot()
    def _add_propagation_distance_cell(self):
        """ Collection of operations that happen when inserting a row in the table. """
        # No need to update_propagation_distance, as an empty cell is inserted; the user
        # will have to specify a value for the cell, which will trigger update_propagation_distance
        self.table.insertRow(self.table.rowCount())
    
    @QtCore.pyqtSlot()
    def _remove_propagation_distance_cell(self):
        """ Collection of operations that happen when removing a row in the table """
        self.table.removeRow(self.table.rowCount() - 1)
        self.update_propagation_distance() 
    
    def _init_ui(self):
        
        # Widgets
        self.table = QtGui.QTableWidget(1, 1, parent = self)
        self.add_btn = QtGui.QPushButton(QtGui.QIcon(os.path.join(ICONS_FOLDER, 'add.png')), '', self)
        self.remove_btn = QtGui.QPushButton(QtGui.QIcon(os.path.join(ICONS_FOLDER, 'remove.png')), '', self)

        # Layouts
        # Fix width to header
        self.table.setHorizontalHeaderLabels(['Propagation \n distances \n (m)'])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table_layout = QtGui.QVBoxLayout()
        self.table_layout.addWidget(self.table)

        self.btn_layout = QtGui.QHBoxLayout()
        self.btn_layout.addWidget(self.add_btn)
        self.btn_layout.addWidget(self.remove_btn)

        # Final layout
        self.layout = QtGui.QVBoxLayout()
        self.layout.addLayout(self.table_layout)
        self.layout.addLayout(self.btn_layout)
        self.setLayout(self.layout)
    
    def _init_actions(self):
        pass
    
    def _connect_signals(self):
        # Buttons add or remove rows
        self.add_btn.clicked.connect(self._add_propagation_distance_cell)
        self.remove_btn.clicked.connect(self._remove_propagation_distance_cell)
    
        # Updating the table updates the propagation distance
        self.table.itemChanged.connect(self.update_propagation_distance)