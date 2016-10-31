
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

    Slots
    -----
    display
        Display raw holographic data.
    
    clear
        Clear view.
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

    Slots
    -----
    display
        Display the phase and amplitude information of a reconstructed hologram.
    
    clear
        Clear view.
    """

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
        pass

class PropagationDistanceSelector(ShampooWidget, QtGui.QWidget):
    """
    QWidget allowing the user to select an array of propagation distances

    Signals
    -------
    propagation_distance_signal
        Emitted when propagation distances are modified.
    """
    propagation_distance_signal = QtCore.pyqtSignal(object, name = 'propagation_distance')

    def __init__(self, parent):

        self.start_value_widget = None
        self.stop_value_widget = None
        self.step_value_widget = None

        super(PropagationDistanceSelector, self).__init__(parent)
    
    @QtCore.pyqtSlot()
    def _update_propagation_distance(self):
        """ Emits the propagation_distance signal with the sorted propagation distance data parsed from the widget. """

        start, stop, step = [float(widget.text()) for widget in (self.start_value_widget, self.stop_value_widget, self.step_value_widget)]

        if step == 0:
            propagation_distance = [start]
        else:
            # Singe arange does not include the 'stop' value, add another step
            propagation_distance = n.arange(start = start, stop = stop + step, step = step)
        
        self.propagation_distance_signal.emit(propagation_distance)

    def _init_ui(self):
        
        # Widgets
        self.start_value_widget = QtGui.QLineEdit(parent = self)
        self.stop_value_widget = QtGui.QLineEdit(parent = self)
        self.step_value_widget = QtGui.QLineEdit(parent = self)

        # Initial values
        self.start_value_widget.setText('0.0')
        self.stop_value_widget.setText('0.0')
        self.step_value_widget.setText('0.0')


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
    
    def _init_actions(self):
        pass
    
    def _connect_signals(self):
        self.start_value_widget.editingFinished.connect(self._update_propagation_distance)
        self.stop_value_widget.editingFinished.connect(self._update_propagation_distance)
        self.step_value_widget.editingFinished.connect(self._update_propagation_distance)