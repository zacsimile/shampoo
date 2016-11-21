
from multiprocessing import Queue
import numpy as np
import os
import pyqtgraph as pg
from pyqtgraph import QtCore, QtGui
from ..reconstruction import Hologram, ReconstructedWave

ICONS_FOLDER = os.path.join(os.path.dirname(__file__), 'icons')
DEFAULT_PROPAGATION_DISTANCE = 0.03658

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

    def _init_ui(self): pass

    def _init_actions(self): pass

    def _connect_signals(self): pass

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

class CameraFeatureDialog(ShampooWidget, QtGui.QDialog):
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
        super(CameraFeatureDialog, self).__init__(parent = self)

        self.setModal(True)
    
    @QtCore.pyqtSlot()
    def accept(self):
        feature_dict = {'exposure': int(self.exposure_edit.text()), 'bit_depth': int(self.bit_depth_edit.text())}
        self.camera_features_update_signal.emit(feature_dict)
        super(CameraFeatureDialog, self).accept()
    
    @QtCore.pyqtSlot()
    def reject(self):
        super(CameraFeatureDialog, self).reject()
    
    def _init_ui(self):
        
        self.save_btn = QtGui.QPushButton('Update values', self)
        self.cancel_btn = QtGui.QPushButton('Cancel changes', self)
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
    
    def _init_actions(self):
        pass
    
    def _connect_signals(self):
        self.save_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

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

class FourierPlaneViewer(Viewer, QtGui.QWidget):
    """
    QWidget displaying the raw holograms, before reconstruction.

    Slots
    -----
    display
        Display holographic data in the Fourier plane.
    """

    def __init__(self, parent, **kwargs):
        """
        Parameters
        ----------
        parent : QObject
        """
        super(FourierPlaneViewer, self).__init__(parent = parent, **kwargs)
        self.viewer = None
    
    @QtCore.pyqtSlot(object)
    def display(self, item):
        """
        Displays a shampoo.Hologram or NumPy array.

        Parameters
        ----------
        item : ndarray
        """
        # FourierPlaneViewer might be hidden most of the time. No point in updating the image
        # in this case
        if self.isVisible():
            self.viewer.setImage(item)

    def _init_ui(self):
        self.viewer = pg.ImageView(parent = self, name = 'Reconstructed amplitude')
        
        layout = QtGui.QVBoxLayout()
        layout.addWidget(QtGui.QLabel('Fourier plane', parent = self))
        layout.addWidget(self.viewer)
        self.setLayout(layout)

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
            propagation_distance = np.arange(start = start, stop = stop + step, step = step)
        
        self.propagation_distance_signal.emit(propagation_distance)

    def _init_ui(self):
        
        # Widgets
        self.start_value_widget = QtGui.QLineEdit(parent = self)
        self.stop_value_widget = QtGui.QLineEdit(parent = self)
        self.step_value_widget = QtGui.QLineEdit(parent = self)

        # Initial values
        self.start_value_widget.setText(str(DEFAULT_PROPAGATION_DISTANCE))
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