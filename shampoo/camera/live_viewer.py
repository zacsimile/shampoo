"""
This module implements a PyQtGraph widget to display a live
feed from a camera.
"""
import pyqtgraph as pg
from pyqtgraph import QtGui, QtCore
from .camera import AlliedVisionCamera
from ..gui.reactor import Reactor, ThreadSafeQueue

class LiveViewer(QtGui.QMainWindow):

    image_signal = QtCore.pyqtSignal(object, name = 'image_signal')

    def __init__(self):
        self.image_viewer = pg.ImageView(parent = self)
        self.camera = AlliedVisionCamera
        self.feed_queue = ThreadSafeQueue()
        self.display_reactor = Reactor(input_queue = self.feed_queue, callback = self.image_signal.emit)
        self.display_reactor.start()
        self.camera.start_acquisition(image_queue = self.feed_queue)

        super(LiveViewer, self).__init__()
    
    @QtCore.pyqtSlot(object)
    def display(self, image):
        self.image_viewer.setImage(image)
    
    def _init_ui(self):       
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.image_viewer)
        self.setLayout(layout)

    def _connect_signals(self):
        self.image_signal.connect(self.display)

def run():   
    app = QtGui.QApplication(sys.argv)
    app.setStyle(QtGui.QStyleFactory.create('cde'))
    gui = LiveViewer()
    
    sys.exit(app.exec_())