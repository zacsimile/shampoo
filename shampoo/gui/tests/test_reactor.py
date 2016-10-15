
from shampoo.gui.reactor import Reactor, ThreadSafeQueue
from time import sleep
import unittest

class TestReactor(unittest.TestCase):
    
    item = 'foobar'

    def setUp(self):
        self.out_queue = ThreadSafeQueue()

    def test_trivial_reactor(self):
        """ 
        Tests that the trivial reactor -- which takes inputs and places them in the
        output queue -- works as intended. 
        """
        reactor = Reactor(out_queue = self.out_queue, function = None)
        reactor.start()
        reactor.send_item(self.item)
        self.assertEqual(self.out_queue.get(), self.item)
    
    def test_reactor_startup(self):
        """ Tests whether the event loop is running right after start(). """
        reactor = Reactor(out_queue = self.out_queue, function = None)
        reactor.start()
        self.assertTrue(reactor.is_alive())

if __name__ == '__main__':
    unittest.main()