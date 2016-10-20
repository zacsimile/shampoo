
from shampoo.gui.reactor import Reactor, ProcessReactor, ThreadSafeQueue, ProcessSafeQueue
from time import sleep
import unittest

class TestThreadReactor(unittest.TestCase):
    
    item = 'foobar'

    def setUp(self):
        self.out_queue = ThreadSafeQueue()

    def test_trivial_reactor(self):
        """ 
        Tests that the trivial reactor -- which takes inputs and places them in the
        output queue -- works as intended. 
        """
        reactor = Reactor(output_queue = self.out_queue, function = None)
        reactor.start()
        reactor.send_item(self.item)
        self.assertEqual(self.out_queue.get(), self.item)
    
    def test_reactor_startup(self):
        """ Tests whether the event loop is running right after start(). """
        reactor = Reactor(output_queue = self.out_queue, function = None)
        reactor.start()
        self.assertTrue(reactor.is_alive())
    
    def test_reactor_start_and_stop(self):
        """ Tests whether the reactor can start, stop, and start again """
        reactor = Reactor(output_queue = self.out_queue)
        reactor.start()
        self.assertTrue(reactor.is_alive())
        reactor.stop()
        self.assertFalse(reactor.is_alive())
        reactor.start()
        self.assertTrue(reactor.is_alive())

class TestProcessReactor(TestThreadReactor):

    item = 10

    def setUp(self):
        self.out_queue = ProcessSafeQueue()
    
    def test_trivial_reactor(self):
        """ 
        Tests that the trivial reactor -- which takes inputs and places them in the
        output queue -- works as intended. 
        """
        reactor = ProcessReactor(output_queue = self.out_queue, function = None)
        reactor.start()
        reactor.send_item(self.item)
        self.assertEqual(self.out_queue.get(), self.item)
    
    def test_reactor_startup(self):
        """ Tests whether the event loop is running right after start(). """
        reactor = ProcessReactor(output_queue = self.out_queue, function = None)
        reactor.start()
        self.assertTrue(reactor.is_alive())
    
    def test_reactor_start_and_stop(self):
        """ Tests whether the reactor can start, stop, and start again """
        reactor = ProcessReactor(output_queue = self.out_queue)
        reactor.start()
        self.assertTrue(reactor.is_alive())
        reactor.stop()
        self.assertFalse(reactor.is_alive())
        reactor.start()
        self.assertTrue(reactor.is_alive())
    


if __name__ == '__main__':
    unittest.main()