from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
                        
from ..gui.reactor import Reactor, ProcessReactor, ThreadSafeQueue, ProcessSafeQueue

# TODO: test multiple items
ITEM = 'foobar'

def test_trivial_thread_reactor():
    """ 
    Tests that the trivial reactor -- which takes inputs and places them in the
    output queue -- works as intended. 
    """
    out_queue = ThreadSafeQueue()
    reactor = Reactor(output_queue = out_queue, function = None)
    reactor.start()
    reactor.send_item(ITEM)
    assert out_queue.get() == ITEM

def test_thread_reactor_startup():
    """ Tests whether the event loop is running right after start(). """
    out_queue = ThreadSafeQueue()
    reactor = Reactor(output_queue = out_queue, function = None)
    reactor.start()
    assert reactor.is_alive()

def test_thread_reactor_start_and_stop():
    """ Tests whether the reactor can start, stop, and start again """
    out_queue = ThreadSafeQueue()
    reactor = Reactor(output_queue = out_queue)
    reactor.start()
    assert reactor.is_alive()
    reactor.stop()
    assert not reactor.is_alive()
    reactor.start()
    assert reactor.is_alive()

def test_trivial_process_reactor():
    """ 
    Tests that the trivial reactor -- which takes inputs and places them in the
    output queue -- works as intended. 
    """
    out_queue = ProcessSafeQueue()
    reactor = ProcessReactor(output_queue = out_queue, function = None)
    reactor.start()
    reactor.send_item(ITEM)
    assert out_queue.get() == ITEM

def test_process_reactor_startup():
    """ Tests whether the event loop is running right after start(). """
    out_queue = ProcessSafeQueue()
    reactor = ProcessReactor(output_queue = out_queue, function = None)
    reactor.start()
    assert reactor.is_alive()

def test_process_reactor_start_and_stop():
    """ Tests whether the reactor can start, stop, and start again """
    out_queue = ProcessSafeQueue()
    reactor = ProcessReactor(output_queue = out_queue)
    reactor.start()
    assert reactor.is_alive()
    reactor.stop()
    assert not reactor.is_alive()
    reactor.start()
    assert reactor.is_alive()