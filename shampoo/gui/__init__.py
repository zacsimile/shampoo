from __future__ import absolute_import

from functools import wraps
import os
import traceback

os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

def error_aware(message):
    """
    Wrap an instance method with a try/except.
    Instance must have a signal called 'error_message_signal' which
    will be emitted with the message upon error. 
    """
    def wrap(func):
        @wraps(func)
        def aware_func(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except: 
                exc = traceback.format_exc()
                self.error_message_signal.emit(message + '\n \n \n' + exc)
        return aware_func
    return wrap

from .gui import run