# -*- coding: utf-8 -*-
from .vimbaobject import VimbaObject
from .vimbaexception import VimbaException
from ctypes import *
# interface features are automatically readable as object attributes.


class VimbaInterface(VimbaObject):

    """
    A Vimba interface object. This class provides the minimal access
    to Vimba functions required to control the interface.
    """

    @property
    def interfaceIdString(self):
        return self._interfaceIdString

    # own handle is inherited as self._handle
    def __init__(self, interfaceIdString):

        # call super constructor
        # self._api = VimbaDLL() happens in VimbaObject
        super(VimbaInterface, self).__init__()

        # set ID
        self._interfaceIdString = interfaceIdString

    def openInterface(self):
        """
        Open the interface.
        """
        errorCode = self._api.interfaceOpen(self._interfaceIdString,
                                           byref(self._handle))
        if errorCode != 0:
            raise VimbaException(errorCode)

    def closeInterface(self):
        """
        Close the interface.
        """
        errorCode = self._api.interfaceClose(self._handle)
        if errorCode != 0:
            raise VimbaException(errorCode)
