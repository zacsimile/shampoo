# -*- coding: utf-8 -*-
from __future __ import absolute_import

from .vimbastructure import VimbaInterfaceInfo, VimbaCameraInfo, VimbaVersion
from .vimbaobject import VimbaObject
from .vimbaexception import VimbaException
from .vimbaframe import VimbaFrame
from .vimbadll import VimbaDLL
from ctypes import *
# camera features are automatically readable as object attributes.


class VimbaCamera(VimbaObject):

    """
    A Vimba camera object. This class provides the minimal access
    to Vimba functions required to control the camera.
    """

    @property
    def cameraIdString(self):
        return self._cameraIdString

    # own handle is inherited as self._handle
    def __init__(self, cameraIdString):

        # call super constructor
        super(VimbaCamera, self).__init__()

        # set ID
        self._cameraIdString = cameraIdString

        # set own info
        self._info = self._getInfo()

        self._api = VimbaDLL()

    def getInfo(self):
        """
        Get info of the camera. Does not require
        the camera to be opened.

        :returns: VimbaCameraInfo object -- camera information.
        """
        return self._info

    def _getInfo(self):
        """
        Get info of the camera. Does not require
        the camera to be opened.

        :returns: VimbaCameraInfo object -- camera information.
        """
        # args for Vimba call
        cameraInfo = VimbaCameraInfo()

        # Vimba DLL will return an error code
        errorCode = self._api.cameraInfoQuery(self._cameraIdString,
                                             byref(cameraInfo),
                                             sizeof(cameraInfo))
        if errorCode != 0:
            raise VimbaException(errorCode)

        return cameraInfo

    def openCamera(self):
        """
        Open the camera.
        """
        # args for Vimba call
        cameraAccessMode = 1  # full access (see VmbAccessModeType)

        errorCode = self._api.cameraOpen(self._cameraIdString,
                                        cameraAccessMode,
                                        byref(self._handle))
        if errorCode != 0:
            raise VimbaException(errorCode)

    def closeCamera(self):
        """
        Close the camera.
        """
        errorCode = self._api.cameraClose(self._handle)
        if errorCode != 0:
            raise VimbaException(errorCode)

    def revokeAllFrames(self):
        """
        Revoke all frames assigned to the camera.
        """
        errorCode = self._api.frameRevokeAll(self._handle)
        if errorCode != 0:
            raise VimbaException(errorCode)

    def startCapture(self):
        """
        Prepare the API for incoming frames.
        """
        errorCode = self._api.captureStart(self._handle)
        if errorCode != 0:
            raise VimbaException(errorCode)

    def endCapture(self):
        """
        Stop the API from being able to receive frames.
        """
        errorCode = self._api.captureEnd(self._handle)
        if errorCode != 0:
            raise VimbaException(errorCode)

    def flushCaptureQueue(self):
        """
        Flush the capture queue.
        """
        errorCode = self._api.captureQueueFlush(self._handle)
        if errorCode != 0:
            raise VimbaException(errorCode)

    # method for easy frame creation
    def getFrame(self):
        """
        Creates and returns a new frame object. Multiple frames
        per camera can therefore be returned.

        :returns: VimbaFrame object -- the new frame.
        """
        return VimbaFrame(self)
