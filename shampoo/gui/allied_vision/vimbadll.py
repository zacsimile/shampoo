# -*- coding: utf-8 -*-
from .vimbastructure import VimbaInterfaceInfo, VimbaCameraInfo, VimbaVersion, VimbaFeatureInfo, VimbaFrame
from .vimbaexception import VimbaException
from sys import platform as sys_plat
import platform
import os
from ctypes import *

def find_win_dll(arch):
    """ 
    Finds the highest versioned windows dll for the specified architecture. 

    Parameters
    ----------
    arch : int, {32, 64}
        Architecture, either 32-bit or 64-bit

    Raises
    ------
    IOError
        If VimbaC.dll cannot be found.
    """
    bases = ['C:\\Program Files\\Allied Vision Technologies\\AVTVimba_%i.%i\\VimbaC\\Bin\\Win%i\\VimbaC.dll',
            'C:\\Program Files\\Allied Vision\\Vimba_%i.%i\\VimbaC\\Bin\Win%i\\VimbaC.dll']
    dlls = []
    for base in bases:
        for major in range(3):
            for minor in range(10):
                candidate = base % (major, minor, arch)
                if os.path.isfile(candidate):
                    dlls.append(candidate)
    if not dlls:
        raise IOError("VimbaC.dll not found.")
    return dlls[-1]

def vimbaC_path():
    """ Returns the path to the Vimba C DLL. """
    if '64' in platform.architecture()[0]:
        return find_win_dll(64)
    else:
        return find_win_dll(32)


# Callback Function Type
if sys_plat == "win32":
    CB_FUNCTYPE = WINFUNCTYPE
else:
    # Untested!
    CB_FUNCTYPE = CFUNCTYPE


class VimbaDLL(object):
    """
    ctypes directives to make the wrapper class work cleanly,
    talks to VimbaC.dll
    """
    # a full list of Vimba API methods
    # (only double dashed methods have been implemented so far)
    #
    # -- VmbVersionQuery()
    #
    # -- VmbStartup()
    # -- VmbShutdown()
    #
    # -- VmbCamerasList()
    # -- VmbCameraInfoQuery()
    # -- VmbCameraOpen()
    # -- VmbCameraClose()
    #
    # -- VmbFeaturesList()
    # -- VmbFeatureInfoQuery()
    # VmbFeatureListAffected()
    # VmbFeatureListSelected()
    # VmbFeatureAccessQuery()
    #
    # -- VmbFeatureIntGet()
    # -- VmbFeatureIntSet()
    # -- VmbFeatureIntRangeQuery()
    # VmbFeatureIntIncrementQuery()
    #
    # -- VmbFeatureFloatGet()
    # -- VmbFeatureFloatSet()
    # -- VmbFeatureFloatRangeQuery()
    #
    # -- VmbFeatureEnumGet()
    # -- VmbFeatureEnumSet()
    # VmbFeatureEnumRangeQuery()
    # VmbFeatureEnumIsAvailable()
    # VmbFeatureEnumAsInt()
    # VmbFeatureEnumAsString()
    # VmbFeatureEnumEntryGet()
    #
    # -- VmbFeatureStringGet()
    # -- VmbFeatureStringSet()
    # VmbFeatureStringMaxlengthQuery()
    #
    # -- VmbFeatureBoolGet()
    # -- VmbFeatureBoolSet()
    #
    # -- VmbFeatureCommandRun()
    # VmbFeatureCommandIsDone()
    #
    # VmbFeatureRawGet()
    # VmbFeatureRawSet()
    # VmbFeatureRawLengthQuery()
    #
    # VmbFeatureInvalidationRegister()
    # VmbFeatureInvalidationUnregister()
    #
    # -- VmbFrameAnnounce()
    # -- VmbFrameRevoke()
    # -- VmbFrameRevokeAll()
    # -- VmbCaptureStart()
    # -- VmbCaptureEnd()
    # -- VmbCaptureFrameQueue()
    # -- VmbCaptureFrameWait()
    # -- VmbCaptureQueueFlush()
    #
    # -- VmbInterfacesList()
    # -- VmbInterfaceOpen()
    # -- VmbInterfaceClose()
    #
    # VmbAncillaryDataOpen()
    # VmbAncillaryDataClose()
    #
    # VmbMemoryRead()
    # VmbMemoryWrite()
    # -- VmbRegistersRead()
    # -- VmbRegistersWrite()

    _instance = None

    def __new__(cls):
        """ Ensure that the DLL is not loaded everytime. """
        if VimbaDLL._instance is None:
            VimbaDLL._instance = super(VimbaDLL, cls).__new__(cls)
        
        return VimbaDLL._instance
        
    def __init__(self):

        self._vimbaDLL = windll.LoadLibrary(vimbaC_path())

        # version query
        self.versionQuery = self._vimbaDLL.VmbVersionQuery
        # returned error code
        self.versionQuery.restype = c_int32
        self.versionQuery.argtypes = (POINTER(VimbaVersion),            # pointer to version structure
                                c_uint32)                                # version structure size

        # startup
        self.startup = self._vimbaDLL.VmbStartup
        # returned error code
        self.startup.restype = c_int32

        # shutdown
        self.shutdown = self._vimbaDLL.VmbShutdown

        # list cameras
        self.camerasList = self._vimbaDLL.VmbCamerasList
        # returned error code
        self.camerasList.restype = c_int32
        self.camerasList.argtypes = (POINTER(VimbaCameraInfo),        # pointer to camera info structure
                                # length of list
                                c_uint32,
                                # pointer to number of cameras
                                POINTER(c_uint32),
                                c_uint32)                                # camera info structure size

        # camera info query
        self.cameraInfoQuery = self._vimbaDLL.VmbCameraInfoQuery
        self.cameraInfoQuery.restype = c_int32
        self.cameraInfoQuery.argtypes = (c_char_p,                            # camera unique id
                                    # pointer to camera info structure
                                    POINTER(VimbaCameraInfo),
                                    c_uint32)                            # size of structure

        # camera open
        self.cameraOpen = self._vimbaDLL.VmbCameraOpen
        # returned error code
        self.cameraOpen.restype = c_int32
        self.cameraOpen.argtypes = (c_char_p,                                # camera unique id
                            # access mode
                            c_uint32,
                            c_void_p)                                # camera handle, pointer to a pointer

        # camera close
        self.cameraClose = self._vimbaDLL.VmbCameraClose
        # returned error code
        self.cameraClose.restype = c_int32
        # camera handle
        self.cameraClose.argtypes = (c_void_p,)

        # list features
        self.featuresList = self._vimbaDLL.VmbFeaturesList
        self.featuresList.restype = c_int32
        self.featuresList.argtypes = (c_void_p,                                # handle, in this case camera handle
                                # pointer to feature info structure
                                POINTER(VimbaFeatureInfo),
                                # list length
                                c_uint32,
                                # pointer to num features found
                                POINTER(c_uint32),
                                c_uint32)                                # feature info size

        # feature info query
        self.featureInfoQuery = self._vimbaDLL.VmbFeatureInfoQuery
        self.featureInfoQuery.restype = c_int32
        self.featureInfoQuery.argtypes = (c_void_p,                            # handle, in this case camera handle
                                    # name of feature
                                    c_char_p,
                                    # pointer to feature info structure
                                    POINTER(VimbaFeatureInfo),
                                    c_uint32)                            # size of structure

        # get the int value of a feature
        self.featureIntGet = self._vimbaDLL.VmbFeatureIntGet
        self.featureIntGet.restype = c_int32
        self.featureIntGet.argtypes = (c_void_p,                                # handle, in this case camera handle
                                # name of the feature
                                c_char_p,
                                POINTER(c_int64))                        # value to get

        # set the int value of a feature
        self.featureIntSet = self._vimbaDLL.VmbFeatureIntSet
        self.featureIntSet.restype = c_int32
        self.featureIntSet.argtypes = (c_void_p,                                # handle, in this case camera handle
                                # name of the feature
                                c_char_p,
                                c_int64)                                # value to set    # get the value of an integer feature

        # query the range of values of the feature
        self.featureIntRangeQuery = self._vimbaDLL.VmbFeatureIntRangeQuery
        self.featureIntRangeQuery.restype = c_int32
        self.featureIntRangeQuery.argtypes = (c_void_p,                        # handle
                                        # name of the feature
                                        c_char_p,
                                        # min range
                                        POINTER(c_int64),
                                        POINTER(c_int64))                # max range

        # get the float value of a feature
        self.featureFloatGet = self._vimbaDLL.VmbFeatureFloatGet
        self.featureFloatGet.restype = c_int32
        self.featureFloatGet.argtypes = (c_void_p,                            # handle, in this case camera handle
                                    # name of the feature
                                    c_char_p,
                                    POINTER(c_double))                    # value to get

        # set the float value of a feature
        self.featureFloatSet = self._vimbaDLL.VmbFeatureFloatSet
        self.featureFloatSet.restype = c_int32
        self.featureFloatSet.argtypes = (c_void_p,                            # handle, in this case camera handle
                                    # name of the feature
                                    c_char_p,
                                    c_double)                            # value to set

        # query the range of values of the feature
        self.featureFloatRangeQuery = self._vimbaDLL.VmbFeatureFloatRangeQuery
        self.featureFloatRangeQuery.restype = c_int32
        self.featureFloatRangeQuery.argtypes = (c_void_p,                    # handle
                                        # name of the feature
                                        c_char_p,
                                        # min range
                                        POINTER(c_double),
                                        POINTER(c_double))            # max range

        # get the enum value of a feature
        self.featureEnumGet = self._vimbaDLL.VmbFeatureEnumGet
        self.featureEnumGet.restype = c_int32
        self.featureEnumGet.argtypes = (c_void_p,                            # handle, in this case camera handle
                                # name of the feature
                                c_char_p,
                                POINTER(c_char_p))                    # value to get

        # set the enum value of a feature
        self.featureEnumSet = self._vimbaDLL.VmbFeatureEnumSet
        self.featureEnumSet.restype = c_int32
        self.featureEnumSet.argtypes = (c_void_p,                            # handle, in this case camera handle
                                # name of the feature
                                c_char_p,
                                c_char_p)                            # value to set

        # get the string value of a feature
        self.featureStringGet = self._vimbaDLL.VmbFeatureStringGet
        self.featureStringGet.restype = c_int32
        self.featureStringGet.argtypes = (c_void_p,                            # handle, in this case camera handle
                                    # name of the feature
                                    c_char_p,
                                    # string buffer to fill
                                    c_char_p,
                                    # size of the input buffer
                                    c_uint32,
                                    POINTER(c_uint32))                    # string buffer to fill

        # set the string value of a feature
        self.featureStringSet = self._vimbaDLL.VmbFeatureStringSet
        self.featureStringSet.restype = c_int32
        self.featureStringSet.argtypes = (c_void_p,                            # handle, in this case camera handle
                                    # name of the feature
                                    c_char_p,
                                    c_char_p)                            # value to set

        # get the boolean value of a feature
        self.featureBoolGet = self._vimbaDLL.VmbFeatureBoolGet
        self.featureBoolGet.restype = c_int32
        self.featureBoolGet.argtypes = (c_void_p,                            # handle, in this case camera handle
                                # name of the feature
                                c_char_p,
                                POINTER(c_bool))                        # value to get

        # set the boolean value of a feature
        self.featureBoolSet = self._vimbaDLL.VmbFeatureBoolSet
        self.featureBoolSet.restype = c_int32
        self.featureBoolSet.argtypes = (c_void_p,                            # handle, in this case camera handle
                                # name of the feature
                                c_char_p,
                                c_bool)                                # value to set

        # run a feature command
        self.featureCommandRun = self._vimbaDLL.VmbFeatureCommandRun
        self.featureCommandRun.restype = c_int32
        self.featureCommandRun.argtypes = (c_void_p,                            # handle for a module that exposes features
                                    c_char_p)                            # name of the command feature

        # announce frames to the API that may be queued for frame capturing later
        self.frameAnnounce = self._vimbaDLL.VmbFrameAnnounce
        self.frameAnnounce.restype = c_int32
        self.frameAnnounce.argtypes = (c_void_p,                                # camera handle
                                # pointer to frame
                                POINTER(VimbaFrame),
                                c_uint32)                                # size of frame

        # callback for frame queue
        self.frameDoneCallback = CB_FUNCTYPE(c_void_p,                     # Return Type
                                        c_void_p,                     # Camera Hanlde
                                        POINTER(VimbaFrame))  # Pointer to frame

        # revoke a frame from the API
        self.frameRevoke = self._vimbaDLL.VmbFrameRevoke
        self.frameRevoke.restype = c_int32
        self.frameRevoke.argtypes = (c_void_p,                                # camera handle
                                POINTER(VimbaFrame))            # pointer to frame

        # revoke all frames assigned to a certain camera
        self.frameRevokeAll = self._vimbaDLL.VmbFrameRevokeAll
        self.frameRevokeAll.restype = c_int32
        # camera handle
        self.frameRevokeAll.argtypes = (c_void_p,)

        # prepare the API for incoming frames
        self.captureStart = self._vimbaDLL.VmbCaptureStart
        self.captureStart.restype = c_int32
        # camera handle
        self.captureStart.argtypes = (c_void_p,)

        # stop the API from being able to receive frames
        self.captureEnd = self._vimbaDLL.VmbCaptureEnd
        self.captureEnd.restype = c_int32
        # camera handle
        self.captureEnd.argtypes = (c_void_p,)

        # queue frames that may be filled during frame capturing
        self.captureFrameQueue = self._vimbaDLL.VmbCaptureFrameQueue
        self.captureFrameQueue.restype = c_int32
        self.captureFrameQueue.argtypes = (c_void_p,
                                    POINTER(VimbaFrame),
                                    c_void_p)                            # callback

        # wait for a queued frame to be filled (or dequeued)
        self.captureFrameWait = self._vimbaDLL.VmbCaptureFrameWait
        self.captureFrameWait.restype = c_int32
        self.captureFrameWait.argtypes = (c_void_p,                            # camera handle
                                    POINTER(VimbaFrame),
                                    c_uint32)                            # timeout

        # flush the capture queue
        self.captureQueueFlush = self._vimbaDLL.VmbCaptureQueueFlush
        self.captureQueueFlush.restype = c_int32
        # camera handle
        self.captureQueueFlush.argtypes = (c_void_p,)

        # list interfaces
        self.interfacesList = self._vimbaDLL.VmbInterfacesList
        self.interfacesList.restype = c_int32
        self.interfacesList.argtypes = (POINTER(VimbaInterfaceInfo),        # pointer to interface info structure
                                # length of list
                                c_uint32,
                                # pointer to number of interfaces
                                POINTER(c_uint32),
                                c_uint32)

        # open interface
        self.interfaceOpen = self._vimbaDLL.VmbInterfaceOpen
        self.interfaceOpen.restype = c_int32
        self.interfaceOpen.argtypes = (c_char_p,                                # unique id
                                c_void_p)                                # handle

        # close interface
        self.interfaceClose = self._vimbaDLL.VmbInterfaceClose
        self.interfaceClose.restype = c_int32
        self.interfaceClose.argtypes = (c_void_p,)                            # handle

        # read from register
        self.registersRead = self._vimbaDLL.VmbRegistersRead
        self.registersRead.restype = c_int32
        self.registersRead.argtypes = (c_void_p,                                # handle
                                # read count
                                c_uint32,
                                # pointer to address array
                                POINTER(c_uint64),
                                # pointer to data array
                                POINTER(c_uint64),
                                POINTER(c_uint32))                    # pointer to num complete reads

        # write to register
        self.registersWrite = self._vimbaDLL.VmbRegistersWrite
        self.registersWrite.restype = c_int32
        self.registersWrite.argtypes = (c_void_p,                            # handle
                                # write count
                                c_uint32,
                                # pointer to address array
                                POINTER(c_uint64),
                                # pointer to data array
                                POINTER(c_uint64),
                                POINTER(c_uint32))                    # pointer to num complete write


class VimbaC_MemoryBlock(object):

    """
    Just a memory block object for dealing
    neatly with C memory allocations.
    """

    @property
    def block(self):
        return self._block

    def __init__(self, blockSize):

        self._crtDLL = cdll.LoadLibrary('msvcrt') # C runtime DLL
        
        # assign memory block
        malloc = self._crtDLL.malloc
        malloc.argtypes = (c_size_t,)
        malloc.restype = c_void_p
        self._block = malloc(blockSize)        # todo check for NULL on failure

        # this seems to be None if too much memory is requested
        if self._block is None:
            raise VimbaException(-51)

    def __del__(self):

        # free memory block
        free = self._crtDLL.free
        free.argtypes = (c_void_p,)
        free.restype = None
        free(self._block)
