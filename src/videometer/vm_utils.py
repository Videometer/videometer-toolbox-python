"""
Common utilities for Videometer image processing.
This module provides shared helper functions and acts as a dispatcher for 
CLR-specific utilities when the 'clr' backend is active.
"""

import numpy as np
import numbers
from videometer import config

def checkIfbandIndexesToUseIsValid(bandIndexesToUse, nBandsInImageClass):
    """Checks if the values in bandIndexesToUse are valid for the given image.

    Args:
        bandIndexesToUse (list or np.ndarray): List of band indexes to validate.
        nBandsInImageClass (int): Total number of bands in the image.

    Raises:
        TypeError: If bandIndexesToUse is not a list or contains non-integers.
        ValueError: If an index is out of range.
    """

    if not (isinstance(bandIndexesToUse, (list, np.ndarray))):
        raise TypeError("bandIndexesToUse should be a list or numpy array")

    for i in range(len(bandIndexesToUse)):
        if (
            not (isinstance(bandIndexesToUse[i], numbers.Integral))
            or bandIndexesToUse[i] < 0
        ):
            raise TypeError(
                f"Element at index {i} (value={bandIndexesToUse[i]}) "
                "is not a positive integer in bandIndexesToUse"
            )

        if bandIndexesToUse[i] >= nBandsInImageClass:
            raise ValueError(
                f"Element at index {i} (value={bandIndexesToUse[i]}) "
                f"is out of range for image with {nBandsInImageClass} bands"
            )

# --- CLR Dispatchers ---
# These functions lazily import vm_utils_clr to avoid loading pythonnet 
# unless specifically requested.

def imageLayer2npArray(imageLayer):
    """Converts a CLR ImageLayer to a NumPy array. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.imageLayer2npArray(imageLayer)

def get_IlluminationLUT():
    """Returns a lookup table for illumination types. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.get_IlluminationLUT()

def illuminationObjects2List(illumnationObjects):
    """Converts CLR illumination objects to a list of names. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.illuminationObjects2List(illumnationObjects)

def illuminationList2Objects(illuminationList):
    """Converts a list of names to CLR illumination objects. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.illuminationList2Objects(illuminationList)

def asNumpyArray(netArray):
    """Converts a CLR array to a NumPy array. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.asNumpyArray(netArray)

def addAllAvailableImageLayers(VMImageObject, ImageClass):
    """Adds all metadata layers from ImageClass to a VMImage object. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.addAllAvailableImageLayers(VMImageObject, ImageClass)

def addImageLayer(VMImageObject, npArray, typeOfLayer):
    """Adds a specific NumPy layer to a VMImage object. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.addImageLayer(VMImageObject, npArray, typeOfLayer)

def setFreehandLayers(VMImageObject, ImageClass):
    """Serializes Freehand layers to the VMImage object. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.setFreehandLayers(VMImageObject, ImageClass)

def vmImage2npArray(vmImage):
    """Converts a CLR VMImage object to a 3-D NumPy array. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.vmImage2npArray(vmImage)

def asNetArrayMemMove(npArray):
    """Converts a NumPy array to a CLR array using memory move. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.asNetArrayMemMove(npArray)

def npArray2VMImage(npArray):
    """Converts a 3-D NumPy array to a CLR VMImage object. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.npArray2VMImage(npArray)

def get_SpectraNamesLUP():
    """Returns a lookup table for spectra names. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.get_SpectraNamesLUP()

def systemDrawingBitmap2npArray(bitmap):
    """Converts a System.Drawing.Bitmap to a NumPy array. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.systemDrawingBitmap2npArray(bitmap)

def get_CompressionAndQuantificationPresetLUT():
    """Returns a lookup table for compression presets. (CLR only)"""
    from videometer import vm_utils_clr
    return vm_utils_clr.get_CompressionAndQuantificationPresetLUT()
