import numpy as np
import numbers
from videometer import config

def checkIfbandIndexesToUseIsValid(bandIndexesToUse, nBandsInImageClass):
    """Checks if the values in bandIndexesToUse are valid

    Parameters:
    -----------
    bandIndexesToUse - list or numpy array of integers
        Additional argument if only certain bands of the image want to be read.

    Outputs: None but raises a TypeError if the bandIndexesToUse is not valid"""

    if not (type(bandIndexesToUse) == np.array or type(bandIndexesToUse) == list):
        raise TypeError("bandIndexesToUse should be a list")

    for i in range(len(bandIndexesToUse)):
        if (
            not (isinstance(bandIndexesToUse[i], numbers.Integral))
            or bandIndexesToUse[i] < 0
        ):
            raise TypeError(
                "Element at index "
                + str(i)
                + "(value="
                + str(bandIndexesToUse[i])
                + ") is not a positive integer in bandIndexesToUse"
            )

        if bandIndexesToUse[i] >= nBandsInImageClass:
            raise TypeError(
                "Element at index "
                + str(i)
                + "(value="
                + str(bandIndexesToUse[i])
                + ") is out of range for imageClass of bands="
                + str(nBandsInImageClass)
            )

# --- CLR Dispatchers ---

def imageLayer2npArray(imageLayer):
    from videometer import vm_utils_clr
    return vm_utils_clr.imageLayer2npArray(imageLayer)

def get_IlluminationLUT():
    from videometer import vm_utils_clr
    return vm_utils_clr.get_IlluminationLUT()

def illuminationObjects2List(illumnationObjects):
    from videometer import vm_utils_clr
    return vm_utils_clr.illuminationObjects2List(illumnationObjects)

def illuminationList2Objects(illuminationList):
    from videometer import vm_utils_clr
    return vm_utils_clr.illuminationList2Objects(illuminationList)

def asNumpyArray(netArray):
    from videometer import vm_utils_clr
    return vm_utils_clr.asNumpyArray(netArray)

def addAllAvailableImageLayers(VMImageObject, ImageClass):
    from videometer import vm_utils_clr
    return vm_utils_clr.addAllAvailableImageLayers(VMImageObject, ImageClass)

def addImageLayer(VMImageObject, npArray, typeOfLayer):
    from videometer import vm_utils_clr
    return vm_utils_clr.addImageLayer(VMImageObject, npArray, typeOfLayer)

def setFreehandLayers(VMImageObject, ImageClass):
    from videometer import vm_utils_clr
    return vm_utils_clr.setFreehandLayers(VMImageObject, ImageClass)

def vmImage2npArray(vmImage):
    from videometer import vm_utils_clr
    return vm_utils_clr.vmImage2npArray(vmImage)

def asNetArrayMemMove(npArray):
    from videometer import vm_utils_clr
    return vm_utils_clr.asNetArrayMemMove(npArray)

def npArray2VMImage(npArray):
    from videometer import vm_utils_clr
    return vm_utils_clr.npArray2VMImage(npArray)

def get_SpectraNamesLUP():
    from videometer import vm_utils_clr
    return vm_utils_clr.get_SpectraNamesLUP()

def systemDrawingBitmap2npArray(bitmap):
    from videometer import vm_utils_clr
    return vm_utils_clr.systemDrawingBitmap2npArray(bitmap)

def get_CompressionAndQuantificationPresetLUT():
    from videometer import vm_utils_clr
    return vm_utils_clr.get_CompressionAndQuantificationPresetLUT()
