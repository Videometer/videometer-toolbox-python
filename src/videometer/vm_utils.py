import numpy as np
import sys
import numbers
import ctypes
from PIL import Image
import tempfile

DLL_PATH = r"C:\Users\heh\repos\VMLab\src\VideometerLab\bin\x64\Release\net8.0-windows"
DLL_PATH = r"C:\Users\heh\repos\VM.Blobs\src\VM.Blobs\bin\x64\Release\net8.0\publish"
sys.path.append(DLL_PATH)


import pythonnet
# This MUST be called before "import clr"
pythonnet.load("coreclr")
import clr, System

clr.AddReference("VM.Blobs")
clr.AddReference("VM.Image.ViewTransforms")
clr.AddReference("VM.Image")
clr.AddReference("VM.Jobs")
clr.AddReference("VM.FreehandLayerIO")

from System.Runtime.InteropServices import GCHandle, GCHandleType

import VM.Image as VMIm
import VM.Image.IO as VMImIO
import VM.Illumination as VMill
import VM.Image.ColorConversion as VMImNatColorConv
import VM.FreehandLayer as VMFreehand
import VM.Image.Compression as VMImgCompression

from VM.Jobs import Job

def event_handler(sender, exception):
    print(sender)
    print(exception)

Job.UnhandledException += event_handler

def imageLayer2npArray(imageLayer):
    if imageLayer is None:
        return None

    if "Image" in dir(imageLayer):
        vmImg = imageLayer.Image
    else:
        pass

    npArray = vmImage2npArray(vmImg)[:, :, 0]
    npMax = np.max(npArray)
    if npMax != 0.0:
        npArray = npArray / npMax

    vmImg.Free()
    return npArray.astype(np.int32)


def get_IlluminationLUT():
    # Illumination look up table <string nameOfIllumation, object enumIllumnationType>
    IlluminationLUT = dict()

    for v in System.Enum.GetValues(VMill.IlluminationType):
        IlluminationLUT[str(v)] = v

    return IlluminationLUT


def illuminationObjects2List(illumnationObjects):
    """
    Takes in the list of illumination objects
    and returns a list of the illumnations names
    """

    return np.array([str(illumObj) for illumObj in illumnationObjects])


def illuminationList2Objects(illuminationList):
    """
    Reverse implementation of illuminationObjects2Name
    """
    illuminationLUP = get_IlluminationLUT()

    illuminationObjects = []
    for ill in illuminationList:
        illuminationObjects.append(illuminationLUP[ill])

    return np.array(illuminationObjects)


def asNumpyArray(netArray):
    """
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for
    the mapping of CLR types to Numpy dtypes.
    """
    _MAP_NET_NP = {"Single": np.float32, "Int32": np.int32, "Byte": np.uint8}

    netType = netArray.GetType().GetElementType().Name
    if not netType in _MAP_NET_NP:
        raise NotImplementedError(
            "asNumpyArray does not yet support System type {}".format(netType)
        )

    # Get shape of netArray
    dims = np.zeros(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)

    # Take in any dimensions of array and iterate through it in one for loop
    npArray = np.ctypeslib.as_array(netArray, shape=dims).astype(_MAP_NET_NP[netType])

    return npArray


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


def addAllAvailableImageLayers(VMImageObject, ImageClass):
    # Set the "CorrectedPixels", "DeadPixels", "ForegroundPixels","SaturatedPixels" layers
    imageLayers = [
        ImageClass.CorrectedPixels,
        ImageClass.DeadPixels,
        ImageClass.ForegroundPixels,
        ImageClass.SaturatedPixels,
    ]
    imageLayerTypes = [
        "CorrectedPixels",
        "DeadPixels",
        "ForegroundPixels",
        "SaturatedPixels",
    ]

    for imgLayer, imgLayerType in zip(imageLayers, imageLayerTypes):
        if imgLayer is None:
            continue
        VMImageObject = addImageLayer(VMImageObject, imgLayer, imgLayerType)

    # Set the FreehandLayer
    if not (ImageClass.FreehandLayers is None):
        VMImageObject = setFreehandLayers(VMImageObject, ImageClass)

    return VMImageObject


def addImageLayer(VMImageObject, npArray, typeOfLayer):

    if VMImageObject.GetType() != VMIm.VMImage:
        raise TypeError("VMImageObject needs to be of VM.Image.VMImage type")

    if not typeOfLayer in [
        "CorrectedPixels",
        "DeadPixels",
        "ForegroundPixels",
        "SaturatedPixels",
    ]:
        raise NotImplementedError(
            'typeOfLayer is only implemented for : "CorrectedPixels","DeadPixels","ForegroundPixels" and "SaturatedPixels"'
        )

    tmpImage = npArray2VMImage(npArray)
    imageLayer = VMIm.ImageLayer(tmpImage, 0)
    VMImageObject.ImageLayers[typeOfLayer] = imageLayer

    return VMImageObject


def setFreehandLayers(VMImageObject, ImageClass):

    # same as VM.GUI.Image.WinForms FreehandLayerIO_Extensions CreateFreehandLayerIO

    # IF we don't need the description we could just use the VM.Image.IO.FreehandlayerIO.SetMaskToFreehandLayerXmlString(this VMImage image, VMImage mask, int layerId) function

    arrayOfContainers = System.Array.CreateInstance(
        VMFreehand.FreehandLayerIOContainer, len(ImageClass.FreehandLayers)
    )

    for i, freehandLayer in enumerate(ImageClass.FreehandLayers):
        # pixels from numpy array to Byte[,]
        pixels = freehandLayer["pixels"].astype(np.float32)
        pixelsVMImage = npArray2VMImage(pixels)

        bitmap = VMImIO.DotNetBitmapIO.GetBitmap(pixelsVMImage)
        stream = clr.System.IO.MemoryStream()

        bitmap.Save(stream, clr.System.Drawing.Imaging.ImageFormat.Png)
        stream.Flush()

        # Create a container
        # FreehandLayerIOContainer(byte[] pixels, int layerId, bool locked, bool visible, string description)
        container = VMFreehand.FreehandLayerIOContainer(
            stream.GetBuffer(),
            freehandLayer["layerId"],
            False,
            True,
            freehandLayer["description"],
        )

        arrayOfContainers[i] = container

        pixelsVMImage.Free()

    FreehandLayerIO = VMFreehand.FreehandLayerIO(arrayOfContainers)

    VMImageObject.FreehandLayersXML = FreehandLayerIO.SerializeToString()

    return VMImageObject


def vmImage2npArray(vmImage):
    height = vmImage.Height
    width = vmImage.Width
    bands = vmImage.Bands

    npArray = np.empty((height, width, bands))
    for b in range(bands):
        bandLayer = VMIm.ImagePixelAccess.GetValues(vmImage, b)
        npArray[:, :, b] = asNumpyArray(bandLayer).reshape(height, width)

    vmImage.Free()

    return npArray


def asNetArrayMemMove(npArray):
    """
    Given a `numpy.ndarray` returns a CLR `System.Array`.  See _MAP_NP_NET for
    the mapping of Numpy dtypes to CLR types.

    """
    _MAP_NP_NET = {
        np.dtype("float32"): System.Single,
        np.dtype("int32"): System.Int32,
        np.dtype("uint8"): System.Byte,
    }
    dims = npArray.shape
    dtype = npArray.dtype

    if not npArray.flags.c_contiguous:
        npArray = npArray.copy(order="C")
    assert npArray.flags.c_contiguous
    try:
        netArray = System.Array.CreateInstance(_MAP_NP_NET[dtype], dims)
    except KeyError:
        raise NotImplementedError(
            "The function does not yet support dtype {}".format(dtype)
        )

    try:  # Memmove
        destHandle = GCHandle.Alloc(netArray, GCHandleType.Pinned)
        sourcePtr = npArray.__array_interface__["data"][0]
        destPtr = destHandle.AddrOfPinnedObject().ToInt64()
        ctypes.memmove(destPtr, sourcePtr, npArray.nbytes)
    finally:
        if destHandle.IsAllocated:
            destHandle.Free()
    return netArray


# NOTE The function uses memmory move so it might not be clear to C# objects and python
#       who actually owns (and can access) if it is changed.
def npArray2VMImage(npArray):
    if type(npArray) != np.ndarray or (
        len(npArray.shape) != 2 and len(npArray.shape) != 3
    ):
        raise TypeError("npArray needs to be a 2-D or 3-D numpy array")

    # Change dimensions to bands x height x width
    if len(npArray.shape) == 2:
        npArray = np.array([npArray])
    else:
        npArray = npArray.transpose([2, 0, 1])

    npArray = npArray.astype(np.float32)

    tmp = asNetArrayMemMove(npArray)

    VMImageObject = VMIm.VMImage(tmp)

    return VMImageObject


def get_SpectraNamesLUP():
    # Spectra names look up table <string nameOfSpectra, object SpectraName>

    SpectraNamesLUT = dict()
    for v in System.Enum.GetValues(VMImNatColorConv.SpectraNames):
        SpectraNamesLUT[str(v)] = v

    return SpectraNamesLUT


def systemDrawingBitmap2npArray(bitmap):
    # input a bitmap and return a numpy array

    # Note from devs :
    # Save bitmap from C# and read it from numpy (honestly not my best idea but works)
    # This is done because its quicker and easier easier than changing it to VM Image
    # and turning that into npArray. Also avoids the problem of reading a RGB VM Image
    # with GetPixelValues().
    # No System.Drawing.Bitmap to numpy array method was found if it is found please
    # switch it out for this.

    with tempfile.NamedTemporaryFile(delete_on_close=False) as f:
        tmp_path = f.name
        f.close()
        bitmap.Save(tmp_path)

        npArray = np.array(Image.open(tmp_path))

    bitmap.Dispose()

    return npArray


def get_CompressionAndQuantificationPresetLUT():
    # CompressionAndQuantificationPreset names look up table <string name, object CompressionsAndQuantificationPreset>

    # Since this is a Struct then we have no way of iterating through it so this has to be hard coded...
    # so if a name is added, deleted or modified in the struct this will need to be updated

    presetStruct = VMImgCompression.CompressionsAndQuantificationPreset

    CAndQLUT = {
        "Uncompressed": presetStruct.Uncompressed,
        "VeryHighQuality": presetStruct.VeryHighQuality,
        "HighQuality": presetStruct.HighQuality,
        "HighCompression": presetStruct.HighCompression,
        "VeryHighCompression": presetStruct.VeryHighCompression,
    }

    return CAndQLUT
