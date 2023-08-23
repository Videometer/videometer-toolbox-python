import numpy as np
import os
import clr, System
import numbers


"""Add Dlls to the clr"""
VMPATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
listOfDlls = ["VM.Image.dll",
              "VM.Image.IO.dll",
              "VM.Illumination",
              "VM.Image.NaturalColorConversion.dll",
              "VM.FreehandLayerIO.dll",
              "VM.Image.Compression.dll"
              ]

for dllName in listOfDlls:
    clr.AddReference(os.path.join(VMPATH,"VM",dllName))

import VM.Image as VMIm
import VM.Image.IO as VMImIO
import VM.Illumination as VMill
import VM.Image.NaturalColorConversion as VMImNatColorConv
import VM.FreehandLayer as VMFreehand
import VM.Image.Compression as VMImgCompression


def imageLayer2npArray(imageLayer):
    if imageLayer is None:
        return None
    
    if "Image" in dir(imageLayer):
        vmImg = imageLayer.Image
    else:
        pass
    
    npArray = vmImage2npArray(vmImg)[:,:,0]

    vmImg.Free()
    return (npArray/np.max(npArray)).astype(np.int32)
    



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
    

def asNetArray(npArray):
    '''
    Given a `numpy.ndarray` returns a CLR `System.Array` of type System.Single
    
    '''
    if type(npArray) != np.ndarray:
        raise TypeError("npArray input needs to a numpy array")
    
    # Take in any dimensions of array and iterate through it in one for loop
    dims = npArray.shape
    netArray = System.Array.CreateInstance(System.Single, dims)
    index = [0 for _ in range(netArray.Rank)]

    for _ in range(netArray.Length):
        ## Set value
        value = clr.System.Single(float(npArray[tuple(index)]))
        netArray.SetValue(value, index)

        ## Update index
        index[-1] += 1
        for j in range(netArray.Rank-1,-1,-1):
            if index[j] == dims[j]:
                index[j-1] += 1
                index[j] = 0 

    return netArray



def asNumpyArray(netArray):
    '''
    Given a CLR `System.Array` returns a `numpy.ndarray`.  See _MAP_NET_NP for 
    the mapping of CLR types to Numpy dtypes.
    '''
    _MAP_NET_NP = {
    'Single' : np.float32, 
    'Int32'  : np.int32,
    'Byte'  : np.uint8
    }

    netType = netArray.GetType().GetElementType().Name
    if not netType in _MAP_NET_NP:
        raise NotImplementedError("asNumpyArray does not yet support System type {}".format(netType))

    # Get shape of netArray
    dims = np.zeros(netArray.Rank, dtype=int)
    for I in range(netArray.Rank):
        dims[I] = netArray.GetLength(I)
    
    # Take in any dimensions of array and iterate through it in one for loop
    npArray = np.empty(dims, dtype=_MAP_NET_NP[netType])
    index = [0 for _ in range(netArray.Rank)]
    for _ in range(netArray.Length):
        ## Set value
        npArray[tuple(index)] =  netArray.GetValue(index)

        ## Update index
        index[-1] += 1
        for j in range(netArray.Rank-1,-1,-1):
            if index[j] == dims[j]:
                index[j-1] += 1
                index[j] = 0 

    return npArray



def checkIfbandIndexesToUseIsValid(bandIndexesToUse, nBandsInImageClass):
    """ Checks if the values in bandIndexesToUse are valid

    Parameters:
    -----------    
    bandIndexesToUse - list or numpy array of integers
        Additional argument if only certain bands of the image want to be read.
    
    Outputs: None but raises a TypeError if the bandIndexesToUse is not valid """

    if not (type(bandIndexesToUse)==np.array or type(bandIndexesToUse)==list):
        raise TypeError("bandIndexesToUse should be a list")

    for i in range(len(bandIndexesToUse)):
        if not (isinstance(bandIndexesToUse[i], numbers.Integral)) or bandIndexesToUse[i]<0:
            raise TypeError("Element at index "+str(i)+"(value="+str(bandIndexesToUse[i])+") is not a positive integer in bandIndexesToUse")
        
        if bandIndexesToUse[i] >= nBandsInImageClass:
            raise TypeError("Element at index "+str(i)+"(value="+str(bandIndexesToUse[i])+") is out of range for imageClass of bands="+str(nBandsInImageClass))
        



def addAllAvailableImageLayers(VMImageObject, ImageClass):
    # Set the "CorrectedPixels", "DeadPixels", "ForegroundPixels","SaturatedPixels" layers
    imageLayers = [ImageClass.CorrectedPixels, ImageClass.DeadPixels, ImageClass.ForegroundPixels, ImageClass.SaturatedPixels]
    imageLayerTypes = ["CorrectedPixels", "DeadPixels", "ForegroundPixels","SaturatedPixels"]
    
    for imgLayer, imgLayerType in zip(imageLayers, imageLayerTypes):
        if imgLayer is None:
            continue
        VMImageObject = addImageLayer(VMImageObject, imgLayer, imgLayerType)

    # Set the FreehandLayer
    VMImageObject = setFreehandLayers(VMImageObject, ImageClass)

    return VMImageObject


def addImageLayer(VMImageObject, npArray, typeOfLayer):

    if VMImageObject.GetType() != VMIm.VMImage:
        raise TypeError("VMImageObject needs to be of VM.Image.VMImage type")
    
    if not typeOfLayer in ["CorrectedPixels", "DeadPixels", "ForegroundPixels", "SaturatedPixels"]:
        raise NotImplementedError("typeOfLayer is only implemented for : \"CorrectedPixels\",\"DeadPixels\",\"ForegroundPixels\" and \"SaturatedPixels\"")
    
    tmpImage = npArray2VMImage(npArray)
    imageLayer = VMIm.ImageLayer(tmpImage,0)
    VMImageObject.ImageLayers[typeOfLayer] = imageLayer

    return VMImageObject


def setFreehandLayers(VMImageObject, ImageClass):

    # same as VM.GUI.Image.WinForms FreehandLayerIO_Extensions CreateFreehandLayerIO

    # IF we don't need the description we could just use the VM.Image.IO.FreehandlayerIO.SetMaskToFreehandLayerXmlString(this VMImage image, VMImage mask, int layerId) function

    arrayOfContainers = System.Array.CreateInstance(VMFreehand.FreehandLayerIOContainer, len( ImageClass.FreehandLayers) )

    for i, freehandLayer in enumerate(ImageClass.FreehandLayers):
        # pixels from numpy array to Byte[,]
        pixels = freehandLayer["pixels"]
        pixelsVMImage = npArray2VMImage(pixels)

        bitmap = VMImIO.DotNetBitmapIO.GetBitmap(pixelsVMImage)
        stream = clr.System.IO.MemoryStream()

        bitmap.Save(stream, clr.System.Drawing.Imaging.ImageFormat.Png)
        stream.Flush()

        # Create a container
        # FreehandLayerIOContainer(byte[] pixels, int layerId, bool locked, bool visible, string description)
        container = VMFreehand.FreehandLayerIOContainer(stream.GetBuffer(), freehandLayer["layerId"], False, True, freehandLayer["description"] )

        arrayOfContainers[i] = container

    FreehandLayerIO = VMFreehand.FreehandLayerIO(arrayOfContainers)
    
    VMImageObject.FreehandLayersXML = FreehandLayerIO.SerializeToString()


    return VMImageObject
    

def vmImage2npArray(vmImage):    
    height = vmImage.Height
    width = vmImage.Width
    bands = vmImage.Bands

    npArray = np.empty((height, width, bands), dtype=np.float32)
    for x in range(width):
        for y in range(height):
            for b in range(bands):
                npArray[y,x,b] = vmImage.GetPixel(x,y,b)

    vmImage.Free()

    return npArray


def npArray2VMImage(npArray):
    if type(npArray) != np.ndarray or (len(npArray.shape) != 2 and len(npArray.shape) != 3):
        raise TypeError("npArray needs to be a 2-D or 3-D numpy array") 

    npArray = npArray.astype(np.float32)
    if len(npArray.shape) == 3:
        npArray = np.transpose(npArray, (2, 0, 1)) ## VMImage(float[,,] data) has this format :: (bands,height,width)
    
    net_im = asNetArray(npArray)
    VMImageObject = VMIm.VMImage(net_im)
    return VMImageObject
            


def get_SpectraNamesLUP():
    # Spectra names look up table <string nameOfSpectra, object SpectraName> 
    
    SpectraNamesLUT = dict()
    for v in System.Enum.GetValues(VMImNatColorConv.SpectraNames):
        SpectraNamesLUT[str(v)] = v
        
    return SpectraNamesLUT


def systemDrawingBitmap2npArray(bitmap):
    npArray = np.zeros((bitmap.Height, bitmap.Width, 3), np.uint8)
    # Ignoring the A 
    for y in range(bitmap.Height):
        for x in range(bitmap.Width):
            pixel = bitmap.GetPixel(x,y)
            npArray[y,x,0] = pixel.R
            npArray[y,x,1] = pixel.G
            npArray[y,x,2] = pixel.B

    return npArray



def get_CompressionAndQuantificationPresetLUT():
    # CompressionAndQuantificationPreset names look up table <string name, object CompressionsAndQuantificationPreset> 
    
    # Since this is a Struct then we have no way of iterating through it so this has to be hard coded...
    # so if a name is added, deleted or modified in the struct this will need to be updated

    presetStruct = VMImgCompression.CompressionsAndQuantificationPreset
    
    CAndQLUT = {
        "Uncompressed" : presetStruct.Uncompressed,
        "VeryHighQuality" : presetStruct.VeryHighQuality,
        "HighQuality" : presetStruct.HighQuality,
        "HighCompression" : presetStruct.HighCompression,
        "VeryHighCompression" : presetStruct.VeryHighCompression       
    }
    
    return CAndQLUT


        
        
        