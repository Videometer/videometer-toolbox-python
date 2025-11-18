# Add path to ipp DLLs in runtime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

VMPATH = os.path.dirname(os.path.abspath(__file__))
path2ipp = os.path.join(VMPATH,"DLLs","IPP2019Update1","intel64")

# If DLLs are not found 
if not os.path.isdir(path2ipp):
    print("Attention. \nRequired DLLs were not found. Let me get them for you")
    from videometer.setup_helper import setupDlls
    setupDlls()

# Add the path to the IPP files at the front to it will be checked first
os.environ["PATH"] = path2ipp + ";" + os.environ["PATH"]

import matplotlib.pyplot as plt
import numpy as np
import clr
from videometer import vm_utils as utils

listOfDlls = ["VM.Image.IO.dll",
              "VM.Image.dll",
              "VM.FreehandLayerIO.dll",
              "VM.Image.ViewTransforms.dll"
              ]

for dllName in listOfDlls:
    path2dll = os.path.join(VMPATH,"DLLs","VM",dllName)
    if not os.path.isfile(path2dll):
        raise FileNotFoundError("File not found : " + path2dll)
    clr.AddReference(path2dll)


"""import class methods from C#"""
import VM.Image as VMIm
import VM.Image.IO as VMImIO
import VM.FreehandLayer as VMFreehand
import VM.Image.ViewTransforms as VMImTransForms

class ImageClass:
    """Attributes:
    --------------
    PixelValues - NumPy array (3-D)
        Contains float pixel values of the HIPS image. Shape of the array is 
        (height, width, bands).
    
    Height - int
        Height of the HIPS image.
    
    Width - int
        Width of the HIPS image.

    MmPixel - float
        physical size of each pixel in mm.
        
    Bands - int
        Number of bands in the image.
    
    BandNames - NumPy array of strings (1-D)
        List of names of the bands.
    
    WaveLengths - NumPy array of floats (1-D)
        Contains wavelenghts of the bands in HIPS image.

    Description - string
        Description set of the image
    
    History - string 
        Explains the history of the image.
    
    Illumination – NumPy array of strings (1-D)
        List of Illumination of each band.
            
    StrobeTimes – NumPy array of int32 (1-D)
        Strobe time of each band in the image.

    StrobeTimesUniversal - NumPy array of floats (1-D)
        Universal strobe time of every band in the image.
    
    FreehandLayers – List of dictionaries
        Each set FreehandLayer is a dictionary (hashmap) with the following keys :
            {
                "name" : (string) name of the layer f.x. Layer 1
                "layerId" : (int) ID of the layer
                "description" : (string) Description set
                "pixels" :  (2D numpy array with 0.0 and 1.0) pixel mask
            }        
    
    RGBPixels – NumPy array (height, width, 3)
        Array representing sRGB pixel values of the image. The values have to be 
        manually initialized using To_sRGB method.
    
    ForegroundPixels – NumPy array (2-D)
        Foreground mask of the image given as a binary 2-D numpy array. If no foreground
        pixels exist on the image, the value of this attribute is given as None. 
    
    DeadPixels – NumPy array (2-D)
        Dead pixels of the image given as a binary 2-D numpy arra. If no foreground
        pixels exist on the image, the value of this attribute is given as None. 
    
    CorrectedPixels – NumPy array (2-D)
        Corrected pixels of the image given as a binary 2-D numpy arra. If no foreground
        pixels exist on the image, the value of this attribute is given as None. 
    
    SaturatedPixels – NumPy array (2-D)
        Saturated pixels of the image given as a binary 2-D numpy arra. If no foreground
        pixels exist on the image, the value of this attribute is given as None. 
    
    ExtraData – dictionary
        Contains additional information about the image (e.g. temperature 
        data and similar). Given as dictionary. 

    ExtraDataInt – dictionary
        Contains additional information about the image (e.g. temperature 
        data and similar). Given as dictionary.

    ExtraDataString – dictionary
        Contains additional information about the image (e.g. temperature 
        data and similar). Given as dictionary.
    
        
    Methods:
    ---------
    init(image_array, image_object, bandIndexesToUse=[])
        Initializes the class. Called when reading an image.
    
    To_sRGB(bandIndexesToUse=[])
        Performs conversion of the spectral image to sRGB image. Updates
        'RGBPixels' attribute.
    
    reduceBands(bandIndexesToUse)
        Reduces bands of the image. The bands that will remain are given by bandIndexesToUse.
    """
        
    def __init__(self, path, bandIndexesToUse=[], ifSkipReadingAllLayers=False, ifSkipReadingFreehandLayer=False):
        """Initializes the class. Called when reading an image.
        Parameters:
        -----------
        path : string
            Path to the image that will be stored as an object of ImageClass.
        
        bandIndexesToUse : Optional argument, give a list or numpy array of band indexes
        
        ifSkipReadingAllLayers : Optional argument, If set to True it will skip reading all the Image Layers. 
                                Makes the reading quicker. Default is False.

        ifSkipReadingFreehandLayer : Optional argument, If set to True it will skip reading all the Freehand Layers. 
                                Makes the reading quicker. Default is False.

        """
        
        if len(bandIndexesToUse) != 0:
            utils.checkIfbandIndexesToUseIsValid(bandIndexesToUse,self.Bands)
        
        VMImageObject= VMImIO.HipsIO.LoadImage(path)
        self.PixelValues = utils.vmImage2npArray(VMImageObject)

        (self.Height, self.Width, self.Bands) = self.PixelValues.shape
    
        self.Bands = int(VMImageObject.Bands)
        self.BandNames=np.array([str(bandname) for bandname in VMImageObject.BandNames]) 
        self.Illumination=utils.illuminationObjects2List(VMImageObject.Illumination)
        self.WaveLengths=utils.asNumpyArray(VMImageObject.WaveLengths)
        self.StrobeTimes=utils.asNumpyArray(VMImageObject.StrobeTimes)
        self.StrobeTimesUniversal = utils.asNumpyArray(VMImageObject.StrobeTimesUniversal)
        
        self._BandCompressionModeObject = VMImageObject.BandCompressionMode
        self._QuantificationParametersObject = VMImageObject.QuantificationParameters
        self.MmPixel=VMImageObject.MmPixel
        self.History=VMImageObject.History
        self.Description=VMImageObject.Description
        self.ImageFileName=os.path.basename(path)
        self.FullPathToImage=os.path.abspath(path)
        self.FreehandLayers=None
        self.ForegroundPixels=None
        self.DeadPixels=None
        self.SaturatedPixels=None
        self.CorrectedPixels=None
        self.RGBPixels=None

        self.ExtraData=dict()
        for i in VMImageObject.ExtraData.Keys:
            self.ExtraData[i]=VMImageObject.ExtraData[i]
        
        self.ExtraDataInt = dict()
        for i in VMImageObject.ExtraDataInt.Keys:
            self.ExtraDataInt[i]=VMImageObject.ExtraDataInt[i]
        
        self.ExtraDataString = dict()
        for i in VMImageObject.ExtraDataString.Keys:
            self.ExtraDataString[i]=VMImageObject.ExtraDataString[i]
            
        if not ifSkipReadingAllLayers:
            self._ReadAllImageLayers(VMImageObject, ifSkipReadingFreehandLayer)
        
        if len(bandIndexesToUse) != 0:
            self.reduceBands(bandIndexesToUse)
            
    
    def _ReadAllImageLayers(self, VMImageObject,ifSkipReadingFreehandLayer):
        """Calls _ReadFreehand, _ReadCorrected, _ReadForeground, _ReadDead and _ReadSaturated.
        This method is called when initializing the class.

        ifSkipReadingFreehandLayer : If set to True it will skip reading all the Freehand Layers. 
                                Makes the reading quicker. 
        
        Parameters: None
        
        Output : No outputs."""
        
        getImageLayer = VMIm.ImageLayerExtensions.GetImageLayer

        # Freehand Layer
        if not ifSkipReadingFreehandLayer:
            self._ReadFreehand(VMImageObject.FreehandLayersXML)
       
        # CorrectedPixels
        self.CorrectedPixels = utils.imageLayer2npArray(getImageLayer(VMImageObject, "CorrectedPixels"))
        
        # DeadPixels
        self.DeadPixels = utils.imageLayer2npArray(getImageLayer(VMImageObject, "DeadPixels"))

        # ForegroundPixels
        self.ForegroundPixels = utils.imageLayer2npArray(getImageLayer(VMImageObject, "ForegroundPixels"))

        # SaturatedPixels
        self.SaturatedPixels = utils.imageLayer2npArray(getImageLayer(VMImageObject, "SaturatedPixels"))
        


    def _ReadFreehand(self, freehandLayersXMLstring):
        """Reads FreehandLayers layers of the image. Updates 'FreehandLayers' attribute.
        The method is called when initializing the class.
        
        Parameters: None
        
        Output: No direct outputs, it updates 'FreehandLayers' attribute, if at least
        one layer has been found."""

        # Would be easier to Get image with the help of VM.Image.IO 
        # vmImg = VMImIO.FreehandLayerIO.GetMaskFromFreehandLayerXmlString(VMImageObject, container.layerId)  
        # but for some reason throws a SystemOutOfMemory exception
        freehandLayersIO = VMFreehand.FreehandLayerIO.DeserializeFromString(freehandLayersXMLstring)
        if freehandLayersIO is None:
            return 
        
        freehandLayerList = []
        for container in freehandLayersIO.containers:
            # FreehandLayerIOContainer.pixels to npArray
            ms = clr.System.IO.MemoryStream(container.pixels)
            bitmap = clr.System.Drawing.Bitmap(ms)
            npArray = utils.systemDrawingBitmap2npArray(bitmap)
            
            if len(npArray.shape) == 3:
                npArray = npArray[:,:,0]

            # Scale down to binary
            npArray = npArray / np.max([np.max(npArray), 1])

            # Add to the freehandLayerDict with the same key as in VideometerLab software
            freehandObject = {
                "name" : "Layer "+str(container.layerId+1),
                "layerId" : container.layerId,
                "description" : container.description,
                "pixels" :  npArray
                }
            freehandLayerList.append(freehandObject)

            bitmap.Dispose()
            ms.Dispose()

        self.FreehandLayers=freehandLayerList 


    def to_sRGB(self,spectraName='D65'):
        """Performs conversion of the spectral image to sRGB image.
        
        Parameters:
        -----------
        spectraName - string
            Name of the spectra to be used in the sRGB conversion. Default 
            spectraName is 'D65'.
        Output : 
            returns the sRGB image and updates the "RGBPixels" attribute """
        
        SpectraNamesLUT = utils.get_SpectraNamesLUP()
        if not (spectraName in SpectraNamesLUT):
            raise NotImplementedError("spectraName=\"" + spectraName + "\" is not implemented. \nList of implemented spectras: "+ str(list(SpectraNamesLUT.keys())))

        # Take only the wavelengths with the right illumination
    
        allowableIlluminations = ["Diffused_Highpower_LED","Diffused_Lowpower_LED","Direct_Lowpower_LED","Coaxial_FrontLight"]
        
        diffusedMask = np.isin(self.Illumination, allowableIlluminations)
    
        # Create a new VM object to parse through the SRGBViewTransform with only the visable wavelengths
        visableMask = (380 <= self.WaveLengths) * (self.WaveLengths <= 780)
        
        visableBands = diffusedMask & visableMask
    
        if np.sum(visableBands) < 3:
            raise TypeError("Image class needs to have 3 or more wavelengths on the visable spectrum (380mm <= wavelength <= 780mm). Number of visable wavelength in ImageClass : "+ str(np.sum(visableBands)))
        VMImageObject = utils.npArray2VMImage(self.PixelValues[:,:,visableBands])

        # Add attributes that are checked in IsValidFor()
        VMImageObject.AddToHistory(self.History)
        indexVisableBands = np.where(visableBands)[0]
        for i in range(len(indexVisableBands)):
            VMImageObject.WaveLengths[i] = float(self.WaveLengths[indexVisableBands[i]])

        # Converter initialization and check
        converter = VMImTransForms.MultiBand.SRGBViewTransform()
        if not converter.IsValidFor(VMImageObject):
            raise TypeError("VM.Image.NaturalColorConversion.InvalidConversionException: Only reflectance calibrated images are supported")

        # Convert to sRGB image
        bitmap = converter.GetBitmap(VMImageObject, SpectraNamesLUT[spectraName])
        srgbImage = utils.systemDrawingBitmap2npArray(bitmap).astype(np.uint8)       
        self.RGBPixels = srgbImage

        VMImageObject.Free()

        return srgbImage


    def reduceBands(self,bandIndexesToUse):
        """Reduces bands of the image.
        
        Parameters:
        -----------
        bandIndexesToUse - list or numpy array of integers
            Specifies which bands will remain in the object. The parameter 
            should be given as a list if more than one band should stay in the
            image.
        
        Output : No direct outputs, 
            Update the attributes : 
                - 'PixelValues' 
                - 'Bands'
                - 'BanbNames'
                - 'WaveLengths'
                - 'Illuminations'
                - 'StrobeTimes'
                - 'StrobeTimesUniversal'
                - A object overseeing the compression for each band (_QuantificationParametersObject)
   
        """
    
        utils.checkIfbandIndexesToUseIsValid(bandIndexesToUse,self.Bands)
        self.PixelValues = self.PixelValues[:,:,bandIndexesToUse]
        self.Bands = len(bandIndexesToUse)
        self.BandNames = self.BandNames[bandIndexesToUse]
        self.WaveLengths = self.WaveLengths[bandIndexesToUse]
        self.StrobeTimes = self.StrobeTimes[bandIndexesToUse]
        self.Illumination = self.Illumination[bandIndexesToUse]
        self.StrobeTimesUniversal = self.StrobeTimesUniversal[bandIndexesToUse]


        if self._QuantificationParametersObject is None:
            return
        tmp = clr.System.Array.CreateInstance(VMIm.Compression.QuantificationParameters, len(bandIndexesToUse))        
        for i, bandIndexToUse in enumerate(bandIndexesToUse):
            tmp[i] = self._QuantificationParametersObject[bandIndexToUse]
        self._QuantificationParametersObject = tmp
    
       
        
    

# ---------------- Start of functions ---------------- 


def read(path, bandIndexesToUse=[], ifSkipReadingAllLayers=False, ifSkipReadingFreehandLayer=False):
    """Reads a HIPS image and stores it as an ImageClass object.
    
    Parameters:
    -----------
    path - string
        Full path to the image that wants to be read.
    
    bandIndexesToUse - list or numpy array of integers
        Additional argument if only certain bands of the image want to be read.        
        
    ifSkipReadingAllLayers : Optional argument, If set to True it will skip reading all the Image Layers. 
                                Makes the reading quicker. Default is False.

    ifSkipReadingFreehandLayer : Optional argument, If set to True it will skip reading all the Freehand Layers. 
                                Makes the reading quicker. Default is False.

    Outputs:
    --------
    image - ImageClass object
        An object of ImageClass. The object will be initialized using init()
        method of the ImageClass."""
    
    if type(path) != str:
        raise TypeError("path needs to be of type str")
    if not path.endswith(".hips"):
        raise ValueError("File needs to contain the .hips extension :" + path)
    if not os.path.isfile(path):
        raise FileNotFoundError("Couldn't locate " + path)

    return ImageClass(path, bandIndexesToUse, ifSkipReadingAllLayers, ifSkipReadingFreehandLayer) 
    

def write(image, path, compression="SameAsImageClass", verbose=False):
    """Writes a HIPS image from an ImageClass object or a NumPy array that
    corresponds to the pixel values of a spectral image.
    
    Parameters:
    -----------
    image - ImageClass object or NumPy array (3-D)
    
    path - string
        path of the HIPS file that will be created by the function. It has to
        include .hips extension.

    compression - string
        Level of compression : 
            {
                "SameAsImageClass" : Keep the same compression as is on the imageClass (if it is a numpy array then it will be Uncompressed) ,
                "Uncompressed" : No compression, (Is marked as ORIGINAL in VideometerLab software),
                "VeryHighQuality" : (see VideometerLab software),
                "HighQuality" : (see VideometerLab software),
                "HighCompression" : (see VideometerLab software),
                "VeryHighCompression" : (see VideometerLab software)
            }

    verbose - bool
        If true then prints out the name of the file otherwise not
        Default is false
    
    Outputs : Returns the path if successful otherwise None."""
    
    if not (path.endswith(".hips")):
        raise TypeError("File needs to contain the .hips extension :" + path)
    folderPath = os.path.dirname(os.path.abspath(path))
    if not (os.path.isdir(folderPath)):
        raise FileNotFoundError("The folder structure not found under " + path + " : " + folderPath )

    if ((type(image) == np.ndarray) and (len(image.shape)==3)):
        imagearr = image
        if compression == "SameAsImageClass":
            compression = "Uncompressed"
    elif(type(image) == ImageClass):
        imagearr = image.PixelValues

        # Check if all Image layers match the size of the image.        
        h,w,_ = imagearr.shape

        if not (image.CorrectedPixels is None) and (image.CorrectedPixels.shape != (h,w)):
            raise ValueError("CorrectedPixels "+str(image.CorrectedPixels.shape)+" and PixelValues "+str((h,w))+" shape do not match")
        if not (image.DeadPixels is None) and  (image.DeadPixels.shape != (h,w)):
            raise ValueError("DeadPixels "+str(image.DeadPixels.shape)+" and PixelValues "+str((h,w))+" shape do not match")
        if not (image.ForegroundPixels is None) and  (image.ForegroundPixels.shape != (h,w)):
            raise ValueError("ForegroundPixels "+str(image.ForegroundPixels.shape)+" and PixelValues "+str((h,w))+" shape do not match")
        if not (image.SaturatedPixels is None) and  (image.SaturatedPixels.shape != (h,w)):
            raise ValueError("SaturatedPixels "+str(image.SaturatedPixels.shape)+" and PixelValues "+str((h,w))+" shape do not match")
        
        if not (image.FreehandLayers is None): 
            for i, freehand in enumerate(image.FreehandLayers):
                if freehand["pixels"].shape != (h,w):
                    raise ValueError("FreehandLayers "+str(freehand["pixels"].shape)+" at i="+str(i)+" and PixelValues "+str((h,w))+" shape do not match")

    else:
        raise TypeError("Image input has to be either ImageClass object or 3-D NumPy array")
    
    if compression != "SameAsImageClass":
        compressionLUT = utils.get_CompressionAndQuantificationPresetLUT()

        if not compression in compressionLUT:
            listOfValid = list(compressionLUT.keys())
            listOfValid.append("SameAsImageClass")
            raise NotImplementedError(compression + " is not implemented. \nList of valid : " + str(listOfValid)) 
        
        # CompressionMode on the Compression Preset Object is set to RAW ... which will fail so this is fixed by setting it to None  
        if compression == "Uncompressed":
            bandCompressionMode = None
        else:
            bandCompressionMode = compressionLUT[compression].CompressionParameters.CompressionMode

        quantValue = compressionLUT[compression].QuantificationParameters
        bands = imagearr.shape[2]
        quantificationParameters = clr.System.Array.CreateInstance(VMIm.Compression.QuantificationParameters, bands)
        for i in range(bands):
            quantificationParameters[i] = quantValue

    else:
        bandCompressionMode = image._BandCompressionModeObject
        quantificationParameters  = image._QuantificationParametersObject

    Image_net = utils.npArray2VMImage(imagearr)

    if(type(image) == ImageClass):
        #NOTE - DO NOT REMOVE THE CASTING TO THE TYPES str(),float(),int()
        #       CLR CAN'T CAST FROM np.str_, np.float32, np.int32 TO 
        #       System.String(), System.Single(), System.Int()  

        Image_net.BandCompressionMode = bandCompressionMode
        Image_net.QuantificationParameters = quantificationParameters
        Image_net.Description = str(image.Description)
        Image_net.AddToHistory(str(image.History))
        Image_net.MmPixel = float(image.MmPixel)

        illuminations_objects = utils.illuminationList2Objects(image.Illumination)
        for i in range(image.Bands):
            Image_net.WaveLengths[i] = float(image.WaveLengths[i])
            Image_net.StrobeTimes[i] = int(image.StrobeTimes[i])
            Image_net.Illumination[i] = illuminations_objects[i]
            Image_net.BandNames[i] = str(image.BandNames[i])
            Image_net.StrobeTimesUniversal[i] = float(image.StrobeTimesUniversal[i])


        for k,v in image.ExtraData.items():
            Image_net.ExtraData[k] = float(v)
        for k,v in image.ExtraDataInt.items():
            Image_net.ExtraDataInt[k] = int(v)
        for k,v in image.ExtraDataString.items():
            Image_net.ExtraDataString[k] = str(v)      
        
        Image_net = utils.addAllAvailableImageLayers(Image_net, image)

    VMImIO.HipsIO.SaveImage(Image_net, str(path))

    Image_net.Free()

    if os.path.isfile(path):
        fullPath = os.path.abspath(path)
        if verbose:
            print("HIPS image successfully written in " + fullPath)
        return fullPath
    else:
        if verbose:
            print("Failed to write HIPS image")
        return None

    
def show(image, ifUseMask=False ,bandIndexesToUse=[], ifOnlyGetListOfPLTObjects=False):
    """Function that shows individual bands of the image. By default it 
    displays all the bands, but if only certain bands want to be ploted, assign 
    bands as a list or integer.
    
    Parameters:
    -----------
    image - ImageClass object or NumPy array (3-D)

    ifUseMask - boolean
        If set to true and mask is set on the ImageClass object 
        then the image will show masked otherwise it won't.
    
    bandIndexesToUse - list 
        Optional argument that corresponds to the bands that want to be shown.

    ifOnlyGetListOfPLTObjects - boolean
        Optional argument to only get a list of the matplotlib.image.AxesImage objects
        instead of using matplotlib.show
    
    Outputs:
    --------
    plt_outputs - list
        List of matplotlib.image.AxesImage objects. This list is used to test
        the function, and on its own, it has no value for the user. Therefore,
        it is excluded from documentation."""
    
    if (type(image)!=ImageClass) or (type(image)==np.ndarray and len(image.shape)==3):
        raise TypeError("image needs to be a ImageClass object or 3-D numpy array")
    
    if(type(image)==ImageClass):
        imagearr = image.PixelValues
        if ifUseMask:
            if image.ForegroundPixels is None:
                raise AttributeError("ForegroundPixels attribute not set")

            for i in range(image.Bands):
                imagearr[: , : , i] = np.multiply(imagearr[: , : , i], image.ForegroundPixels)

    else:
        imagearr = image
        if ifUseMask:
            raise TypeError("image needs to be a ImageClass object to be able to use ForegroundMask")

    
    if len(bandIndexesToUse) != 0:
        utils.checkIfbandIndexesToUseIsValid(bandIndexesToUse,imagearr.shape[2])
    else:
        bandIndexesToUse = list(range(imagearr.shape[2]))

    plt_outputs=[]
    for i in bandIndexesToUse:
        plt.figure(num=i)
        v_min=np.min(imagearr[:,:,i])
        v_max=np.max(imagearr[:,:,i])
        ax_im=plt.imshow(imagearr[:,:,i],cmap='gray',vmin=v_min,vmax=v_max)
        plt_outputs.append(ax_im)
        plt.title("Band {}".format(i+1))
        plt.axis('off')

    if not ifOnlyGetListOfPLTObjects:
        plt.show()

    return plt_outputs

                                                                           

def showRGB(imageClass, ifUseMask=False):
    """Function that shows srgb representation of the image. If ifUseForegroundMask is 
        set to true then the image will be shown masked.

    Parameters:
    -----------
    image - ImageClass object.
    
    ifUseMask - boolean     
        To toggle mask on or off if set on the ImageClass object
        
    
    Outputs:
    --------
    plt_outputs - matplotlib.image.AxesImage
        A matplotlib.image.AxesImage object. This object is used to test
        the function, and on its own, it has no value for the user. Therefore,
        it is excluded from documentation. The same will apply for the remaining
        show functions."""

    
    if type(imageClass) != ImageClass:
        raise TypeError("imageClass needs to be a ImageClass object")

    # This is a problematic statement 
    # if the user calls it > changes the images > calls it again
    # then it will display the old image .... 
    if (imageClass.RGBPixels is None):  
        imageClass.to_sRGB()
    image_arr = imageClass.RGBPixels

    imrgb = np.empty_like(image_arr)
    if ifUseMask:
        if imageClass.ForegroundPixels is None:
            raise AttributeError("ForegroundPixels attribute not set")
        
        for i in range(3):
            imrgb[: , : , i] = np.multiply(image_arr[: , : , i], imageClass.ForegroundPixels)
    else :
        imrgb = image_arr

    plt.figure(num = 1)
    ax_im = plt.imshow(imrgb)
    plt.title('RGB image')
    plt.axis('off')
    return ax_im




def readOnlyPixelValues(path):
    """Function that reads the HIPS image and returns its pixel values

    Parameters:
    -----------
    path - path to the .hips image
            
    
    Outputs:
    --------
    Returns a 3-D numpy float array.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if not path.endswith(".hips"):
        raise TypeError("File name has to have the .hips extension")
    
    VMImageObject = VMImIO.HipsIO.LoadImage(path)
    npArray = utils.vmImage2npArray(VMImageObject)
    VMImageObject.Free()

    return npArray


