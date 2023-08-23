# Add path to ipp DLLs in runtime
import os
VMPATH = os.path.dirname(os.path.abspath(__file__))
path2ipp = os.path.join(VMPATH,"DLLs","IPP2019Update1","intel64")

# If DLLs 
if not os.path.isdir(path2ipp):
    print("Attention. \nRequired DLLs were not found. Let me get them for you")
    from videometer.setupHelper import setupDlls
    setupDlls()


os.environ["PATH"] += ";"+ path2ipp

import matplotlib.pyplot as plt
import numpy as np
import clr
import videometer.HelpFunctions as hf 


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
    MmPixel - float
        physical size of each pixel in mm.

    StrobeTimesUniversal - NumPy array of floats (1-D)
        Universal strobe time of every band in the image.

    BandNames - NumPy array of strings (1-D)
        List of names of the bands.

    PixelValues - NumPy array of floats (3-D)
        Contains pixel values of the HIPS image. Shape of an array is 
        (height, width, bands).
    
    Height - int
        Height of the HIPS image.
    
    Width - int
        Width of the HIPS image.

    Bands - int
        Number of bands in the image.
    
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
    
    dir()
        Displays attributes of the class.
    
    
    _ReadAllImageLayers(VM.Image.VMImage)
        Reads the CorrectedPixels, DeadPixels, ForegroundPixels and SaturatePixels
        from the VM.Image.VMImage object and sets them to their respective attribute. 
        Then calls _ReadFreehand. 
        This method is called when initializing the class

    _ReadFreehand(VM.Image.VMImage)
        Reads FreehandLayers layers of the image. Updates 'FreehandLayers' attribute.
        The method is called when initializing the class.
    
    To_sRGB(bandIndexesToUse=[])
        Performs conversion of the spectral image to sRGB image. Updates
        'RGBPixels' attribute.
    
    _ReduceBands(bandIndexesToUse)
        Reduces bands of the image. The bands that will remain are given by bandIndexesToUse.
    
    """
        
    def __init__(self, path, bandIndexesToUse=[]):
        """Initializes the class. Called when reading an image.
        Parameters:
        -----------
        path : string
            Path to the image that will be stored as an object of ImageClass.
        
        bandIndexesToUse : Optional argument, give a list or numpy array of band indexes.."""
        
        if len(bandIndexesToUse) != 0:
            hf.checkIfbandIndexesToUseIsValid(bandIndexesToUse,self.Bands)
        
        VMImageObject= VMImIO.HipsIO.LoadImage(path)
        self.PixelValues = hf.vmImage2npArray(VMImageObject)
        
        (self.Height, self.Width, self.Bands) = self.PixelValues.shape
    
        self.Bands = int(VMImageObject.Bands)
        self.BandNames=np.array([str(bandname) for bandname in VMImageObject.BandNames]) 
        self.Illumination=hf.illuminationObjects2List(VMImageObject.Illumination)
        self.WaveLengths=hf.asNumpyArray(VMImageObject.WaveLengths)
        self.StrobeTimes=hf.asNumpyArray(VMImageObject.StrobeTimes)
        self.StrobeTimesUniversal = hf.asNumpyArray(VMImageObject.StrobeTimesUniversal)

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
                  
        self._ReadAllImageLayers(VMImageObject)

        
        if len(bandIndexesToUse) != 0:
            self.ReduceBands(bandIndexesToUse)
            
    
    def _ReadAllImageLayers(self, VMImageObject):
        """Calls _ReadFreehand, _ReadCorrected, _ReadForeground, _ReadDead and _ReadSaturated.
        This method is called when initializing the class.
        
        Parameters: None
        
        Output : No outputs."""
        
        getImageLayer = VMIm.ImageLayerExtensions.GetImageLayer

        # Freehand Layer
        self._ReadFreehand(VMImageObject)
       
        # CorrectedPixels
        self.CorrectedPixels = hf.imageLayer2npArray(getImageLayer(VMImageObject, "CorrectedPixels"))
        
        # DeadPixels
        self.DeadPixels = hf.imageLayer2npArray(getImageLayer(VMImageObject, "DeadPixels"))
        
        # ForegroundPixels
        self.ForegroundPixels = hf.imageLayer2npArray(getImageLayer(VMImageObject, "ForegroundPixels"))

        # SaturatedPixels
        self.SaturatedPixels = hf.imageLayer2npArray(getImageLayer(VMImageObject, "SaturatedPixels"))
        


    def _ReadFreehand(self, VMImageObject):
        """Reads FreehandLayers layers of the image. Updates 'FreehandLayers' attribute.
        The method is called when initializing the class.
        
        Parameters: None
        
        Output: No direct outputs, it updates 'FreehandLayers' attribute, if at least
        one layer has been found."""

        # Would be easier to Get image with the help of VM.Image.IO 
        # vmImg = VMImIO.FreehandLayerIO.GetMaskFromFreehandLayerXmlString(VMImageObject, container.layerId)  
        # but for some reason throws a SystemOutOfMemory exception
        
        freehandLayersIO = VMFreehand.FreehandLayerIO.DeserializeFromString(VMImageObject.FreehandLayersXML)
        if freehandLayersIO is None:
            return 
        
        freehandLayerList = []
        for container in freehandLayersIO.containers:
            # FreehandLayerIOContainer.pixels to npArray
            ms = clr.System.IO.MemoryStream(container.pixels)
            bitmap = clr.System.Drawing.Bitmap(ms)
            vmImage = VMImIO.DotNetBitmapIO.GetVMImage(bitmap)

            npArray = hf.vmImage2npArray(vmImage)[:,:,0]  
            npArray = npArray / np.max(npArray)           
            
            # Add to the freehandLayerDict with the same key as in VideometerLab software
            freehandObject = {
                "name" : "Layer "+str(container.layerId+1),
                "layerId" : container.layerId,
                "description" : container.description,
                "pixels" :  npArray
                }
            freehandLayerList.append(freehandObject)
            
            vmImage.Free()

        self.FreehandLayers=freehandLayerList 


    def To_sRGB(self,spectraName='D65'):
        """Performs conversion of the spectral image to sRGB image.
        
        Parameters:
        -----------
        spectraName - string
            Name of the spectra to be used in the sRGB conversion. Default 
            spectraName is 'D65'.
        Output : 
            returns the sRGB image and updates the "RGBPixels" attribute """
        
        SpectraNamesLUT = hf.get_SpectraNamesLUP()
        if not (spectraName in SpectraNamesLUT):
            raise NotImplementedError("spectraName=\"" + spectraName + "\" is not implemented. \nList of implemented spectras: "+str(list(SpectraNamesLUT.keys())))

        # Create a new VM object to parse through the SRGBViewTransform
        VMImageObject = hf.npArray2VMImage(self.PixelValues)

        # Add attributes that are checked in IsValidFor()
        VMImageObject.AddToHistory(self.History)
        for i in range(self.Bands):
            VMImageObject.WaveLengths[i] = float(self.WaveLengths[i])

        # Converted initialization and check
        converter = VMImTransForms.MultiBand.SRGBViewTransform()
        if not converter.IsValidFor(VMImageObject):
            raise TypeError("VM.Image.NaturalColorConversion.InvalidConversionException: Only reflectance calibrated images are supported")

        # Convert to sRGB image
        bitmap = converter.GetBitmap(VMImageObject, SpectraNamesLUT[spectraName])
        srgbImage = hf.systemDrawingBitmap2npArray(bitmap)
        
        self.RGBPixels = srgbImage

        return srgbImage


    def _ReduceBands(self,bandIndexesToUse):
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
    
        hf.checkIfbandIndexesToUseIsValid(bandIndexesToUse,self.Bands)
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


def read(path, bandIndexesToUse=[]):
    """Reads a HIPS image and stores it as an ImageClass object.
    
    Parameters:
    -----------
    path - string
        Full path to the image that wants to be read.
    
    bandIndexesToUse - list or numpy array of integers
        Additional argument if only certain bands of the image want to be read.
    
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
     

    return ImageClass(path, bandIndexesToUse) 
    

def write(image, path, compression="SameAsImageClass"):
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
    
    Outputs : Returns the path if successful otherwise None."""
    
    if not (path.endswith(".hips")):
        raise TypeError("File needs to contain the .hips extension :" + path)
        
    if ((type(image) == np.ndarray) and (len(image.shape)==3)):
        imagearr = image
        if compression == "SameAsImageClass":
            compression = "Uncompressed"
    elif(type(image) == ImageClass):
        imagearr = image.PixelValues
    else:
        raise TypeError("Image input has to be either ImageClass object or 3-D NumPy array")
    
    if compression != "SameAsImageClass":
        compressionLUT = hf.get_CompressionAndQuantificationPresetLUT()

        if not compression in compressionLUT:
            listOfValid = list(compressionLUT.keys())
            listOfValid.append("SameAsImageClass")
            raise NotImplementedError(compression + " is not implemented. \nList of valid : " + str(listOfValid)) 
        
        bandCompressionMode = compressionLUT[compression].CompressionParameters
        quantValue = compressionLUT[compression].QuantificationParameters
        bands = image.shape[2]
        quantificationParameters = clr.System.Array.CreateInstance(VMIm.Compression.QuantificationParameters, bands)
        for i in range(bands):
            quantificationParameters[i] = quantValue

    else:
        bandCompressionMode = image._BandCompressionModeObject
        quantificationParameters  = image._QuantificationParametersObject


    Image_net = hf.npArray2VMImage(imagearr)

    if(type(image) == ImageClass):
        #NOTE - DO NOT REMOVE THE CASTING TO THE TYPES str(),float(),int()
        #       CLR CAN'T CAST FROM np.str_, np.float32, np.int32 TO 
        #       System.String(), System.Single(), System.Int()  

        Image_net.BandCompressionMode = bandCompressionMode
        Image_net.QuantificationParameters = quantificationParameters
        Image_net.Description = str(image.Description)
        Image_net.AddToHistory(str(image.History))
        Image_net.MmPixel = float(image.MmPixel)

        illuminations_objects = hf.illuminationList2Objects(image.Illumination)

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

        
        Image_net = hf.addAllAvailableImageLayers(Image_net, image)

    VMImIO.HipsIO.SaveImage(Image_net, str(path))

    Image_net.Free()

    if os.path.isfile(path):
        fullPath = os.path.abspath(path)
        print("HIPS image successfully written in " + fullPath)
        return fullPath
    else:
        print("Failed to write HIPS image")
        return None
    

def ReduceBands(imageClass, bandIndexesToUse=[]):
    """Reduces the bands in the imageClass to the ones who are in bandIndexesToUse
    
    Parameters:
    -----------
    imageClass - ImageClass object.
    
    bandIndexesToUse - list or numpy array
        Optional argument that corresponds to the bands that want to be shown.
    
    Outputs:
    --------
    imageClass - Same as input but only with bands in the bandIndexesToUse 
    """
    
    if type(imageClass) != ImageClass:
        raise TypeError("imageClass needs to be a ImageClass object")

    imageClass._ReduceBands(bandIndexesToUse)
    return imageClass


    
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
        it is excluded from documentation. The same will apply for the remaining
        show functions."""
    
    if (type(image)!=ImageClass) or (type(image)==np.ndarray and len(image.shape)==3):
        raise TypeError("image needs to be a ImageClass object or 3-D numpy array")
    
    plt_outputs=[]
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
        hf.checkIfbandIndexesToUseIsValid(bandIndexesToUse,imagearr.shape[2])
    else:
        bandIndexesToUse = list(range(imagearr.shape[2]))

    for i in bandIndexesToUse:
        plt.figure(num=i)
        v_min=np.min(imagearr[:,:,i])
        v_max=np.max(imagearr[:,:,i])
        ax_im=plt.imshow(imagearr[:,:,i],cmap='gray',vmin=v_min,vmax=v_max)
        plt_outputs.append(ax_im)
        plt.title("Band {}".format(i+1))
        plt.axis('off')

    if not ifOnlyGetListOfPLTObjects:
        plt.show(block=False)

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

    if (imageClass.RGBPixels is None):
        imageClass.To_sRGB()
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


# def __dir__():
#     return ["ImageClass","read","write","ReduceBands","show", "showRGB"]



