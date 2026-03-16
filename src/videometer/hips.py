import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tempfile
from videometer import vm_utils as utils
from videometer import config

class ImageClass:
    """A class representing a Videometer HIPS image.

    This class provides an interface to access image data, metadata, and perform
    common operations like band reduction and sRGB conversion. It supports both
    'clr' and 'python' backends for reading and writing files.

    Attributes:
        PixelValues (np.ndarray): 3-D NumPy array of pixel values (height, width, bands).
        Height (int): Image height.
        Width (int): Image width.
        Bands (int): Number of spectral bands.
        MmPixel (float): Physical size of each pixel in millimeters.
        BandNames (np.ndarray): 1-D array of band names.
        WaveLengths (np.ndarray): 1-D array of wavelengths.
        Illumination (np.ndarray): 1-D array of illumination names for each band.
        StrobeTimes (np.ndarray): 1-D array of strobe times.
        History (str): History log of the image.
        Description (str): Description of the image.
        FreehandLayers (List[dict]): List of freehand annotation layers.
        RGBPixels (np.ndarray): sRGB representation of the image (after calling `to_sRGB`).
        ForegroundPixels (np.ndarray, optional): Binary mask for foreground pixels.
        DeadPixels (np.ndarray, optional): Binary mask for dead pixels.
        CorrectedPixels (np.ndarray, optional): Binary mask for corrected pixels.
        SaturatedPixels (np.ndarray, optional): Binary mask for saturated pixels.
        ExtraData (dict): Dictionary for numeric extra metadata.
        ExtraDataInt (dict): Dictionary for integer extra metadata.
        ExtraDataString (dict): Dictionary for string extra metadata.
    """

    def __init__(
        self,
        path,
        bandIndexesToUse=[],
        ifSkipReadingAllLayers=False,
        ifSkipReadingFreehandLayer=False,
    ):
        """Initializes an ImageClass object by reading a HIPS file.

        Args:
            path (str): Path to the .hips file.
            bandIndexesToUse (List[int], optional): Bands to load.
            ifSkipReadingAllLayers (bool, optional): Skip metadata masks.
            ifSkipReadingFreehandLayer (bool, optional): Skip freehand layers.
        """
        self.PixelValues = None
        self.Height = 0
        self.Width = 0
        self.Bands = 0
        self.BandNames = None
        self.Illumination = None
        self.WaveLengths = None
        self.StrobeTimes = None
        self.StrobeTimesUniversal = None
        self.MmPixel = 0.0
        self.History = ""
        self.Description = ""
        self.ImageFileName = ""
        self.FullPathToImage = ""
        self.FreehandLayers = None
        self.ForegroundPixels = None
        self.DeadPixels = None
        self.SaturatedPixels = None
        self.CorrectedPixels = None
        self.RGBPixels = None
        self.ExtraData = dict()
        self.ExtraDataInt = dict()
        self.ExtraDataString = dict()
        
        self._BandCompressionModeObject = None
        self._QuantificationParametersObject = None

        if config.get_backend() == "clr":
            self._init_clr(path, bandIndexesToUse, ifSkipReadingAllLayers, ifSkipReadingFreehandLayer)
        else:
            self._init_python(path, bandIndexesToUse, ifSkipReadingAllLayers, ifSkipReadingFreehandLayer)

    def _init_clr(self, path, bandIndexesToUse, ifSkipReadingAllLayers, ifSkipReadingFreehandLayer):
        from videometer import vm_utils_clr
        import VM.Image.IO as VMImIO
        import VM.Image as VMIm
        import clr
        
        VMImageObject = VMImIO.HipsIO.LoadImage(path)
        self.PixelValues = vm_utils_clr.vmImage2npArray(VMImageObject)

        (self.Height, self.Width, self.Bands) = self.PixelValues.shape

        self.Bands = int(VMImageObject.Bands)
        self.BandNames = np.array(
            [str(bandname) for bandname in VMImageObject.BandNames]
        )
        self.Illumination = vm_utils_clr.illuminationObjects2List(VMImageObject.Illumination)
        self.WaveLengths = vm_utils_clr.asNumpyArray(VMImageObject.WaveLengths)
        self.StrobeTimes = vm_utils_clr.asNumpyArray(VMImageObject.StrobeTimes)
        self.StrobeTimesUniversal = vm_utils_clr.asNumpyArray(
            VMImageObject.StrobeTimesUniversal
        )

        self._BandCompressionModeObject = VMImageObject.BandCompressionMode
        self._QuantificationParametersObject = VMImageObject.QuantificationParameters
        self.MmPixel = VMImageObject.MmPixel
        self.History = VMImageObject.History
        self.Description = VMImageObject.Description
        self.ImageFileName = os.path.basename(path)
        self.FullPathToImage = os.path.abspath(path)

        for i in VMImageObject.ExtraData.Keys:
            self.ExtraData[i] = VMImageObject.ExtraData[i]

        for i in VMImageObject.ExtraDataInt.Keys:
            self.ExtraDataInt[i] = VMImageObject.ExtraDataInt[i]

        for i in VMImageObject.ExtraDataString.Keys:
            self.ExtraDataString[i] = VMImageObject.ExtraDataString[i]

        if not ifSkipReadingAllLayers:
            self._ReadAllImageLayers_clr(VMImageObject, ifSkipReadingFreehandLayer)

        if len(bandIndexesToUse) != 0:
            self.reduceBands(bandIndexesToUse)

    def _init_python(self, path, bandIndexesToUse, ifSkipReadingAllLayers, ifSkipReadingFreehandLayer):
        from videometer.hips_core import HipsImage
        
        img = HipsImage.read(path)
        self.PixelValues = img.pixels
        self.Height = img.height
        self.Width = img.width
        self.Bands = img.bands
        self.BandNames = np.array(img.band_names)
        self.Illumination = np.array(img.illumination_names)
        self.WaveLengths = img.wavelengths
        self.StrobeTimes = img.strobe_times
        self.StrobeTimesUniversal = img.strobe_times_universal
        
        self.MmPixel = img.mm_pixel
        self.History = img.history
        self.Description = img.description
        self.ImageFileName = os.path.basename(path)
        self.FullPathToImage = os.path.abspath(path)
        
        self.ExtraData = img.extra_data.copy()
        self.ExtraDataInt = img.extra_data_int.copy()
        self.ExtraDataString = img.extra_data_string.copy()
        
        # Quantification parameters in python backend are stored in the HipsImage object
        # we don't have a direct equivalent of the CLR _QuantificationParametersObject 
        # but we can store them if needed. For now, let's just keep them in HipsImage.
        self._python_hips_image = img 

        # TODO: Implement layer reading in hips_core if needed
        if not ifSkipReadingAllLayers:
             # hips_core currently doesn't support all layers (Corrected, Dead, etc.)
             # but we can at least handle what it has
             pass

        if len(bandIndexesToUse) != 0:
            self.reduceBands(bandIndexesToUse)

    def _ReadAllImageLayers_clr(self, VMImageObject, ifSkipReadingFreehandLayer):
        from videometer import vm_utils_clr
        import VM.Image as VMIm
        
        getImageLayer = VMIm.ImageLayerExtensions.GetImageLayer

        # Freehand Layer
        if not ifSkipReadingFreehandLayer:
            self._ReadFreehand_clr(VMImageObject.FreehandLayersXML)

        # CorrectedPixels
        self.CorrectedPixels = vm_utils_clr.imageLayer2npArray(
            getImageLayer(VMImageObject, "CorrectedPixels")
        )

        # DeadPixels
        self.DeadPixels = vm_utils_clr.imageLayer2npArray(
            getImageLayer(VMImageObject, "DeadPixels")
        )

        # Attempt to load foreground pixels from blob image
        try:
            from VM.Blobs import BlobImage
            blobImage = BlobImage.CreateFromXmlAndCreateMaskImage(VMImageObject.History, VMImageObject.ImageWidth, VMImageObject.ImageHeight)
            VMIm.ForegroundPixelsLayerHelperMethods.SetForegroundPixelsImageLayer(VMImageObject, blobImage.MaskImage)
        except:
            pass

        # ForegroundPixels
        self.ForegroundPixels = vm_utils_clr.imageLayer2npArray(
            getImageLayer(VMImageObject, "ForegroundPixels")
        )

        # SaturatedPixels
        self.SaturatedPixels = vm_utils_clr.imageLayer2npArray(
            getImageLayer(VMImageObject, "SaturatedPixels")
        )

    def _ReadFreehand_clr(self, freehandLayersXMLstring):
        from videometer import vm_utils_clr
        import VM.FreehandLayer as VMFreehand
        import clr
        import System.IO
        import System.Drawing
        
        freehandLayersIO = VMFreehand.FreehandLayerIO.DeserializeFromString(
            freehandLayersXMLstring
        )
        if freehandLayersIO is None:
            return

        freehandLayerList = []
        for container in freehandLayersIO.containers:
            ms = System.IO.MemoryStream(container.pixels)
            bitmap = System.Drawing.Bitmap(ms)
            npArray = vm_utils_clr.systemDrawingBitmap2npArray(bitmap)

            if len(npArray.shape) == 3:
                npArray = npArray[:, :, 0]

            # Scale down to binary
            npArray = npArray / np.max([np.max(npArray), 1])

            # Add to the freehandLayerDict with the same key as in VideometerLab software
            freehandObject = {
                "name": "Layer " + str(container.layerId + 1),
                "layerId": container.layerId,
                "description": container.description,
                "pixels": npArray,
            }
            freehandLayerList.append(freehandObject)

            bitmap.Dispose()
            ms.Dispose()

        self.FreehandLayers = freehandLayerList

    def to_sRGB(self, spectraName="D65", useMask=False):
        """Performs conversion of the spectral image to sRGB image.

        Parameters:
        -----------
        spectraName - string
            Name of the spectra to be used in the sRGB conversion. Default
            spectraName is 'D65'.
        useMask - boool
            Whether to apply the foreground mask
        Output :
            returns the sRGB image and updates the "RGBPixels" attribute"""

        if config.get_backend() == "python":
            # hips_core doesn't have to_sRGB yet.
            raise NotImplementedError("to_sRGB is not yet implemented for the 'python' backend.")

        from videometer import vm_utils_clr
        import VM.Image.ViewTransforms as VMImTransForms
        
        SpectraNamesLUT = vm_utils_clr.get_SpectraNamesLUP()
        if not (spectraName in SpectraNamesLUT):
            raise NotImplementedError(
                'spectraName="'
                + spectraName
                + '" is not implemented. \nList of implemented spectras: '
                + str(list(SpectraNamesLUT.keys()))
            )

        # Take only the wavelengths with the right illumination

        allowableIlluminations = [
            "Diffused_Highpower_LED",
            "Diffused_Lowpower_LED",
            "Direct_Lowpower_LED",
            "Coaxial_FrontLight",
        ]

        diffusedMask = np.isin(self.Illumination, allowableIlluminations)

        # Create a new VM object to parse through the SRGBViewTransform with only the visable wavelengths
        visibleMask = (380 <= self.WaveLengths) * (self.WaveLengths <= 780)

        visibleBands = diffusedMask & visibleMask

        if np.sum(visibleBands) < 3:
            raise TypeError(
                "Image class needs to have 3 or more wavelengths on the visable spectrum (380mm <= wavelength <= 780mm). Number of visable wavelength in ImageClass : "
                + str(np.sum(visibleBands))
            )
        VMImageObject = vm_utils_clr.npArray2VMImage(self.PixelValues[:, :, visibleBands])

        # Add attributes that are checked in IsValidFor()
        VMImageObject.AddToHistory(self.History)
        indexVisableBands = np.where(visibleBands)[0]
        for i in range(len(indexVisableBands)):
            VMImageObject.WaveLengths[i] = float(self.WaveLengths[indexVisableBands[i]])

        # Converter initialization and check
        converter = VMImTransForms.MultiBand.SrgbViewTransform()
        if not converter.IsValidFor(VMImageObject):
            raise TypeError(
                "VM.Image.NaturalColorConversion.InvalidConversionException: Only reflectance calibrated images are supported"
            )

        # Convert to sRGB image
        bitmap = converter.GetBitmap(VMImageObject, SpectraNamesLUT[spectraName])
        srgbImage = vm_utils_clr.systemDrawingBitmap2npArray(bitmap).astype(np.uint8)

        imrgb = np.empty_like(srgbImage)
        if useMask:
            if self.ForegroundPixels is None:
                raise AttributeError("ForegroundPixels attribute not set")

            for i in range(3):
                imrgb[:, :, i] = np.multiply(
                    srgbImage[:, :, i], self.ForegroundPixels
                )
        else:
            imrgb = srgbImage
        self.RGBPixels = imrgb

        VMImageObject.Free()

        return imrgb

    def reduceBands(self, bandIndexesToUse):
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

        utils.checkIfbandIndexesToUseIsValid(bandIndexesToUse, self.Bands)
        self.PixelValues = self.PixelValues[:, :, bandIndexesToUse]
        self.Bands = len(bandIndexesToUse)
        self.BandNames = self.BandNames[bandIndexesToUse]
        self.WaveLengths = self.WaveLengths[bandIndexesToUse]
        self.StrobeTimes = self.StrobeTimes[bandIndexesToUse]
        self.Illumination = self.Illumination[bandIndexesToUse]
        self.StrobeTimesUniversal = self.StrobeTimesUniversal[bandIndexesToUse]

        if config.get_backend() == "clr":
            import clr
            import VM.Image as VMIm
            if self._QuantificationParametersObject is None:
                return
            tmp = clr.System.Array.CreateInstance(
                VMIm.Compression.QuantificationParameters, len(bandIndexesToUse)
            )
            for i, bandIndexToUse in enumerate(bandIndexesToUse):
                tmp[i] = self._QuantificationParametersObject[bandIndexToUse]
            self._QuantificationParametersObject = tmp
        else:
            if hasattr(self, '_python_hips_image'):
                self._python_hips_image.reduce_bands(list(bandIndexesToUse))

    @staticmethod
    def from_bytes(bytes) -> "ImageClass":
        # Create a temporary file. 
        # delete=False is required so the file persists for the caller to use.
        with tempfile.NamedTemporaryFile(delete_on_close=False, suffix='.hips', mode='wb') as tmp_file:
            tmp_file.write(bytes)
            tmp_file.close()
            img = ImageClass(tmp_file.name)
            return img


# ---------------- Start of functions ----------------


def read(
    path,
    bandIndexesToUse=[],
    ifSkipReadingAllLayers=False,
    ifSkipReadingFreehandLayer=False,
):
    """Reads a HIPS image and stores it as an ImageClass object.

    Args:
        path (str): Full path to the .hips image.
        bandIndexesToUse (List[int], optional): List of band indexes to read.
            If empty, all bands are read.
        ifSkipReadingAllLayers (bool, optional): If True, skip reading metadata layers
            like CorrectedPixels, DeadPixels, etc. Defaults to False.
        ifSkipReadingFreehandLayer (bool, optional): If True, skip reading Freehand layers.
            Defaults to False.

    Returns:
        ImageClass: An initialized ImageClass object.

    Raises:
        TypeError: If path is not a string.
        ValueError: If path doesn't end with .hips.
        FileNotFoundError: If the file doesn't exist.
    """

    if type(path) != str:
        raise TypeError("path needs to be of type str")
    if not path.endswith(".hips"):
        raise ValueError("File needs to contain the .hips extension :" + path)
    if not os.path.isfile(path):
        raise FileNotFoundError("Couldn't locate " + path)

    return ImageClass(
        path, bandIndexesToUse, ifSkipReadingAllLayers, ifSkipReadingFreehandLayer
    )


def write(image, path, compression="SameAsImageClass", verbose=False):
    """Writes a HIPS image from an ImageClass object or a NumPy array.

    Args:
        image (ImageClass or np.ndarray): The image data to write.
        path (str): Target file path (must end in .hips).
        compression (str, optional): Compression level. One of:
            'SameAsImageClass', 'Uncompressed', 'VeryHighQuality',
            'HighQuality', 'HighCompression', 'VeryHighCompression'.
            Defaults to 'SameAsImageClass'.
        verbose (bool, optional): If True, print status messages. Defaults to False.

    Returns:
        str: Absolute path to the written file if successful, else None.

    Raises:
        TypeError: If path doesn't end in .hips or image type is invalid.
        FileNotFoundError: If the target directory doesn't exist.
        ValueError: If layer dimensions don't match pixel data.
    """

    if not (path.endswith(".hips")):
        raise TypeError("File needs to contain the .hips extension :" + path)
    folderPath = os.path.dirname(os.path.abspath(path))
    if not (os.path.isdir(folderPath)):
        raise FileNotFoundError(
            "The folder structure not found under " + path + " : " + folderPath
        )

    if config.get_backend() == "python":
        return _write_python(image, path, compression, verbose)
    else:
        return _write_clr(image, path, compression, verbose)

def _write_python(image, path, compression, verbose):
    from videometer.hips_core import HipsImage, COMPRESSION_PRESETS
    
    if isinstance(image, np.ndarray):
        img_obj = HipsImage()
        img_obj.pixels = image
    elif isinstance(image, ImageClass):
        img_obj = HipsImage()
        img_obj.pixels = image.PixelValues
        img_obj.history = image.History
        img_obj.description = image.Description
        img_obj.mm_pixel = image.MmPixel
        img_obj.band_names = list(image.BandNames)
        img_obj.wavelengths = image.WaveLengths
        img_obj.strobe_times = image.StrobeTimes
        img_obj.strobe_times_universal = image.StrobeTimesUniversal
        
        # Mapping illumination names back to IDs
        from videometer.hips_core import ILLUMINATION_TYPES
        reverse_illum = {v: k for k, v in ILLUMINATION_TYPES.items()}
        img_obj.illumination = np.array([reverse_illum.get(name, 0) for name in image.Illumination])
        
        img_obj.extra_data = image.ExtraData.copy()
        img_obj.extra_data_int = image.ExtraDataInt.copy()
        img_obj.extra_data_string = image.ExtraDataString.copy()
        
    else:
        raise TypeError("Image input has to be either ImageClass object or 3-D NumPy array")

    if compression == "SameAsImageClass":
        compression = None
        
    img_obj.write(path, compression)
    
    if os.path.isfile(path):
        fullPath = os.path.abspath(path)
        if verbose:
            print("HIPS image successfully written (python backend) in " + fullPath)
        return fullPath
    return None

def _write_clr(image, path, compression, verbose):
    from videometer import vm_utils_clr
    import VM.Image as VMIm
    import VM.Image.IO as VMImIO
    import clr
    
    if (type(image) == np.ndarray) and (len(image.shape) == 3):
        imagearr = image
        if compression == "SameAsImageClass":
            compression = "Uncompressed"
    elif type(image) == ImageClass:
        imagearr = image.PixelValues

        # Check if all Image layers match the size of the image.
        h, w, _ = imagearr.shape

        if not (image.CorrectedPixels is None) and (
            image.CorrectedPixels.shape != (h, w)
        ):
            raise ValueError(
                "CorrectedPixels "
                + str(image.CorrectedPixels.shape)
                + " and PixelValues "
                + str((h, w))
                + " shape do not match"
            )
        if not (image.DeadPixels is None) and (image.DeadPixels.shape != (h, w)):
            raise ValueError(
                "DeadPixels "
                + str(image.DeadPixels.shape)
                + " and PixelValues "
                + str((h, w))
                + " shape do not match"
            )
        if not (image.ForegroundPixels is None) and (
            image.ForegroundPixels.shape != (h, w)
        ):
            raise ValueError(
                "ForegroundPixels "
                + str(image.ForegroundPixels.shape)
                + " and PixelValues "
                + str((h, w))
                + " shape do not match"
            )
        if not (image.SaturatedPixels is None) and (
            image.SaturatedPixels.shape != (h, w)
        ):
            raise ValueError(
                "SaturatedPixels "
                + str(image.SaturatedPixels.shape)
                + " and PixelValues "
                + str((h, w))
                + " shape do not match"
            )

        if not (image.FreehandLayers is None):
            for i, freehand in enumerate(image.FreehandLayers):
                if freehand["pixels"].shape != (h, w):
                    raise ValueError(
                        "FreehandLayers "
                        + str(freehand["pixels"].shape)
                        + " at i="
                        + str(i)
                        + " and PixelValues "
                        + str((h, w))
                        + " shape do not match"
                    )

    else:
        raise TypeError(
            "Image input has to be either ImageClass object or 3-D NumPy array"
        )

    if compression != "SameAsImageClass":
        compressionLUT = vm_utils_clr.get_CompressionAndQuantificationPresetLUT()

        if not compression in compressionLUT:
            listOfValid = list(compressionLUT.keys())
            listOfValid.append("SameAsImageClass")
            raise NotImplementedError(
                compression
                + " is not implemented. \nList of valid : "
                + str(listOfValid)
            )

        # CompressionMode on the Compression Preset Object is set to RAW ... which will fail so this is fixed by setting it to None
        if compression == "Uncompressed":
            bandCompressionMode = None
        else:
            bandCompressionMode = compressionLUT[
                compression
            ].CompressionParameters.CompressionMode

        quantValue = compressionLUT[compression].QuantificationParameters
        bands = imagearr.shape[2]
        quantificationParameters = clr.System.Array.CreateInstance(
            VMIm.Compression.QuantificationParameters, bands
        )
        for i in range(bands):
            quantificationParameters[i] = quantValue

    else:
        bandCompressionMode = image._BandCompressionModeObject
        quantificationParameters = image._QuantificationParametersObject

    Image_net = vm_utils_clr.npArray2VMImage(imagearr)

    if type(image) == ImageClass:
        # NOTE - DO NOT REMOVE THE CASTING TO THE TYPES str(),float(),int()
        #       CLR CAN'T CAST FROM np.str_, np.float32, np.int32 TO
        #       System.String(), System.Single(), System.Int()

        if bandCompressionMode is not None:
            import VM.Image.Compression as VMComp
            Image_net.BandCompressionMode = VMComp.BandCompressionMode(int(bandCompressionMode))
        else:
            Image_net.BandCompressionMode = None

        Image_net.QuantificationParameters = quantificationParameters
        Image_net.Description = str(image.Description)
        Image_net.AddToHistory(str(image.History))
        Image_net.MmPixel = float(image.MmPixel)

        illuminations_objects = vm_utils_clr.illuminationList2Objects(image.Illumination)
        for i in range(image.Bands):
            Image_net.WaveLengths[i] = float(image.WaveLengths[i])
            Image_net.StrobeTimes[i] = int(image.StrobeTimes[i])
            Image_net.Illumination[i] = illuminations_objects[i]
            Image_net.BandNames[i] = str(image.BandNames[i])
            Image_net.StrobeTimesUniversal[i] = float(image.StrobeTimesUniversal[i])

        for k, v in image.ExtraData.items():
            Image_net.ExtraData[k] = float(v)
        for k, v in image.ExtraDataInt.items():
            Image_net.ExtraDataInt[k] = int(v)
        for k, v in image.ExtraDataString.items():
            Image_net.ExtraDataString[k] = str(v)

        Image_net = vm_utils_clr.addAllAvailableImageLayers(Image_net, image)

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


def show(image, ifUseMask=False, bandIndexesToUse=[], ifOnlyGetListOfPLTObjects=False):
    """Function that shows individual bands of the image.

    Args:
        image (ImageClass or np.ndarray): Image data to display.
        ifUseMask (bool, optional): If True, apply the foreground mask.
            Only works if image is an ImageClass object. Defaults to False.
        bandIndexesToUse (List[int], optional): List of band indexes to show.
            If empty, all bands are shown. Defaults to [].
        ifOnlyGetListOfPLTObjects (bool, optional): If True, return the matplotlib
            AxesImage objects instead of calling plt.show(). Defaults to False.

    Returns:
        List[matplotlib.image.AxesImage]: List of matplotlib image objects.
    """

    if (type(image) != ImageClass) and not (
        type(image) == np.ndarray and len(image.shape) == 3
    ):
        raise TypeError("image needs to be a ImageClass object or 3-D numpy array")

    if type(image) == ImageClass:
        imagearr = image.PixelValues.copy()
        if ifUseMask:
            if image.ForegroundPixels is None:
                raise AttributeError("ForegroundPixels attribute not set")

            for i in range(image.Bands):
                imagearr[:, :, i] = np.multiply(
                    imagearr[:, :, i], image.ForegroundPixels
                )

    else:
        imagearr = image
        if ifUseMask:
            raise TypeError(
                "image needs to be a ImageClass object to be able to use ForegroundMask"
            )

    if len(bandIndexesToUse) != 0:
        utils.checkIfbandIndexesToUseIsValid(bandIndexesToUse, imagearr.shape[2])
    else:
        bandIndexesToUse = list(range(imagearr.shape[2]))

    plt_outputs = []
    for i in bandIndexesToUse:
        plt.figure(num=i)
        v_min = np.min(imagearr[:, :, i])
        v_max = np.max(imagearr[:, :, i])
        ax_im = plt.imshow(imagearr[:, :, i], cmap="gray", vmin=v_min, vmax=v_max)
        plt_outputs.append(ax_im)
        plt.title("Band {}".format(i + 1))
        plt.axis("off")

    if not ifOnlyGetListOfPLTObjects:
        plt.show()

    return plt_outputs


def showRGB(img, ifUseMask=False):
    """Function that shows sRGB representation of the image.

    Note: This currently requires the 'clr' backend.

    Args:
        img (ImageClass): The image object to display.
        ifUseMask (bool, optional): If True, apply the foreground mask.
            Defaults to False.

    Returns:
        matplotlib.image.AxesImage: The matplotlib image object.
    """

    if type(img) != ImageClass:
        raise TypeError("imageClass needs to be a ImageClass object")

    img.to_sRGB(useMask=ifUseMask)

    plt.figure(num=1)
    ax_im = plt.imshow(img.RGBPixels)
    plt.title("RGB image")
    plt.axis("off")
    plt.show()
    return ax_im


def readOnlyPixelValues(path):
    """Function that reads the HIPS image and returns only its pixel values.

    Args:
        path (str): Full path to the .hips image.

    Returns:
        np.ndarray: A 3-D NumPy array of pixel values.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    if not path.endswith(".hips"):
        raise TypeError("File name has to have the .hips extension")

    if config.get_backend() == "python":
        from videometer.hips_core import HipsImage
        img = HipsImage.read(path)
        return img.pixels
    else:
        import VM.Image.IO as VMImIO
        from videometer import vm_utils_clr
        VMImageObject = VMImIO.HipsIO.LoadImage(path)
        npArray = vm_utils_clr.vmImage2npArray(VMImageObject)
        VMImageObject.Free()
        return npArray
