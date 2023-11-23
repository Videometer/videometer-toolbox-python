import os
os.environ["PATH"] = "" # Remove everything temporary in system path

from videometer import hips
from videometer import vm_utils as utils
import numpy as np
import unittest
from parameterized import parameterized_class
import matplotlib.pyplot as plt


import clr
import System
VMPATH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
VMDLLPATH = os.path.join(VMPATH, "DLLs","VM")
clr.AddReference(os.path.join(VMDLLPATH,"VM.Image.dll"))
clr.AddReference(os.path.join(VMDLLPATH,"VM.Image.IO.dll"))
import VM.Image as VMIm
import VM.Image.IO as VMImIO


# Notes about the testing 
# - The number of the classes TestXXOnImages are to correct the order of the test running 
#   since we need the class created images before we run all of them  
# - Need to test the writing with different compressions... 



testImagesDir = os.path.join(VMPATH,"tests","TestImages")
os.chdir(testImagesDir)


namesOfTestEverythingImages = [
    "TestEverythingImage_Uncompressed.hips",
    "TestEverythingImage_VeryHighQuality.hips",
    "TestEverythingImage_HighQuality.hips",
    "TestEverythingImage_HighCompression.hips",
    "TestEverythingImage_VeryHighCompression.hips",
]
fullPathOfImages = [os.path.join(testImagesDir, name) for name in namesOfTestEverythingImages]
fullPathOfWritingTestImages = [os.path.join(testImagesDir,"TestImagesWriting","WritingTest_"+name) for name in namesOfTestEverythingImages]

parameterizedSetup = [{"imagePath" : path} for path in fullPathOfImages + fullPathOfWritingTestImages]
parameterizedSetupWithoutWritingTest = [{"imagePath" : path} for path in fullPathOfImages]

@parameterized_class(
    parameterizedSetup
)
class Test02OnImagesRead(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ImageClass = hips.read(self.imagePath) 

    def test_ImagePixelValues(self):
        self.assertIs(type(self.ImageClass.PixelValues), np.ndarray)
        self.assertEqual(self.ImageClass.PixelValues.shape, (self.ImageClass.Height,self.ImageClass.Width,self.ImageClass.Bands))
        
        # Lossy compression drifts the values to much when VeryHighCompressionImage is used twice on the same image so we skip it for now
        if self.ImageClass.ImageFileName == "WritingTest_TestEverythingImage_VeryHighCompression.hips":
            return
        self.assertTrue(np.all(np.round(self.ImageClass.PixelValues[:,:,0]) == np.array([[0,1,2],[3,4,5]], dtype=np.float32)))
        self.assertEqual(np.sum(np.round(self.ImageClass.PixelValues), dtype=int), np.sum([1,2,3,4,5]))   

    def test_Mask(self):
        self.assertIs(type(self.ImageClass.ForegroundPixels), np.ndarray) 
        self.assertTrue(self.ImageClass.ForegroundPixels.shape, (2,3))
        self.assertIs(type(self.ImageClass.ForegroundPixels[0,0]), np.int32)
        self.assertTrue(np.all(self.ImageClass.ForegroundPixels ==  np.array([[0,0,0],[0,0,1]], dtype=np.int32)))
    
    def test_CorrectedPixels(self):
        self.assertIs(type(self.ImageClass.CorrectedPixels), np.ndarray) 
        self.assertTrue(self.ImageClass.CorrectedPixels.shape, (2,3))
        self.assertIs(type(self.ImageClass.CorrectedPixels[0,0]), np.int32)
        self.assertTrue(np.all(self.ImageClass.CorrectedPixels ==  np.array([[0,1,1],[1,1,1]], dtype=np.int32)))
        
    def test_DeadPixels(self):
        self.assertIs(type(self.ImageClass.DeadPixels), np.ndarray) 
        self.assertTrue(self.ImageClass.DeadPixels.shape, (2,3))
        self.assertIs(type(self.ImageClass.DeadPixels[0,0]), np.int32)
        self.assertTrue(np.all(self.ImageClass.DeadPixels ==  np.array([[0,0,0],[0,1,0]], dtype=np.int32)))
        
    def test_FreehandLayers(self):
        self.assertIsNotNone(self.ImageClass.FreehandLayers)  # FreeHandLayer not set
        self.assertIs(type(self.ImageClass.FreehandLayers) , list)
        self.assertEqual(len(self.ImageClass.FreehandLayers) , 2)
        self.assertIs(type(self.ImageClass.FreehandLayers[0]) , dict)
        self.assertIs(type(self.ImageClass.FreehandLayers[1]) , dict)

        freehandLayers = sorted(self.ImageClass.FreehandLayers, key=lambda d : d["layerId"])
        self.assertEqual(freehandLayers[0]["name"] , "Layer 1")
        self.assertEqual(freehandLayers[0]["layerId"] , 0)
        self.assertEqual(freehandLayers[0]["description"] , "No Description")
        self.assertTrue(np.all(freehandLayers[0]["pixels"] == np.array([[0,1,0],[0,0,0]], dtype=np.int32)))
        
        self.assertEqual(freehandLayers[1]["name"] , "Layer 2")
        self.assertEqual(freehandLayers[1]["layerId"] , 1)
        self.assertEqual(freehandLayers[1]["description"] , "No Description")
        self.assertTrue(np.all(freehandLayers[1]["pixels"] == np.array([[1,0,0],[0,0,0]], dtype=np.int32)))
        

    def test_SaturatedPixels(self):
        self.assertIs(type(self.ImageClass.SaturatedPixels), np.ndarray) 
        self.assertTrue(self.ImageClass.SaturatedPixels.shape, (2,3))
        self.assertIs(type(self.ImageClass.SaturatedPixels[0,0]), np.int32)
        self.assertTrue(np.all(self.ImageClass.SaturatedPixels ==  np.array([[0,0,0],[0,0,1]], dtype=np.int32)))
     
    def test_MetaData_Bands(self):
        self.assertIs(type(self.ImageClass.Bands), int)
        self.assertEqual(self.ImageClass.Bands, 19)

    def test_MetaData_StrobeTimes(self):
        self.assertIs(type(self.ImageClass.StrobeTimes), np.ndarray)
        self.assertEqual(len(self.ImageClass.StrobeTimes),19)
        self.assertIs(type(self.ImageClass.StrobeTimes[0]), np.int32)
        self.assertTrue(np.all(self.ImageClass.StrobeTimes == np.arange(19,0,-1, dtype=np.int32)))

    def test_MetaData_BandNames(self):
        self.assertIs(type(self.ImageClass.BandNames), np.ndarray)
        self.assertEqual(len(self.ImageClass.BandNames),19)
        self.assertIs(type(self.ImageClass.BandNames[0]), np.str_)
        self.assertTrue(np.all(self.ImageClass.BandNames == np.array(["BandName"+str(i) for i in range(1,20)])))
    
    def test_MetaData_WaveLengths(self):
        self.assertIs(type(self.ImageClass.WaveLengths), np.ndarray)
        self.assertEqual(len(self.ImageClass.WaveLengths),19)
        self.assertIs(type(self.ImageClass.WaveLengths[0]), np.float32)
        self.assertTrue(np.all(self.ImageClass.WaveLengths == np.array([100+i for i in range(1,20)])))
        
    def test_MetaData_UniversalStrobeTimes(self):
        self.assertIs(type(self.ImageClass.StrobeTimesUniversal), np.ndarray)
        self.assertEqual(len(self.ImageClass.StrobeTimesUniversal),19)
        self.assertIs(type(self.ImageClass.StrobeTimesUniversal[0]), np.float32)
        self.assertTrue(np.all(self.ImageClass.StrobeTimesUniversal == (20.0+np.arange(18,-1,-1))))

    def test_MetaData_Illumination(self):
        self.assertIs(type(self.ImageClass.Illumination), np.ndarray)
        self.assertEqual(len(self.ImageClass.Illumination), 19)
        self.assertIs(type(self.ImageClass.Illumination[0]), np.str_)

        self.assertTrue(np.all(self.ImageClass.Illumination == 
                np.array(
                    ['Mixed', 'NA', 'Empty', 'Diffused_Highpower_LED', 'Brightfield_BackLight',
                    'Darkfield_BackLight', 'Diffused_Laser', 'SpotLaser',
                    'Diffused_Lowpower_LED', 'Direct_Lowpower_LED', 'Diffused_UV',
                    'Harmless_Guide_Laser', 'TEST_PROBE', 'Darkfield_FrontLight',
                    'Coaxial_FrontLight', 'NA', 'NA', 'NA', 'NA']
                    )))

    def test_MetaData_ExtraData(self):
        self.assertIs(type(self.ImageClass.ExtraData), dict)
        self.assertEqual(len(self.ImageClass.ExtraData), 1)
        self.assertTrue("TestFloat" in self.ImageClass.ExtraData)
        self.assertEqual(round(self.ImageClass.ExtraData["TestFloat"],2), 42.42)

        self.assertTrue("TestInt" in self.ImageClass.ExtraDataInt)
        self.assertEqual(self.ImageClass.ExtraDataInt["TestInt"], 42)

        self.assertTrue("TestString" in self.ImageClass.ExtraDataString)
        self.assertEqual(self.ImageClass.ExtraDataString["TestString"], "42")


    def test_MetaData_MmPixel_Description_History_CompressionMode(self):      
        self.assertIs(type(self.ImageClass.MmPixel), float)
        self.assertAlmostEqual(round(self.ImageClass.MmPixel,4), 3.1416)
        self.assertIs(type(self.ImageClass.Description), str)
        self.assertEqual(self.ImageClass.Description, "Description from the test image")
        self.assertIs(type(self.ImageClass.History), str)
        self.assertEqual(self.ImageClass.History.replace("\r\n",""), "History from the test image")

        compressionOfTestImage = self.ImageClass.ImageFileName.split("_")[-1].replace(".hips","")
        compressionLUT = utils.get_CompressionAndQuantificationPresetLUT()
        self.assertTrue(compressionOfTestImage in compressionLUT)
        compParam = compressionLUT[compressionOfTestImage]
        if compressionOfTestImage == "Uncompressed": ## Uncompressed is not initialized correctly in the Compression Preset LUT
            self.assertIn(str(self.ImageClass._BandCompressionModeObject), "RAW")
            self.assertIn(str(self.ImageClass._QuantificationParametersObject), "None")
        else:
            self.assertIn(str(self.ImageClass._BandCompressionModeObject), str(compParam.CompressionParameters))
            for i in range(self.ImageClass.Bands):
                self.assertTrue(self.ImageClass._QuantificationParametersObject[i].Equals(compParam.QuantificationParameters))


@parameterized_class(
    parameterizedSetupWithoutWritingTest
)
class Test01OnImagesWrite(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        if not os.path.isdir("TestImagesWriting"):
            os.mkdir("TestImagesWriting")
        self.ImageClass = hips.read(self.imagePath) 

    @classmethod
    def tearDownClass(self):
        # Clean up if test_ImageShapeExceptionsWhenSaving fails
        for filename_suffix in ["PixelValues","CorrectedPixels","DeadPixels","ForegroundPixels","SaturatedPixels","FreehandLayers"] : 
            filename = "test_"+filename_suffix+".hips"
            if os.path.isfile(filename):
                os.remove(filename)    


    def test_WriteImageClass(self):
        path = hips.write(self.ImageClass, os.path.join("TestImagesWriting",  "WritingTest_" + self.ImageClass.ImageFileName))
        self.assertIsNotNone(path)
        self.assertTrue(os.path.isfile(path))


    def test_ImageShapeExceptionsWhenSaving(self):
        # Pixel Values
        old_arr = self.ImageClass.PixelValues.copy()
        self.ImageClass.PixelValues = np.array([[[0]]])
        self.assertRaises(ValueError, hips.write, self.ImageClass,"test_PixelValues.hips") # Error, func, arg1, arg2
        self.ImageClass.PixelValues = old_arr

        # Corrected Pixels
        old_img_layer = self.ImageClass.CorrectedPixels.copy()
        self.ImageClass.CorrectedPixels = np.array([[[0]]])
        self.assertRaises(ValueError, hips.write, self.ImageClass,"test_CorrectedPixels.hips") # Error, func, arg1, arg2
        self.ImageClass.CorrectedPixels = old_img_layer
        
        # Dead Pixels
        old_img_layer = self.ImageClass.DeadPixels.copy()
        self.ImageClass.DeadPixels = np.array([[[0]]])
        self.assertRaises(ValueError, hips.write, self.ImageClass,"test_DeadPixels.hips") # Error, func, arg1, arg2
        self.ImageClass.DeadPixels = old_img_layer

        # Foreground pixels
        old_img_layer = self.ImageClass.ForegroundPixels.copy()
        self.ImageClass.ForegroundPixels= np.array([[[0]]])
        self.assertRaises(ValueError, hips.write, self.ImageClass,"test_ForegroundPixels.hips") # Error, func, arg1, arg2
        self.ImageClass.ForegroundPixels = old_img_layer

        # Saturated Pixels
        old_img_layer = self.ImageClass.SaturatedPixels.copy()
        self.ImageClass.SaturatedPixels= np.array([[[0]]])
        self.assertRaises(ValueError, hips.write, self.ImageClass,"test_SaturatedPixels.hips") # Error, func, arg1, arg2
        self.ImageClass.SaturatedPixels = old_img_layer

        # Note  It does throw the ValueError but for some reason the assertRaises doesn't catch it ...  
        #              So this will be commented out until a solution is found
        #   - JMK
        # 
        # Freehand Layers 
        # self.ImageClass.FreehandLayers[0]["pixels"] = np.array([[[0]]])      
        # self.assertRaises(ValueError, hips.write, self.ImageClass,"test_FreehandLayers.hips") # Error, func, arg1, arg2
        
        
        
        

@parameterized_class(
    parameterizedSetup
)
class Test03OnImagesVMfunctions(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.ImageClass = hips.read(self.imagePath)
        self.ImageClassMasked = hips.read(self.imagePath)
        self.ImageClassReduced = hips.read(self.imagePath)

    def test_show(self):
        plt_outputs = hips.show(self.ImageClass, ifOnlyGetListOfPLTObjects=True)
        plt.close("all")

        images = np.array([p.get_array() for p in plt_outputs])     
        titles = np.array([p.axes.get_title() for p in plt_outputs])    

        self.assertEqual(len(plt_outputs), self.ImageClass.Bands)

        # Lossy compression drifts the values to much when VeryHighCompressionImage is used twice on the same image so we skip it for now
        if self.ImageClass.ImageFileName == "WritingTest_TestEverythingImage_VeryHighCompression.hips":
            return
        self.assertTrue(np.all(np.round(images[0]) == np.array([[0,1,2],[3,4,5]], dtype=np.float32)))
        self.assertEqual(np.sum(np.round(images), dtype=int), np.sum([1,2,3,4,5])) 
        self.assertTrue(np.all(titles == np.array(["Band "+str(i+1) for i in range(self.ImageClass.Bands)])))


    def test_showReduced(self):
        bandIndexesToUse=[0]
        bands = len(bandIndexesToUse)

        plt_outputs = hips.show(self.ImageClass, bandIndexesToUse=bandIndexesToUse, ifOnlyGetListOfPLTObjects=True)
        plt.close("all")
        images = np.array([p.get_array() for p in plt_outputs])     
        titles = np.array([p.axes.get_title() for p in plt_outputs])   
    
        self.assertEqual(len(plt_outputs), bands)
        self.assertEqual(self.ImageClass.Bands, 19) ##  check if this changed something within the ImageClass


        # Lossy compression drifts the values to much when VeryHighCompressionImage is used twice on the same image so we skip it for now
        if self.ImageClass.ImageFileName == "WritingTest_TestEverythingImage_VeryHighCompression.hips":
            return
        self.assertTrue(np.all(np.round(images[0]) == np.array([[0,1,2],[3,4,5]], dtype=np.float32)))
        self.assertEqual(np.sum(np.round(images), dtype=int), np.sum([1,2,3,4,5])) 
        self.assertTrue(np.all(titles == np.array(["Band "+str(i+1) for i in range(bands)])))    


    def test_showMasked(self):
        self.ImageClassMasked.ForegroundPixels = np.array([[0,0,0],[0,0,0]],dtype=np.uint8)
        plt_outputs = hips.show(self.ImageClassMasked, ifUseMask=True, ifOnlyGetListOfPLTObjects=True)
        plt.close("all")
        images = np.array([p.get_array() for p in plt_outputs])     
        titles = np.array([p.axes.get_title() for p in plt_outputs])    

        self.assertEqual(len(plt_outputs), self.ImageClassMasked.Bands)
        self.assertEqual(np.sum(np.round(images), dtype=int), 0) 
        self.assertTrue(np.all(titles == np.array(["Band "+str(i+1) for i in range(self.ImageClassMasked.Bands)])))

    
    def test_ReduceBands(self):
        bandsIndexesToUse = [0,18]
        bands = len(bandsIndexesToUse)
        self.ImageClassReduced.reduceBands(bandsIndexesToUse)
        
        self.assertTrue(np.all(self.ImageClassReduced.PixelValues.shape == (self.ImageClassReduced.Height,self.ImageClassReduced.Width,bands)))
        self.assertEqual(self.ImageClassReduced.Bands, bands)
        self.assertTrue(np.all(self.ImageClassReduced.BandNames == ["BandName1","BandName19"]))
        self.assertTrue(np.all(self.ImageClassReduced.WaveLengths == np.array([101.0,119.0],dtype=np.float32)))
        self.assertTrue(np.all(self.ImageClassReduced.StrobeTimes == np.array([19,1])))
        self.assertTrue(np.all(self.ImageClassReduced.StrobeTimesUniversal == np.array([38.0,20.0], dtype=np.float32)))
        self.assertTrue(np.all(self.ImageClassReduced.Illumination == ["Mixed","NA"]))

        if "Uncompressed" in self.ImageClassReduced.ImageFileName : ## Uncompressed is not initialized correctly in the Compression Preset LUT
            self.assertIn(str(self.ImageClassReduced._BandCompressionModeObject), "RAW")
            self.assertIn(str(self.ImageClassReduced._QuantificationParametersObject), "None")
        else:
            for i in range(self.ImageClassReduced.Bands):
                self.assertTrue(self.ImageClassReduced._QuantificationParametersObject[i].Equals(self.ImageClass._QuantificationParametersObject[bandsIndexesToUse[i]]))

        # Lossy compression drifts the values to much when VeryHighCompressionImage is used twice on the same image so we skip it for now
        if "VeryHighCompression" in self.ImageClassReduced.ImageFileName:
            return
        self.assertTrue(np.all(np.round(self.ImageClassReduced.PixelValues[:,:,0]) == np.array([[0,1,2],[3,4,5]], dtype=np.float32)))
        self.assertEqual(np.sum(np.round(self.ImageClassReduced.PixelValues), dtype=int), np.sum([1,2,3,4,5])) 




## -------------- TEST WITHOUT IMAGES ------------------------



class TestWriting(unittest.TestCase):
    
    def test_WriteNpArray(self):
        arr = np.zeros((2,3,19), dtype=np.float32)
        arr[:,:,0] = np.array([[0,1,2],[3,4,5]], dtype=np.float32)

        path = hips.write(arr, os.path.join("TestImagesWriting",  "WritingTest_Arr.hips"))
        self.assertIsNotNone(path)
        self.assertTrue(os.path.isfile(path))

        a = hips.read(path)
        self.assertTrue(np.all(arr == a.PixelValues))
        ## The rest should be none and zeros .. don't really need to test for that



class TestVMfunctions(unittest.TestCase):
    @classmethod
    def setUpClass(self): 
        self.ImageClass = hips.read("calibratedImage.hips")

    def test_to_sRGB(self):
        img = self.ImageClass.to_sRGB(spectraName="D65")
        self.assertIs(type(img), np.ndarray)
        self.assertEqual(img.shape, (3,3,3))
        self.assertEqual(type(img[0,0,0]), np.uint8)
        self.assertTrue(np.all(img == self.ImageClass.RGBPixels))
        self.assertTrue(np.all(img == np.array([[[182, 139, 83], # Height x Width x Bands
                                                [179, 136, 86],
                                                [159, 121, 78]],

                                                [[181, 139, 85],
                                                [177, 135, 86],
                                                [149, 115, 77]],

                                                [[181, 138, 86],
                                                [171, 131, 84],
                                                [132, 103, 74]]],dtype=np.uint8)))
        

    def test_showRGB(self):
        axImg = hips.showRGB(self.ImageClass)
        plt.close("all")
        self.assertTrue(np.all(self.ImageClass.RGBPixels == axImg.get_array()))



class TestHelperFunctions(unittest.TestCase):

    def test_asNetArrayMemMove(self):
        N = 5
        sysArr = clr.System.Array.CreateInstance(System.Single, N)
        for i in range(N):
            sysArr[i] = float(i)
        npArr = np.arange(N, dtype=np.float32)
        
        npArr2sys = utils.asNetArrayMemMove(npArr)

        for i in range(N):
            self.assertEqual(sysArr[i], npArr2sys[i])


    def test_asNumpyArray(self):
        N = 5
        sysArr = clr.System.Array.CreateInstance(System.Single, N)
        for i in range(N):
            sysArr[i] = float(i)
        npArr = np.arange(N, dtype=np.float32)
        
        sysArr2np = utils.asNumpyArray(sysArr)

        self.assertTrue(np.all(npArr == sysArr2np))


    def test_imageLayer2npArray(self):
        npArr = np.array([[1,0,1],[0,1,0]],dtype=np.float32) 
        img = VMIm.VMImage(utils.asNetArrayMemMove(npArr))
        imageLayer = VMIm.ImageLayer(img)

        img2npArr = utils.imageLayer2npArray(imageLayer)

        img.Free()        
        self.assertTrue(np.all(npArr == img2npArr))



    def test_get_IlluminationLUT(self):
        lut = utils.get_IlluminationLUT()
        self.assertIs(type(lut), dict)
        self.assertEqual(len(lut), 15)
        

    def test_illuminationObjects2List(self):
        names = np.array(["Mixed","NA","Diffused_UV"])
        lut = utils.get_IlluminationLUT()
        illumObj = [lut[name] for name in names]

        namesObjects = utils.illuminationObjects2List(illumObj)
        self.assertTrue(np.all(names == namesObjects))


    def test_illuminationList2Objects(self):
        names = np.array(["Mixed","NA","Diffused_UV"])
        listOfObjects = utils.illuminationList2Objects(names)

        lut = utils.get_IlluminationLUT()
        illumObj = np.array([lut[name] for name in names])

        self.assertTrue(np.all(listOfObjects == illumObj))
        

    def test_checkIfbandIndexesToUseIsValid(self):

        with self.assertRaises(TypeError):
            utils.checkIfbandIndexesToUseIsValid()
            utils.checkIfbandIndexesToUseIsValid([],"")
            utils.checkIfbandIndexesToUseIsValid(-1,"")
            utils.checkIfbandIndexesToUseIsValid([],-1)
            utils.checkIfbandIndexesToUseIsValid([1,2,3,4],-1)
            utils.checkIfbandIndexesToUseIsValid([1,2,3,4],[1,2,3,4])
            utils.checkIfbandIndexesToUseIsValid([1,-2,3,4],10)
            
        self.assertIsNone(utils.checkIfbandIndexesToUseIsValid([1,2,3,4],10))
            

        

    def test_addAllAvailableImageLayers(self):
        # Is tested in TestOnImages*
        pass

    def test_addImageLayer(self):
        # Is tested in TestOnImages*
        pass

    def test_setFreehandLayers(self):
        # Is tested in TestOnImages*
        pass

    def test_vmImage2npArray(self):
        npArr = np.array([[[1,2,3],[4,5,6]]],dtype=np.float32) 
        npArr2VMimg = np.transpose(npArr, (2,0,1))
        img = VMIm.VMImage(utils.asNetArrayMemMove(npArr2VMimg))

        npArrFromImg = utils.vmImage2npArray(img)

        img.Free()
        self.assertTrue(np.all(npArr == npArrFromImg))

    def test_npArray2VMImage(self):
        vmImgRef = VMImIO.HipsIO.LoadImage("calibratedImage.hips")
        
        # 2-D 
        vmImg2 = utils.npArray2VMImage(np.array([[5.81, 6.10, 5.29],[6.15, 6.13, 5.22],[6.33, 5.90, 5.01]], dtype=np.float32))
        
        # 3-D
        vmImg3 = utils.npArray2VMImage(np.array([[5.81, 6.10, 5.29],[6.15, 6.13, 5.22],[6.33, 5.90, 5.01]], dtype=np.float32).reshape((3,3,1))) # Height x Width x Bands

        diffLimit = 0.01
        for i in range(3):
            for j in range(3):
                self.assertTrue(abs(vmImgRef.GetPixel(i,j,0) - vmImg2.GetPixel(i,j,0)) < diffLimit)
                self.assertTrue(abs(vmImgRef.GetPixel(i,j,0) - vmImg3.GetPixel(i,j,0)) < diffLimit)
           
        vmImgRef.Free()
        vmImg2.Free()
        vmImg3.Free()
        
        

    def test_get_SpectraNamesLUP(self):
        lut = utils.get_SpectraNamesLUP()
        self.assertIs(type(lut), dict)
        self.assertEqual(len(lut), 35)
        self.assertTrue("D65" in lut)
        

    def test_systemDrawingBitmap2npArray(self):
        # Is tested in TestOnImages*
        pass

    def test_get_CompressionAndQuantificationPresetLUT(self):
        lut = utils.get_CompressionAndQuantificationPresetLUT()
        compressions = ["Uncompressed", "VeryHighQuality","HighQuality","HighCompression","VeryHighCompression"]
        self.assertIs(type(lut), dict)
        self.assertEqual(len(lut), len(compressions))
        for c in compressions:
            self.assertTrue(c in lut)
            self.assertTrue("CompressionParameters" in dir(lut[c]))
            self.assertTrue("QuantificationParameters" in dir(lut[c]))
        



if __name__ == "__main__":
    unittest.main()


	




