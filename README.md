# Videometer toolbox for Python 

A toolbox for multispectral .HIPS images from Videometer A/S

<br>

## Installation 
  
System requirements: 
- Windows operating system
- 64-bit Intel processor



Use the package manager [pip](https://pip.pypa.io/en/stable/) to install
```bash
pip install videometer
```


**Note : First time the videometer.hips is imported it will fetch DLLs**

<br>



## Usage - Read

### videometer.hips.read(path, bandIndexesToUse=[])  
&emsp;&emsp;  Reads a HIPS image and stores it as an ImageClass object.  

&emsp;&emsp;  *Parameters*  
&emsp;&emsp; &emsp;&emsp; **Input** :  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **path \<String>**: Path to the image.   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **bandIndexesToUse \<List of Ints>** (optional) List of the indexes who are suppose to be in      
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp; &emsp;&emsp; the returned ImageClass object. If empty then it will return all of them (default).   


&emsp;&emsp; &emsp;&emsp; **Output** :   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - ImageClass object 


ImageClass is explained below. 


### Example

```python
from videometer import hips

imageCls = hips.read("image.hips")
```

<br>


## Usage - readOnlyPixelValues

### videometer.hips.readOnlyPixelValues(path)  
&emsp;&emsp;  Reads a HIPS image and returns pixel values in a 3-D Numpy array.    
&emsp;&emsp;  Makes the reading quicker when only the pixel values are wanted.


&emsp;&emsp;  *Parameters*  
&emsp;&emsp; &emsp;&emsp; **Input** :  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **path \<String>**: Path to the image.


&emsp;&emsp; &emsp;&emsp; **Output** :   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - 3-D Numpy array 


### Example

```python
from videometer import hips

img = hips.readOnlyPixelValues("image.hips")
```

<br>

## Usage - Write

### videometer.hips.write(image, path, compression="SameAsImageClass", verbose=False)  
&emsp;&emsp;  Writes a HIPS image from an ImageClass object or a NumPy array that corresponds  
&emsp;&emsp; to the pixel values of a spectral image.  

&emsp;&emsp;  *Parameters*  
&emsp;&emsp; &emsp;&emsp; **Input** :  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **image \<ImageClass OR 3-D NumPy array>**:  The image to write.      
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **path \<String>**:  path of the to be written HIPS file.   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Note it has to include the .hips extension and a existing folder structure.  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **compression \<String>**:   Compression level of the image, same as in VideometerLab   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp; (Table can be seen below and more detailed one in VideometerLab under   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; File > Preferences > Compression ) :   
```
Level of Compression : {
    "SameAsImageClass" : Keep the same compression as is in the ImageClass 
                        (if it is a numpy array then it will be Uncompressed) 
    "Uncompressed" : No compression
                    (Same as Original in VideometerLab software),
    "VeryHighQuality" : (see VideometerLab software),
    "HighQuality" : (see VideometerLab software),
    "HighCompression" : (see VideometerLab software),
    "VeryHighCompression" : (see VideometerLab software)
}
```
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **verbose \<Boolean>**:   If true then prints out the name of the file otherwise not.   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; &emsp;&emsp;  Default is false   

&emsp;&emsp; &emsp;&emsp; **Output** :   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - Returns a full path to the written .HIPS image if successful otherwise None.



### Compression Preset Table

Preset  | Storage | Reflectance Precision | Typical Image Size
------------- | ------------- | ------------- | ------------- 
Uncompressed  | RAW (32bit float) |  N/A | 912MB (100%) | 
Very High Quality  | 12 bit PNG | 0.03 | 171MB (19%)  |  
High Quality  | 10 bit PNG | 0.12 | 159MB (17%) | 
High Compression  | 8 bit PNG | 0.47 | 110MB (10%)  | 
Very High Compression  | 8 bit JPEG | 0.47 + Edge artifacts | 31MB (3%)| 
### Example - Writing ImageClass

```python
from videometer import hips

imageCls = hips.read("image.hips")
fullPath2Image = hips.write(imageCls,"image.hips",compression="HighCompression")

if fullPath2Image is None:
    print("FAILED")
else:
    print("SUCCESS!")
```

### Example - Writing NumPy array
```python
from videometer import hips
import numpy as np

npArray = np.array(
    [[0.,1.,2.],
    [3.,4.,5.],
    [6.,7.,8.]], dtype=np.float32)
npArray = npArray.reshape((3,3,1)) # Height, Width, Bands

fullPath2Image = hips.write(npArray,"imageFromNumpyArray.hips")

if fullPath2Image is None:
    print("Failed")
else:
    print("Success!")
```


<br>

## Usage - Show

### videometer.hips.show(  image, ifUseMask=False, bandIndexesToUse=[], ifOnlyGetListOfPLTObjects=False):  
&emsp;&emsp;  Function that shows images of individual bands.   

&emsp;&emsp;  *Parameters*  
&emsp;&emsp; &emsp;&emsp; **Input** :    
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **image \<ImageClass OR 3-D NumPy array>**:  ImageClass object or NumPy array (3-D).   

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **ifUseMask \<Boolean>** If set to true and mask is set on the ImageClass object then the image will show masked.  

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **bandIndexesToUse \<List of Ints>** (optional) : List of Indexes of the bands to be shown.   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; If empty then it will return all of them (default).   

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **ifOnlyGetListOfPLTObjects  \<Boolean>** (Optional): If set to True it won't plot the images (Mainly used in testing)


&emsp;&emsp; &emsp;&emsp; **Output** :     
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - Shows the images  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - Returns the list of the matplotlib.image.AxesImage objects (Mostly used in testing)



<br>

## Usage - Show RGB

### videometer.hips.showRGB(imageClass, ifUseMask=False):  
&emsp;&emsp;  Function that shows srgb representation of the image.  
&emsp;&emsp; If ifUseForegroundMask is true then the image will be shown masked.  

&emsp;&emsp; *Parameters*    
&emsp;&emsp; &emsp;&emsp; **Input** :    
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **image \<ImageClass>**:    ImageClass object to be shown.    
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **ifUseMask \<Boolean>**:  To toggle mask on or off if set on the ImageClass object.  



&emsp;&emsp; &emsp;&emsp; **Output** :   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - Shows the images  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - Returns the matplotlib.image.AxesImage object (Mostly used for testing). 







<br>



## ImageClass
The center of the toolbox. 

### Attributes 

- **PixelValues** - NumPy array of floats (3-D)  
 &emsp;&emsp; Contains float pixel values of the HIPS image. Shape of the array is (height, width, bands).

- **Height** - Int  
  &emsp;&emsp;   Height of the HIPS image.

- **Width** - Int  
  &emsp;&emsp;   Width of the HIPS image.

- **MmPixel** - Float    
&emsp;&emsp;     Physical size of each pixel in mm.
    
- **Bands** - Int    
&emsp;&emsp; Number of bands in the image.

- **BandNames** - NumPy array of strings    
&emsp;&emsp; List of names of the bands.

- **WaveLengths** - NumPy array of floats  
&emsp;&emsp;     Contains wavelenghts of the bands in HIPS image.

- **Description** - String  
  &emsp;&emsp;   Description set of the image

- **History** - String   
  &emsp;&emsp;   Explains the history of the image.
 
- **Illumination** – NumPy array of strings  
&emsp;&emsp;     List of Illumination name type of each band.
        
- **StrobeTimes** – NumPy array of int  
  &emsp;&emsp;   Strobe time in ms of each band in the image.

- **StrobeTimesUniversal** - NumPy array of floats  
 &emsp;&emsp;    Universal strobe time of each band in the image.

- **FreehandLayers** – List of dictionaries  
&emsp;&emsp;  Each set Freehand layer is a dictionary with the following template :
```
{
    "name" : <string>
    "layerId" : <int>
    "description" : <string>
    "pixels" :  <2-D numpy array>
}    
```    

- **RGBPixels** – NumPy array (height, width, 3)  
 &emsp;&emsp;    Array representing sRGB pixel values of the image.   
&emsp;&emsp;   The values are set by using to_sRGB method in the ImageClass otherwise it will return None 

- **ForegroundPixels** – NumPy array (2-D)  
  &emsp;&emsp;   Foreground mask of the image given as a binary 2-D numpy array.   
  &emsp;&emsp;  If Foreground pixels are not set on the image object it will return None. 

- **DeadPixels** – NumPy array (2-D)  
  &emsp;&emsp;   Dead pixels of the image given as a binary 2-D numpy arra.    
  &emsp;&emsp;  If Dead pixels are not set on the image object it will return None.  

- **CorrectedPixels** – NumPy array (2-D)  
 &emsp;&emsp;    Corrected pixels of the image given as a binary 2-D numpy array.  
 &emsp;&emsp;  If Corrected pixels are not set on the image object it will return None. 

- **SaturatedPixels** – NumPy array (2-D)  
 &emsp;&emsp;    Saturated pixels of the image given as a binary 2-D numpy arra.   
 &emsp;&emsp;  If Saturated pixels are not set on the image object it will return None. 

- **ExtraData** – Dictionary \<string> : \<Float>  
  &emsp;&emsp;   Contains additional information about the image (e.g. temperature 
    data and similar). 

- **ExtraDataInt** – Dictionary  \<string> : \<Int>  
  &emsp;&emsp;   Contains additional information about the image.

- **ExtraDataString** – Dictionary \<string> : \<string>  
   &emsp;&emsp;  Contains additional information about the image. 





### Functions
- **To_sRGB**(spectraName="D65")  
&emsp;&emsp;  Performs conversion of the spectral image to sRGB image.   

&emsp;&emsp;  *Parameters*  
&emsp;&emsp; &emsp;&emsp; **Input** :  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **SpectraName (optional) \<String>** : Name of the spectra used for the transformation, default is D65.   

&emsp;&emsp; &emsp;&emsp; **Output** :   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - Returns the numpy sRGB image  \<NumPy array 2-D>  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - Updates the "RGBPixels" attribute

    
- **extractBands**(bandIndexesToUse)  
&emsp;&emsp;    Similar to the Image Tools > Conversion > Extract bands in VideometerLab software.   
&emsp;&emsp;  The bands and their information given in the bandsIndexesToUse remain in the ImageClass,   
&emsp;&emsp; others are deleted.

&emsp;&emsp;  *Parameters*  
&emsp;&emsp; &emsp;&emsp; **Input** :  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; - **bandIndexesToUse \<List of Ints>**: List of indexes of the bands to remain in the class.

&emsp;&emsp; &emsp;&emsp; **Output** : None




<br>

## Bugs or suggestions

Suggestions and bug reports may be sent to asc@videometer.com or jmk@videometer.com 
 

Enjoy!


## Licence 

[BSD-3-Clause-Clear license](LICENSE.TXT)
