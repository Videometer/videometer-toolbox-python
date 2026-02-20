from videometer.new import HipsImage
from videometer import hips
import matplotlib.pyplot as plt

path = r"C:\Users\heh\repos\videometer-toolbox-python\TestData\1c8f82ed-2ede-48c7-a0be-4978f282a6ea.hips"
path = r"C:\Users\heh\Documents\Videometer\PersonalWorkspaces\dev\Images\Image_20260130_090542_615.hips"
path = r"C:\Users\heh\repos\videometer-toolbox-python\tests\TestImages\TestEverythingImage_Uncompressed.hips"
path = r"C:\Users\heh\Documents\Videometer\PersonalWorkspaces\dev\Images\washers.hips"
img = HipsImage(path)

if False:
    print(img.image_data)

    plt.imshow(img.image_data[0, :, :], cmap='gray')
    plt.show()

import matplotlib.pyplot as plt
import math

def show_all_bands(hips_img, cols=4):
    data = hips_img.image_data
    bands = hips_img.bands
    
    # Calculate required rows for the grid
    rows = math.ceil(bands / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten() # Flatten to 1D array for easy iteration
    
    for i in range(bands):
        # Extract the 2D slice for the current band
        # Data is stored band-sequentially
        band_slice = data[i, :, :]
        
        im = axes[i].imshow(band_slice, cmap='gray', interpolation='nearest')
        axes[i].set_title(f"Band {i+1}")
        axes[i].axis('off')
        
        # Optional: Add individual colorbars to see relative intensity
        # fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Hide unused subplots if bands < rows*cols
    for j in range(bands, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

# Usage
show_all_bands(img)