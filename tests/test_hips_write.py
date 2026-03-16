import pytest
import os
import sys
import numpy as np
from videometer.hips_core import HipsImage, COMPRESSION_PRESETS

# Initialize DLL paths and pythonnet (copied from hips.py)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
VMPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
VMPATH_SRC = os.path.join(VMPATH, "src", "videometer")
IPP_PATH = os.path.join(VMPATH_SRC, "DLLs", "IPP2019Update1", "intel64")
DLL_PATH = os.path.join(VMPATH_SRC, "DLLs", "VM")

os.environ["PATH"] = IPP_PATH + ";" + os.environ["PATH"]
sys.path.append(DLL_PATH)

import pythonnet
if pythonnet.get_runtime_info() is None:
    pythonnet.load("coreclr")
import clr

clr.AddReference("VM.Image")

def test_symmetrical_roundtrip_uncompressed(tmp_path):
# ... (rest of function unchanged)
    # 1. Create a dummy image
    height, width, bands = 10, 10, 3
    pixels = np.random.rand(height, width, bands).astype(np.float32)
    
    img = HipsImage(
        width=width, height=height, bands=bands,
        history="Pure Python Write Test",
        description="Testing Symmetry",
        mm_pixel=0.1
    )
    img._pixels = pixels
    
    # 2. Write
    test_file = tmp_path / "test_uncompressed.hips"
    img.write(str(test_file), compression="Uncompressed")
    
    # 3. Read back
    read_img = HipsImage.read(str(test_file))
    
    # 4. Assert
    assert read_img.width == width
    assert read_img.height == height
    assert read_img.bands == bands
    # Exact parity for uncompressed
    np.testing.assert_allclose(read_img.pixels, pixels, rtol=1e-5)

@pytest.mark.parametrize("preset", [
    "VeryHighQuality", 
    "HighQuality", 
    "HighCompression", 
    "VeryHighCompression"
])
def test_symmetrical_roundtrip_compressed(tmp_path, preset):
    height, width, bands = 20, 20, 2
    # Ensure some interesting data
    pixels = np.random.rand(height, width, bands).astype(np.float32) * 100.0
    
    img = HipsImage(
        width=width, height=height, bands=bands,
        history=f"Preset {preset} Test",
        mm_pixel=0.05
    )
    img._pixels = pixels
    
    test_file = tmp_path / f"test_{preset}.hips"
    img.write(str(test_file), compression=preset)
    
    # Read back
    read_img = HipsImage.read(str(test_file))
    
    # Assert metadata
    assert read_img.bands == bands
    
    # Assert pixels with appropriate tolerance
    # Quantization error for 8-bit (HighCompression) can be up to (Max-Min)/255 / 2
    # For 120.0 range, that's ~0.235.
    # JPEG (VeryHighCompression) is much lossier.
    if preset == "VeryHighCompression":
        atol = 10.0 # JPEG is very lossy
    elif preset == "HighCompression":
        atol = 0.5  # 8-bit quantization
    else:
        atol = 0.1  # 10 or 12-bit quantization
        
    np.testing.assert_allclose(read_img.pixels, pixels, atol=atol)

def test_static_oracle_validation(tmp_path):
    """
    Test using a static oracle: Load a known good file, 
    write it back, and check if legacy reader can still open it.
    """
    oracle_path = os.path.join("tests", "TestImages", "TestEverythingImage_Uncompressed.hips")
    if not os.path.exists(oracle_path):
        pytest.skip("Oracle file not found")
        
    img = HipsImage.read(oracle_path)
    clone_path = tmp_path / "clone.hips"
    img.write(str(clone_path), compression="Uncompressed")
    
    # Try to use legacy reader
    import VM.Image.IO as VMImIO
    try:
        vm_image = VMImIO.HipsIO.LoadImageHeader(str(clone_path))
        assert vm_image.ImageWidth == img.width
        assert vm_image.Bands == img.bands
        vm_image.Free()
    except Exception as e:
        pytest.fail(f"Legacy reader failed to open our written file: {e}")
