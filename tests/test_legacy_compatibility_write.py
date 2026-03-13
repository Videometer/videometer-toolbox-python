import pytest
import os
import sys
import numpy as np
import tempfile
from videometer.hips_core import HipsImage, HipsFormat, COMPRESSION_PRESETS

# Initialize DLL paths and pythonnet (copied from hips.py)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
VMPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VMPATH_SRC = os.path.join(VMPATH, "src", "videometer")
IPP_PATH = os.path.join(VMPATH_SRC, "DLLs", "IPP2019Update1", "intel64")
DLL_PATH = os.path.join(VMPATH_SRC, "DLLs", "VM")

os.environ["PATH"] = IPP_PATH + ";" + os.environ["PATH"]
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(IPP_PATH)
    os.add_dll_directory(DLL_PATH)
sys.path.append(DLL_PATH)

import pythonnet
if pythonnet.get_runtime_info() is None:
    pythonnet.load("coreclr")
import clr

clr.AddReference("VM.Image")

from videometer.hips import ImageClass

import uuid

def create_synthetic_image(width, height, bands, dtype=np.float32):
    """Creates a synthetic image with a gradient for testing."""
    img = np.zeros((height, width, bands), dtype=dtype)
    # Use a larger range for float32 to test quantization robustly (matches Q_Max=120)
    scale = 100.0 if dtype == np.float32 else 1.0
    for b in range(bands):
        # Create a gradient that varies by band
        y, x = np.mgrid[0:height, 0:width]
        img[:, :, b] = ((x + y * width) / (width * height) * (b + 1) / bands) * scale
    return img

# Test Matrix: (Pixel Format, Compression Preset, Tolerance)
# Tolerances are adjusted for a [0, 100] range
TEST_MATRIX = [
    # Lossless / Raw
    (np.float32, "Uncompressed", 1e-5),
    (np.uint8, None, 0), # PFBYTE Raw
    (np.int16, None, 0), # PFSHORT Raw
    
    # Quantized Presets (Step size = 120 / (2^Q - 1))
    (np.float32, "VeryHighQuality", 0.05),   # 12-bit PNG (step ~0.03)
    (np.float32, "HighQuality", 0.15),       # 10-bit PNG (step ~0.11)
    (np.float32, "HighCompression", 0.5),    # 8-bit PNG (step ~0.47)
    (np.float32, "VeryHighCompression", 1.0), # 8-bit JPEG (Lossy)
]

@pytest.mark.parametrize("dtype, compression, tol", TEST_MATRIX)
def test_write_compatibility_roundtrip(dtype, compression, tol):
    """
    Tests that files written by the new implementation can be 
    correctly opened by the legacy Oracle.
    """
    width, height, bands = 32, 32, 3
    source_pixels = create_synthetic_image(width, height, bands, dtype)
    
    fd, tmp_path = tempfile.mkstemp(suffix=".hips")
    os.close(fd)
        
    try:
        # 1. Create and Write with New Implementation
        img = HipsImage()
        img.pixels = source_pixels
        img.id = str(uuid.uuid4())
        img.write(tmp_path, compression=compression)
        
        # 2. Read back with Legacy Oracle
        try:
            oracle = ImageClass(tmp_path)
        except Exception as e:
            pytest.fail(f"Legacy Oracle failed to load file written by Python: {e}")
            
        try:
            # 3. Assert Metadata Parity
            assert oracle.Width == width
            assert oracle.Height == height
            assert oracle.Bands == bands
            
            # Verify Compression Mode Mapping
            if compression:
                preset = COMPRESSION_PRESETS[compression]
                # Map format to BandCompressionMode string
                fmt = preset["format"]
                expected_mode = "RAW"
                if fmt & 0x80: expected_mode = "GZ"
                if fmt & 0x100: expected_mode = "JPEG"
                if fmt & 0x200: expected_mode = "PNG"
                
                assert str(oracle._BandCompressionModeObject) == expected_mode
            
            # 4. Assert Pixel Parity
            if tol == 0:
                np.testing.assert_array_equal(oracle.PixelValues, source_pixels.astype(np.float32 if dtype == np.float32 else dtype))
            else:
                # Normalization check for quantified data
                # Quantization usually maps to float32 even if input was something else
                np.testing.assert_allclose(oracle.PixelValues, source_pixels, atol=tol, rtol=tol)
                
        finally:
            # No Free() method on ImageClass wrapper, but we rely on GC or manual if existed
            pass
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    pytest.main([__file__])
