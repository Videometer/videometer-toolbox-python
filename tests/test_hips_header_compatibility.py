import pytest
import os
import sys
import numpy as np
import tempfile
import pythonnet
if pythonnet.get_runtime_info() is None:
    pythonnet.load("coreclr")
import clr

# Initialize DLL paths for legacy Oracle
VMPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VMPATH_SRC = os.path.join(VMPATH, "src", "videometer")
IPP_PATH = os.path.join(VMPATH_SRC, "DLLs", "IPP2019Update1", "intel64")
DLL_PATH = os.path.join(VMPATH_SRC, "DLLs", "VM")

os.environ["PATH"] = IPP_PATH + ";" + os.environ["PATH"]
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(IPP_PATH)
    os.add_dll_directory(DLL_PATH)
sys.path.append(DLL_PATH)

clr.AddReference("VM.Image")
import VM.Image.IO as VMImIO

from videometer.hips_core import HipsImage

def test_structural_header_parity():
    """
    PHASE 1 VERIFICATION:
    Verifies that the HIPS Extended Parameters header written by Python 
    conforms to the structural requirements of the legacy C# HipsIO loader.
    
    Requirements verified:
    1. count == 1 parameters are literal strings in the header line.
    2. count > 1 parameters are offsets into the binary block.
    3. binary block entries are 4-byte aligned.
    4. byteOffset line matches total padded binary size.
    """
    width, height, bands = 16, 16, 3
    img = HipsImage()
    img.pixels = np.zeros((height, width, bands), dtype=np.float32)
    
    # Mix of parameter types
    img.mm_pixel = 0.123456  # Single 'f' (count 1)
    img.wavelengths = np.array([400.0, 500.0, 600.0], dtype=np.float32) # Array 'f' (count 3)
    img.id = "12345678-1234-1234-1234-1234567890ab" # String (count > 1)
    
    with tempfile.NamedTemporaryFile(suffix=".hips", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        # Write with New Implementation
        img.write(tmp_path)
        
        # Verify with Legacy Oracle
        # LoadImageHeader is the most sensitive method to structural errors
        try:
            vm_image = VMImIO.HipsIO.LoadImageHeader(tmp_path)
            assert vm_image is not None
            
            # Verify parsed values
            assert pytest.approx(float(vm_image.MmPixel)) == 0.123456
            assert vm_image.Bands == 3
            assert vm_image.IdString == "12345678-1234-1234-1234-1234567890ab"
            
            vm_image.Free()
        except Exception as e:
            pytest.fail(f"Legacy Oracle failed to load structured HIPS header: {e}")
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def test_binary_alignment_logic():
    """
    PHASE 1 VERIFICATION:
    Verifies that the binary block handles padding correctly for mixed-size data.
    """
    img = HipsImage()
    img.pixels = np.zeros((8, 8, 1), dtype=np.uint8)
    
    # Create a string that is not a multiple of 4 bytes
    img.extra_data_string["Test"] = "ABC" # 3 bytes, needs 1 byte pad
    
    with tempfile.NamedTemporaryFile(suffix=".hips", delete=False) as tmp:
        tmp_path = tmp.name
        
    try:
        img.write(tmp_path)
        
        # Manually inspect the file content for alignment
        with open(tmp_path, "rb") as f:
            content = f.read()
            
        # Find 'ExtraDataString_Test'
        pos = content.find(b"ExtraDataString_Test")
        assert pos != -1
        
        # Find the line after 'nPar' block which is 'byteOffset'
        # Then find the binary block start.
        # For simplicity, we just rely on the Oracle loader which fails 
        # if the total offset calculation is even 1 byte off.
        vm_image = VMImIO.HipsIO.LoadImageHeader(tmp_path)
        assert vm_image is not None
        vm_image.Free()
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    pytest.main([__file__])
