import pytest
import os
import sys
import numpy as np
import glob

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

from videometer.hips_core import HipsImage
from videometer.hips import ImageClass

# Locate test files in TestData/compression
TEST_DATA_DIR = os.path.join("TestData", "compression")
TEST_FILES = glob.glob(os.path.join(TEST_DATA_DIR, "*.hips"))

def get_compression_mode_name(format_code):
    """Maps HIPS format code to BandCompressionMode string."""
    if format_code & 0x80:
        return "GZ"
    if format_code & 0x100:
        return "JPEG"
    if format_code & 0x200:
        return "PNG"
    return "RAW"

@pytest.mark.parametrize("filepath", TEST_FILES)
def test_compression_quantization_parity(filepath):
    """
    Compares compression and quantization parameters between 
    the legacy (pythonnet) and pure Python implementations.
    """
    if not os.path.exists(filepath):
        pytest.skip(f"Test file {filepath} not found")

    # Load with pure Python implementation
    test_subject = HipsImage.read_header(filepath)
    
    # Load with legacy Oracle
    try:
        oracle = ImageClass(filepath)
    except Exception as e:
        pytest.fail(f"Legacy Oracle failed to load {filepath}: {e}")

    try:
        # 1. Compare Compression Mode
        oracle_compression = str(oracle._BandCompressionModeObject)
        subject_compression = get_compression_mode_name(test_subject.format)
        
        assert subject_compression == oracle_compression, f"Compression mode mismatch in {filepath}"

        # 2. Compare Quantification Parameters
        oracle_quant = oracle._QuantificationParametersObject
        subject_quant = test_subject._quantization_parameters
        
        if oracle_quant is None:
            assert subject_quant is None, f"Expected no quantization in {filepath}, but found some."
        else:
            assert subject_quant is not None, f"Expected quantization in {filepath}, but found none."
            assert len(subject_quant) == len(oracle_quant), f"Quantization parameter count mismatch in {filepath}"
            
            for i in range(len(oracle_quant)):
                o_qp = oracle_quant[i]
                s_qp = subject_quant[i]
                
                assert int(o_qp.Q) == s_qp.Q, f"Q mismatch in band {i} of {filepath}"
                assert pytest.approx(float(o_qp.Q_Min)) == s_qp.Q_Min, f"Q_Min mismatch in band {i} of {filepath}"
                assert pytest.approx(float(o_qp.Q_Max)) == s_qp.Q_Max, f"Q_Max mismatch in band {i} of {filepath}"

        # 3. Compare Pixels (to ensure the de-quantization logic works as expected)
        # We only do this if it's a small file or we want to be thorough.
        # For quantization tests, it's essential.
        # Note: We need to load pixels for the test_subject.
        test_subject.load_pixels()
        np.testing.assert_allclose(test_subject.pixels, oracle.PixelValues, rtol=1e-5, atol=1e-5, 
                                   err_msg=f"Pixel values mismatch in {filepath} after de-quantization")

    finally:
        pass

if __name__ == "__main__":
    pytest.main([__file__])
