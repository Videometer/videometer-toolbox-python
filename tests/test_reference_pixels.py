import pytest
import os
import numpy as np
from videometer.hips_core import HipsImage

def test_pfrgb_reference_pixels():
    path = os.path.join("TestData", "test_PFRGB.hips")
    if not os.path.exists(path):
        pytest.skip(f"Test file {path} not found")
        
    img = HipsImage.read(path)
    
    # Check dimensions
    assert img.width == 252
    assert img.height == 182
    assert img.bands == 3
    
    # Check specific pixel values (reference values obtained empirically from current HipsImage implementation)
    # (y, x, band)
    
    # Corner pixel (0,0)
    np.testing.assert_array_equal(img.pixels[0, 0, :], [240, 240, 240])
    
    # Middle-ish pixel (100, 100)
    np.testing.assert_array_equal(img.pixels[100, 100, :], [255, 218, 149])
    
    # Another pixel (50, 200)
    np.testing.assert_array_equal(img.pixels[50, 200, :], [255, 255, 255])

def test_calibrated_reference_pixels():
    path = os.path.join("tests", "TestImages", "calibratedImage.hips")
    if not os.path.exists(path):
        pytest.skip(f"Test file {path} not found")
        
    img = HipsImage.read(path)
    
    # Check dimensions
    assert img.width == 3
    assert img.height == 3
    assert img.bands == 19
    
    # Check a specific pixel (center)
    # Values were obtained from the current python implementation
    expected_center_pixel = np.array([
        6.13183, 7.788884, 9.776011, 13.465557, 15.965685, 
        18.46778, 23.118782, 22.334417, 31.833454, 34.609097, 
        41.082695, 42.45146, 44.012646, 46.162968, 50.64141, 
        54.375626, 55.447056, 57.15812, 58.346397
    ], dtype=np.float32)
    
    np.testing.assert_allclose(img.pixels[1, 1, :], expected_center_pixel, rtol=1e-5)

def test_everything_uncompressed_reference_pixels():
    path = os.path.join("tests", "TestImages", "TestEverythingImage_Uncompressed.hips")
    if not os.path.exists(path):
        pytest.skip(f"Test file {path} not found")
        
    img = HipsImage.read(path)
    
    # Check dimensions
    assert img.width == 3
    assert img.height == 2
    assert img.bands == 19
    
    # Band 0 has values [[0, 1, 2], [3, 4, 5]]
    # This is a synthetic image for testing
    expected_band_0 = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
    np.testing.assert_allclose(img.pixels[:, :, 0], expected_band_0, rtol=1e-5)
    
    # Verification of a few other bands being present
    assert img.pixels.shape == (2, 3, 19)

def test_everything_high_quality_reference_pixels():
    path = os.path.join("tests", "TestImages", "TestEverythingImage_HighQuality.hips")
    if not os.path.exists(path):
        pytest.skip(f"Test file {path} not found")
        
    img = HipsImage.read(path)
    
    # Check dimensions
    assert img.width == 3
    assert img.height == 2
    assert img.bands == 19
    
    # Band 0 has quantized values that should be approximately [[0, 1, 2], [3, 4, 5]]
    # These are the actual values after quantization/de-quantization
    expected_band_0 = np.array([
        [0.0, 1.0557185, 1.994135],
        [3.0498536, 3.98827, 5.0439887]
    ], dtype=np.float32)
    np.testing.assert_allclose(img.pixels[:, :, 0], expected_band_0, rtol=1e-5)
