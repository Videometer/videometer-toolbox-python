import pytest
import os
import numpy as np
from videometer.hips_core import HipsImage, write as pure_write

# Setup paths
testImagesDir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "TestImages"))

namesOfTestEverythingImages = [
    "TestEverythingImage_Uncompressed.hips",
    "TestEverythingImage_VeryHighQuality.hips",
    "TestEverythingImage_HighQuality.hips",
    "TestEverythingImage_HighCompression.hips",
    "TestEverythingImage_VeryHighCompression.hips",
]

# We don't have all the WritingTest images yet, so let's stick to the main ones
@pytest.mark.parametrize("filename", namesOfTestEverythingImages)
class TestHipsCoreRead:
    @pytest.fixture(autouse=True)
    def setup(self, filename):
        self.imagePath = os.path.join(testImagesDir, filename)
        self.img = HipsImage.read(self.imagePath)

    def test_ImagePixelValues(self, filename):
        assert isinstance(self.img.pixels, np.ndarray)
        assert self.img.pixels.shape == (self.img.height, self.img.width, self.img.bands)

        # Lossy compression drifts values
        if "VeryHighCompression" in filename:
            return
            
        # The test image expects these specific values in the first band
        # [[0, 1, 2], [3, 4, 5]]
        np.testing.assert_allclose(
            np.round(self.img.pixels[:, :, 0]),
            np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32),
            atol=1e-5
        )
        
        # Test sum of all pixels
        # TestEverythingImage has values 0,1,2,3,4,5 in each band? 
        # Actually test_main.py says np.sum([1, 2, 3, 4, 5]) which is 15.
        # Each band seems to have sum 15.
        assert np.sum(np.round(self.img.pixels[:, :, 0])) == 15

    def test_MetaData_Bands(self):
        assert isinstance(self.img.bands, int)
        assert self.img.bands == 19

    def test_MetaData_StrobeTimes(self):
        assert isinstance(self.img.strobe_times, np.ndarray)
        assert len(self.img.strobe_times) == 19
        assert self.img.strobe_times.dtype == np.int32
        np.testing.assert_array_equal(
            self.img.strobe_times, 
            np.arange(19, 0, -1, dtype=np.int32)
        )

    def test_MetaData_BandNames(self):
        assert len(self.img.band_names) == 19
        expected = [f"BandName{i}" for i in range(1, 20)]
        assert self.img.band_names == expected

    def test_MetaData_WaveLengths(self):
        assert len(self.img.wavelengths) == 19
        np.testing.assert_allclose(
            self.img.wavelengths,
            np.array([100 + i for i in range(1, 20)], dtype=np.float32)
        )

    def test_MetaData_UniversalStrobeTimes(self):
        assert len(self.img.strobe_times_universal) == 19
        np.testing.assert_allclose(
            self.img.strobe_times_universal,
            (20.0 + np.arange(18, -1, -1)).astype(np.float32)
        )

    def test_MetaData_Illumination(self):
        expected_names = [
            "Mixed", "NA", "Empty", "Diffused_Highpower_LED",
            "Brightfield_BackLight", "Darkfield_BackLight",
            "Diffused_Laser", "SpotLaser", "Diffused_Lowpower_LED",
            "Direct_Lowpower_LED", "Diffused_UV", "Harmless_Guide_Laser",
            "TEST_PROBE", "Darkfield_FrontLight", "Coaxial_FrontLight",
            "NA", "NA", "NA", "NA"
        ]
        assert self.img.illumination_names == expected_names

    def test_MetaData_ExtraData(self):
        assert self.img.extra_data["TestFloat"] == pytest.approx(42.42)
        assert self.img.extra_data_int["TestInt"] == 42
        assert self.img.extra_data_string["TestString"] == "42"

    def test_MetaData_MmPixel_Description_History(self):
        assert self.img.mm_pixel == pytest.approx(3.141593, abs=1e-6)
        assert self.img.description == "Description from the test image"
        assert "History from the test image" in self.img.history

    def test_ReduceBands(self, filename):
        img_reduced = HipsImage.read(self.imagePath)
        bands_to_use = [0, 18]
        img_reduced.reduce_bands(bands_to_use)
        
        assert img_reduced.pixels.shape == (img_reduced.height, img_reduced.width, 2)
        assert img_reduced.bands == 2
        assert img_reduced.band_names == ["BandName1", "BandName19"]
        np.testing.assert_allclose(img_reduced.wavelengths, [101.0, 119.0])
        np.testing.assert_array_equal(img_reduced.strobe_times, [19, 1])
        assert img_reduced.illumination_names == ["Mixed", "NA"]

def test_WriteNpArray(tmp_path):
    arr = np.zeros((2, 3, 19), dtype=np.float32)
    arr[:, :, 0] = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)

    output_path = tmp_path / "WritingTest_Arr.hips"
    # We need a top-level write function like in hips.py
    from videometer.hips_core import write as pure_write
    pure_write(arr, str(output_path))
    
    assert os.path.isfile(output_path)
    
    img_read = HipsImage.read(str(output_path))
    np.testing.assert_allclose(img_read.pixels, arr)
