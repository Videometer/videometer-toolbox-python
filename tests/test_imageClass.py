# DLL paths and the pythonnet runtime are initialized once in tests/conftest.py.
import videometer.hips as hips


def test_load_foreground_mask_from_blob():
    path = r"TestData\1c8f82ed-2ede-48c7-a0be-4978f282a6ea.hips";
    img = hips.ImageClass(path)
    img.to_sRGB()

def test_optical_filters_are_read_per_band():
    # Plain reflectance bands have no optical filter (None); filtered bands give a dict
    # with type and pass band (see ImageClass.OpticalFilters).
    img = hips.ImageClass(r"TestData\1c8f82ed-2ede-48c7-a0be-4978f282a6ea.hips")
    assert len(img.OpticalFilters) == img.Bands
    assert all(f is None for f in img.OpticalFilters) # No optical filters

    img.reduceBands([0, 2])
    assert len(img.OpticalFilters) == 2