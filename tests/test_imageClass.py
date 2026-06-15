# DLL paths and the pythonnet runtime are initialized once in tests/conftest.py.
import videometer.hips as hips


def test_load_foreground_mask_from_blob():
    path = r"TestData\1c8f82ed-2ede-48c7-a0be-4978f282a6ea.hips";
    img = hips.ImageClass(path)
    img.to_sRGB()
