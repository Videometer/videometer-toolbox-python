# Design Plan: Pure Python HIPS Header Reader and Writer

## 1. Objective
Implement a pure-Python class `HipsImage` in `src/videometer/hips_core.py` that can read and write the HIPS file format header, matching the functionality of the legacy C#-based implementation while being Pythonic and clean.

## 2. Class Structure: `HipsImage`
The `HipsImage` class will store all metadata found in the HIPS header and extended parameters.
- **Standard Attributes**: `width`, `height`, `bands`, `pixel_format`, `history`, `description`, `roi_rect`.
- **Extended Attributes**: `mm_pixel`, `band_names`, `wavelengths`, `strobe_times`, `illumination`, `extra_data`, etc.
- **Methods**:
    - `read_header(cls, path)`: A class method to parse the header from a file.
    - `write_header(self, path)`: A method to serialize the current metadata back to a HIPS header.
    - `__repr__`: For easy debugging and inspection.

## 3. Implementation Details
- **Header Parsing**: 
    - Sequential reading of the standard ASCII header lines.
    - Binary block reading for `history` and `description` (using sizes `szhist` and `szdesc`).
    - Line-by-line parsing of extended parameter definitions.
    - Binary block reading for array parameters (e.g., wavelengths), handling the 4-byte padding/alignment.
- **Data Types**: Use `numpy` arrays for numeric sequences (wavelengths, strobe times) to match the existing API in `hips.py`.
- **Pixel Formats**: Implement a mapping for the `Format` enum found in `Format.cs` to translate HIPS format integers to human-readable compression and data format types.

## 4. Verification & Testing Strategy (The "Oracle" Approach)
A dedicated test suite `tests/test_hips_core.py` will be created.
- **Oracle Validation**: For a set of representative `.hips` files (from `TestData/` and `tests/TestImages/`), the test will:
    1. Load the file using the legacy `hips.ImageClass`.
    2. Load the file using the new `hips_core.HipsImage.read_header()`.
    3. Assert that all metadata fields match (with tolerance for float comparisons where applicable).
- **Round-Trip Validation**:
    1. Load a header using `HipsImage`.
    2. Write it to a temporary file.
    3. Read it back using `HipsImage` and ensure parity.
    4. (Optional) Append dummy pixel data and verify the legacy `ImageClass` can still parse the resulting file.

## 5. Compatibility
Ensure that `HipsImage` attributes are named consistently with `ImageClass` (e.g., `MmPixel`, `WaveLengths`) or provide aliases to ensure it can eventually serve as a drop-in replacement.
