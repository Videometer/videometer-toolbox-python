import os
import struct
import numpy as np
from typing import List, Dict, Any, Optional, Union
from enum import IntEnum
from dataclasses import dataclass, field

class HipsFormat(IntEnum):
    """
    Mapping of HIPS pixel format integers to their meanings.
    Based on Format.cs from the legacy implementation.
    """
    PFBYTE = 0
    PFSHORT = 1
    PFINT = 2
    PFFLOAT = 3
    PFASCII = 5
    PFDOUBLE = 6
    PFRGB = 35
    
    # Compressed formats
    PFBYTE_GZ = 0x80 + 0
    PFSHORT_GZ = 0x80 + 1
    PFFLOAT_GZ = 0x80 + 3
    PFRGB_GZ = 0x80 + 35

    PFBYTE_JPG = 0x100 + 0

    PFBYTE_PNG = 0x200 + 0
    PFSHORT_PNG = 0x200 + 1

@dataclass
class HipsImage:
    """
    Pure Python implementation of the HIPS image format header.
    Contains metadata and methods to read/write HIPS headers.
    """
    width: int = 0
    height: int = 0
    bands: int = 0
    format: int = HipsFormat.PFBYTE
    history: str = ""
    description: str = ""
    
    # ROI Information
    roi_height: int = 0
    roi_width: int = 0
    roi_y: int = 0
    roi_x: int = 0
    
    # Extended Parameters (X-tra parameters)
    mm_pixel: float = 0.0
    band_names: List[str] = field(default_factory=list)
    wavelengths: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    strobe_times: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    strobe_times_universal: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    illumination: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    extra_data: Dict[str, float] = field(default_factory=dict)
    extra_data_int: Dict[str, int] = field(default_factory=dict)
    extra_data_string: Dict[str, str] = field(default_factory=dict)
    
    # Other X-tra parameters
    camera_temperature: float = 0.0
    freehand_layers_xml: str = ""
    drawing_primitive_xml: str = ""
    id: str = ""
    
    # internal offset tracking
    _data_offset: int = 0

    @classmethod
    def read_header(cls, path: str) -> 'HipsImage':
        """
        Reads the HIPS header from a file.
        """
        with open(path, 'rb') as f:
            # First line: HIPS
            line = f.readline().decode('ascii').strip()
            if "HIPS" not in line:
                raise ValueError(f"File {path} is not a valid HIPS image.")
            
            # Skip 4 lines (onm, snm, frames, odt)
            # Actually 'frames' is used as bands if it's > colors
            f.readline() # onm
            f.readline() # snm
            frames = int(f.readline().decode('ascii').strip())
            f.readline() # odt
            
            # orows, ocols
            height = int(f.readline().decode('ascii').strip())
            width = int(f.readline().decode('ascii').strip())
            
            # roirows, roicols, frow, fcol
            roi_height = int(f.readline().decode('ascii').strip())
            roi_width = int(f.readline().decode('ascii').strip())
            roi_y = int(f.readline().decode('ascii').strip())
            roi_x = int(f.readline().decode('ascii').strip())
            
            # format
            pixel_format = HipsFormat(int(f.readline().decode('ascii').strip()))
            
            # colors
            colors = int(f.readline().decode('ascii').strip())
            
            # Number of bands calculation (matching HipsIO.cs)
            # bands = ((frames > colors || format == ImageDataFormat.ByteRGBPixel) ? frames : colors);
            # We don't have ImageDataFormat enum yet here, but ByteRGBPixel corresponds to PFRGB (35)
            bands = frames if (frames > colors or pixel_format == HipsFormat.PFRGB) else colors
            
            img = cls(
                width=width, height=height, bands=bands, format=pixel_format,
                roi_height=roi_height, roi_width=roi_width, roi_y=roi_y, roi_x=roi_x
            )
            
            # History
            szhist_line = f.readline().decode('ascii').strip()
            szhist = int(szhist_line)
            history_bytes = f.read(szhist)
            img.history = history_bytes.decode('utf-8', errors='replace').rstrip('\n\r\0')
            
            # Swallow newline if any after history
            curr = f.tell()
            if f.read(1) != b'\n':
                f.seek(curr)
                
            # Description
            szdesc_line = f.readline().decode('ascii').strip()
            szdesc = int(szdesc_line)
            description_bytes = f.read(szdesc)
            img.description = description_bytes.decode('utf-8', errors='replace').rstrip('\n\r\0')
            
            # Swallow newline if any after description
            curr = f.tell()
            if f.read(1) != b'\n':
                f.seek(curr)
                
            # Extended Parameters
            img._read_x_params(f)
            
            img._data_offset = f.tell()
            return img

    def _read_x_params(self, f):
        """
        Parses extended (X-tra) parameters from the HIPS file.
        """
        line = f.readline().decode('ascii').strip()
        if not line:
            return
        
        n_param = int(line)
        x_params = []
        
        for _ in range(n_param):
            headerline = f.readline().decode('ascii').strip()
            parts = headerline.split(' ', 3)
            if len(parts) < 4:
                continue
                
            name = parts[0]
            fmt_char = parts[1][0]
            count = int(parts[2])
            val_or_offset = parts[3]
            
            x_params.append({
                'name': name,
                'format': fmt_char,
                'count': count,
                'val_or_offset': val_or_offset
            })
            
        # Binary block offset
        line = f.readline().decode('ascii').strip()
        if not line:
            return
        byte_offset_total = int(line)
        
        # Current position is where the binary block starts
        binary_start = f.tell()
        
        # Process each parameter
        for xp in x_params:
            name = xp['name']
            fmt = xp['format']
            count = xp['count']
            
            if count == 1:
                # Single value parameter
                val = xp['val_or_offset']
                self._set_single_x_param(name, fmt, val)
            else:
                # Array parameter in binary block
                offset = int(xp['val_or_offset'])
                f.seek(binary_start + offset)
                
                # Calculate size and read
                data_size = self._get_format_size(fmt) * count
                # Alignment: (data_size + 3) & ~3
                padded_size = (data_size + 3) & ~3
                data = f.read(padded_size)
                
                self._set_array_x_param(name, fmt, count, data[:data_size])
                
        # Move file pointer to end of binary block
        f.seek(binary_start + byte_offset_total)

    def _get_format_size(self, fmt_char: str) -> int:
        return {'b': 1, 's': 2, 'i': 4, 'f': 4, 'd': 8, 'c': 1}.get(fmt_char, 1)

    def _set_single_x_param(self, name: str, fmt: str, val: str):
        if name == "MmPixel":
            self.mm_pixel = float(val)
        elif name == "CameraTemperature":
            self.camera_temperature = float(val)
        elif name == "Id":
            self.id = val
        elif name == "FreehandLayersXML":
            self.freehand_layers_xml = val
        elif name == "DrawingPrimitiveXML":
            self.drawing_primitive_xml = val
        elif name.startswith("BandName"):
            idx = int(name[len("BandName"):])
            while len(self.band_names) <= idx:
                self.band_names.append("")
            self.band_names[idx] = val
        elif name.startswith("ExtraData_"):
            self.extra_data[name[len("ExtraData_"):]] = float(val)
        elif name.startswith("ExtraDataInt_"):
            self.extra_data_int[name[len("ExtraDataInt_"):]] = int(val)
        elif name.startswith("ExtraDataString_"):
            self.extra_data_string[name[len("ExtraDataString_"):]] = val

    def _set_array_x_param(self, name: str, fmt: str, count: int, data: bytes):
        if fmt == 'f':
            arr = np.frombuffer(data, dtype=np.float32, count=count)
        elif fmt == 'i':
            arr = np.frombuffer(data, dtype=np.int32, count=count)
        elif fmt == 's':
            arr = np.frombuffer(data, dtype=np.int16, count=count)
        elif fmt == 'd':
            arr = np.frombuffer(data, dtype=np.float64, count=count)
        elif fmt == 'b':
            arr = np.frombuffer(data, dtype=np.uint8, count=count)
        elif fmt == 'c':
            arr = data.decode('ascii', errors='replace')
        else:
            return

        if name == "BandWaveLength":
            self.wavelengths = arr.copy()
        elif name == "BandStrobeTime":
            self.strobe_times = arr.astype(np.int32).copy()
        elif name == "BandStrobeTimesUniversal":
            self.strobe_times_universal = arr.copy()
        elif name == "BandIllumination":
            self.illumination = arr.astype(np.int32).copy()

    def write_header(self, path: str):
        """
        Writes the HIPS header to a file.
        NOTE: This only writes the header. Pixel data must be appended separately.
        """
        with open(path, 'wb') as f:
            f.write(b"HIPS\n")
            f.write(b"\n") # onm
            f.write(b"\n") # snm
            f.write(f"{self.bands}\n".encode('ascii')) # frames
            f.write(b"\n") # odt
            f.write(f"{self.height}\n".encode('ascii'))
            f.write(f"{self.width}\n".encode('ascii'))
            f.write(f"{self.roi_height}\n".encode('ascii'))
            f.write(f"{self.roi_width}\n".encode('ascii'))
            f.write(f"{self.roi_y}\n".encode('ascii'))
            f.write(f"{self.roi_x}\n".encode('ascii'))
            f.write(f"{int(self.format)}\n".encode('ascii'))
            f.write(f"{self.bands}\n".encode('ascii')) # colors
            
            # History
            hist_bytes = self.history.encode('utf-8') + b'\n'
            f.write(f"{len(hist_bytes)}\n".encode('ascii'))
            f.write(hist_bytes)
            
            # Description
            desc_bytes = self.description.encode('utf-8') + b'\n'
            f.write(f"{len(desc_bytes)}\n".encode('ascii'))
            f.write(desc_bytes)
            
            # X-tra parameters
            self._write_x_params(f)

    def _write_x_params(self, f):
        """
        Serializes extended (X-tra) parameters.
        """
        x_params = []
        binary_data = b""
        curr_offset = 0
        
        # Simple values
        if self.mm_pixel != 0.0:
            x_params.append(f"MmPixel f 1 {self.mm_pixel}")
        if self.camera_temperature != 0.0:
            x_params.append(f"CameraTemperature f 1 {self.camera_temperature}")
        if self.id:
            x_params.append(f"Id c 1 {self.id}")
        if self.freehand_layers_xml:
            x_params.append(f"FreehandLayersXML c 1 {self.freehand_layers_xml}")
        if self.drawing_primitive_xml:
            x_params.append(f"DrawingPrimitiveXML c 1 {self.drawing_primitive_xml}")
            
        # Band names
        for i, name in enumerate(self.band_names):
            if name:
                x_params.append(f"BandName{i} c 1 {name}")
                
        # Extra data
        for k, v in self.extra_data.items():
            x_params.append(f"ExtraData_{k} f 1 {v}")
        for k, v in self.extra_data_int.items():
            x_params.append(f"ExtraDataInt_{k} i 1 {v}")
        for k, v in self.extra_data_string.items():
            x_params.append(f"ExtraDataString_{k} c 1 {v}")
                
        # Arrays
        def add_array(name, arr, fmt_char):
            nonlocal binary_data, curr_offset
            if len(arr) == 0:
                return
            count = len(arr)
            if fmt_char == 'f':
                data = arr.astype(np.float32).tobytes()
            elif fmt_char == 'i':
                data = arr.astype(np.int32).tobytes()
            elif fmt_char == 'd':
                data = arr.astype(np.float64).tobytes()
            else:
                return
                
            x_params.append(f"{name} {fmt_char} {count} {curr_offset}")
            binary_data += data
            pad = (len(data) + 3) & ~3
            binary_data += b'\0' * (pad - len(data))
            curr_offset += pad

        add_array("BandWaveLength", self.wavelengths, 'f')
        add_array("BandStrobeTime", self.strobe_times, 'f') # Legacy stores as float
        add_array("BandStrobeTimesUniversal", self.strobe_times_universal, 'f')
        add_array("BandIllumination", self.illumination, 'f') # Legacy stores as float
            
        # Write n_param
        f.write(f"{len(x_params)}\n".encode('ascii'))
        for p in x_params:
            f.write(f"{p}\n".encode('ascii'))
            
        # Write total byte offset
        f.write(f"{len(binary_data)}\n".encode('ascii'))
        f.write(binary_data)
        
    def __str__(self) -> str:
        """Returns a string summary of the HIPS image."""
        lines = [
            f"HIPS Image Summary:",
            f"  Dimensions: {self.width}x{self.height} pixels",
            f"  Bands: {self.bands}",
            f"  Format: {self.format.name} ({int(self.format)})",
            f"  MmPixel: {self.mm_pixel:.6f}",
            f"  ROI: {self.roi_width}x{self.roi_height} at ({self.roi_x}, {self.roi_y})",
            f"  History: {self.history[:100]}..." if len(self.history) > 100 else f"  History: {self.history}",
            f"  Description: {self.description[:100]}..." if len(self.description) > 100 else f"  Description: {self.description}",
        ]
        
        if len(self.wavelengths) > 0:
            wl_str = ", ".join([f"{w:.1f}" for w in self.wavelengths[:5]])
            if len(self.wavelengths) > 5:
                wl_str += ", ..."
            lines.append(f"  Wavelengths: [{wl_str}]")
            
        if self.extra_data:
            lines.append(f"  Extra Data: {list(self.extra_data.keys())}")
            
        if self.id:
            lines.append(f"  ID: {self.id}")
            
        return "\n".join(lines)

def main():
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Inspect HIPS file header information.")
    parser.add_argument("path", help="Path to the .hips file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all extended parameters")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.path):
        print(f"Error: File {args.path} not found.")
        sys.exit(1)
        
    try:
        img = HipsImage.read_header(args.path)
        print(img)
        
        if args.verbose:
            print("\nExtended Parameters:")
            if img.band_names:
                print(f"  Band Names: {img.band_names}")
            if len(img.strobe_times) > 0:
                print(f"  Strobe Times: {img.strobe_times.tolist()}")
            if len(img.illumination) > 0:
                print(f"  Illumination: {img.illumination.tolist()}")
            if img.extra_data_int:
                print(f"  Extra Data (Int): {img.extra_data_int}")
            if img.extra_data_string:
                print(f"  Extra Data (String): {img.extra_data_string}")
            if img.camera_temperature != 0.0:
                print(f"  Camera Temperature: {img.camera_temperature}")
            if img.freehand_layers_xml:
                print(f"  Freehand Layers XML: {len(img.freehand_layers_xml)} bytes")
                
    except Exception as e:
        print(f"Error reading HIPS file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
