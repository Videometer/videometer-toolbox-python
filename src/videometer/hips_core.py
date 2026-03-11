import os
import struct
import numpy as np
import xml.etree.ElementTree as ET
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

class BaseEncoder:
    """Base class for HIPS band encoders."""
    def encode_band(self, band_data: np.ndarray) -> bytes:
        raise NotImplementedError()

class RawEncoder(BaseEncoder):
    """Encodes band data as raw bytes."""
    def __init__(self, dtype: np.dtype):
        self.dtype = dtype
        
    def encode_band(self, band_data: np.ndarray) -> bytes:
        # Ensure data is in the correct dtype and contiguous
        return band_data.astype(self.dtype).tobytes()

class GzipEncoder(BaseEncoder):
    """Encodes band data using GZIP compression."""
    def __init__(self, dtype: np.dtype):
        self.dtype = dtype
        
    def encode_band(self, band_data: np.ndarray) -> bytes:
        import gzip
        raw_bytes = band_data.astype(self.dtype).tobytes()
        return gzip.compress(raw_bytes)

class PngEncoder(BaseEncoder):
    """Encodes band data as a PNG image."""
    def encode_band(self, band_data: np.ndarray) -> bytes:
        import io
        from PIL import Image
        
        # PIL expects (H, W) for grayscale
        # band_data should be uint8 or uint16 for PNG
        img = Image.fromarray(band_data)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

class JpegEncoder(BaseEncoder):
    """Encodes band data as a JPEG image."""
    def __init__(self, quality: int = 90):
        self.quality = quality
        
    def encode_band(self, band_data: np.ndarray) -> bytes:
        import io
        from PIL import Image
        
        # JPEG only supports uint8
        if band_data.dtype != np.uint8:
            band_data = band_data.astype(np.uint8)
            
        img = Image.fromarray(band_data)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.quality)
        return buf.getvalue()

# Pre-defined Videometer compression presets
COMPRESSION_PRESETS = {
    "Uncompressed": {
        "format": HipsFormat.PFFLOAT,
        "quantize": False
    },
    "VeryHighQuality": {
        "format": HipsFormat.PFSHORT_PNG,
        "quantize": True, "Q": 12, "Q_Min": 0.0, "Q_Max": 120.0
    },
    "HighQuality": {
        "format": HipsFormat.PFSHORT_PNG,
        "quantize": True, "Q": 10, "Q_Min": 0.0, "Q_Max": 120.0
    },
    "HighCompression": {
        "format": HipsFormat.PFBYTE_PNG,
        "quantize": True, "Q": 8, "Q_Min": 0.0, "Q_Max": 120.0
    },
    "VeryHighCompression": {
        "format": HipsFormat.PFBYTE_JPG,
        "quantize": True, "Q": 8, "Q_Min": 0.0, "Q_Max": 120.0
    }
}

# Illumination types mapping
ILLUMINATION_TYPES = {
    -1: "Mixed",
    0: "NA",
    1: "Empty",
    2: "Diffused_Highpower_LED",
    3: "Brightfield_BackLight",
    4: "Darkfield_BackLight",
    5: "Diffused_Laser",
    6: "SpotLaser",
    7: "Diffused_Lowpower_LED",
    8: "Direct_Lowpower_LED",
    9: "Diffused_UV",
    10: "Harmless_Guide_Laser",
    11: "TEST_PROBE",
    12: "Darkfield_FrontLight",
    13: "Coaxial_FrontLight",
}

@dataclass
class QuantificationParameters:
    """Parameters for de-quantizing image data."""
    Q: int = 8
    Q_Min: float = 0.0
    Q_Max: float = 1.0

@dataclass
class HipsImage:
    """
    Pure Python implementation of the HIPS image format header and data.
    Contains metadata and methods to read/write HIPS files.
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
    
    # Quantification
    _quantization_parameters: Optional[List[QuantificationParameters]] = None
    _original_format: Optional[int] = None

    # Internal state
    _data_offset: int = 0
    _pixels: Optional[np.ndarray] = None
    _path: Optional[str] = None
    _x_params_raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def pixels(self) -> np.ndarray:
        """Access the pixel data as a 3D numpy array (height, width, bands)."""
        if self._pixels is None:
            self.load_pixels()
        return self._pixels

    @pixels.setter
    def pixels(self, value: np.ndarray):
        """Sets the pixel data and updates dimensions."""
        self._pixels = value
        self.height, self.width = value.shape[:2]
        if len(value.shape) > 2:
            self.bands = value.shape[2]
        else:
            self.bands = 1
        # Update ROI to match new full dimensions
        self.roi_width = self.width
        self.roi_height = self.height
        self.roi_x = 0
        self.roi_y = 0

    @property
    def illumination_names(self) -> List[str]:
        """Returns the string names of the illumination types for each band."""
        return [ILLUMINATION_TYPES.get(int(i), "NA") for i in self.illumination]

    def reduce_bands(self, indexes: List[int]):
        """Reduces the image to only the specified band indexes."""
        if self._pixels is None:
            self.load_pixels()
            
        self._pixels = self._pixels[:, :, indexes]
        self.bands = len(indexes)
        
        if len(self.wavelengths) > 0:
            self.wavelengths = self.wavelengths[indexes]
        if len(self.strobe_times) > 0:
            self.strobe_times = self.strobe_times[indexes]
        if len(self.strobe_times_universal) > 0:
            self.strobe_times_universal = self.strobe_times_universal[indexes]
        if len(self.illumination) > 0:
            self.illumination = self.illumination[indexes]
        if self.band_names:
            self.band_names = [self.band_names[i] for i in indexes if i < len(self.band_names)]
        if self._quantization_parameters:
            self._quantization_parameters = [self._quantization_parameters[i] for i in indexes]

    def load_pixels(self):
        """Loads the pixel data from the file."""
        if not self._path:
            raise ValueError("No file path associated with this HipsImage.")
            
        with open(self._path, 'rb') as f:
            f.seek(self._data_offset)
            
            # Identify compression
            actual_format = self.format & 0x7F
            is_gz = bool(self.format & 0x80)
            is_jpg = bool(self.format & 0x100)
            is_png = bool(self.format & 0x200)
            is_compressed = is_gz or is_jpg or is_png
            
            if not is_compressed and self._quantization_parameters is None:
                self._load_raw_pixels(f)
            else:
                # Some files might have the format flag but NO size table 
                # (e.g. if they were saved with a different version or specific settings)
                # Check if first 4 bytes look like a plausible chunk size
                curr_pos = f.tell()
                first_4 = f.read(4)
                f.seek(curr_pos)
                
                if len(first_4) < 4:
                    raise EOFError("File ended before pixel data")
                    
                first_val = struct.unpack('<I', first_4)[0]
                # If first_val matches total remaining size, or is just very large, 
                # it might be a size table.
                # If it's NOT a size table, it might be RAW data starting immediately.
                
                # For plate.hips, first_val was ~2.3M, which is smaller than raw band (~19M)
                # so it IS likely a size table.
                self._load_compressed_or_quantified_pixels(f, is_gz, is_jpg, is_png)

    def _load_compressed_or_quantified_pixels(self, f, is_gz, is_jpg, is_png):
        """Handles chunked compressed data and de-quantization."""
        import gzip
        import io
        from PIL import Image
        
        # Determine target dtype
        if self._quantization_parameters:
            target_dtype = np.float32
        else:
            actual_format = (self._original_format if self._original_format is not None 
                             else (self.format & 0x7F))
            dtype_map = {
                HipsFormat.PFBYTE: np.uint8,
                HipsFormat.PFSHORT: np.int16,
                HipsFormat.PFINT: np.int32,
                HipsFormat.PFFLOAT: np.float32,
                HipsFormat.PFDOUBLE: np.float64,
                HipsFormat.PFRGB: np.uint8,
            }
            target_dtype = dtype_map.get(actual_format, np.uint8)

        self._pixels = np.zeros((self.height, self.width, self.bands), dtype=target_dtype)
        
        is_rgb = (self.format & 0x7F) == HipsFormat.PFRGB
        
        # In HIPS chunked formats (256, 512, 513), chunks are interleaved: [Size][Data][Size][Data]
        for b in range(self.bands):
            size_data = f.read(4)
            if not size_data:
                break
            chunk_size = struct.unpack('<I', size_data)[0]
            compressed_data = f.read(chunk_size)
            
            # if b == 0:
            #    print(f"DEBUG: Chunk 0 size={chunk_size}, hex={compressed_data[:16].hex(' ')}")
            
            if is_gz:
                decompressed_data = gzip.decompress(compressed_data)
                if is_rgb:
                    temp_pixels = np.frombuffer(decompressed_data, dtype=np.uint8).reshape(self.height, self.width, 3)
                    self._pixels[:, :, :3] = temp_pixels
                    break # PFRGB is 1 chunk
                else:
                    decompressed_band = np.frombuffer(decompressed_data, 
                                                     dtype=np.uint8 if self._quantization_parameters else target_dtype).reshape(self.height, self.width)
            elif is_png or is_jpg:
                # Pillow can handle PNG and JPEG from bytes
                with Image.open(io.BytesIO(compressed_data)) as img:
                    decompressed_band = np.array(img)
                    if is_rgb and len(decompressed_band.shape) == 3:
                        # Interleaved RGB
                        self._pixels[:, :, :3] = decompressed_band
                        break # 1 chunk
            else:
                # RAW but quantified
                stored_dtype = np.uint8
                if self._quantization_parameters:
                    stored_dtype = np.uint8 if self._quantization_parameters[b].Q <= 8 else np.int16
                decompressed_band = np.frombuffer(compressed_data, dtype=stored_dtype).reshape(self.height, self.width)
                
            if self._quantization_parameters:
                q_params = self._quantization_parameters[b]
                max_q = float(2**q_params.Q - 1)
                factor = (q_params.Q_Max - q_params.Q_Min) / max_q
                self._pixels[:, :, b] = decompressed_band.astype(np.float32) * factor + q_params.Q_Min
            else:
                # Handle potential (H, W, 1) from Pillow
                if len(decompressed_band.shape) == 3 and decompressed_band.shape[2] == 1:
                    decompressed_band = decompressed_band.reshape(self.height, self.width)
                self._pixels[:, :, b] = decompressed_band.astype(target_dtype)

    def _load_raw_pixels(self, f):
        """Reads uncompressed band-sequential pixel data."""
        dtype_map = {
            HipsFormat.PFBYTE: np.uint8,
            HipsFormat.PFSHORT: np.int16,
            HipsFormat.PFINT: np.int32,
            HipsFormat.PFFLOAT: np.float32,
            HipsFormat.PFDOUBLE: np.float64,
            HipsFormat.PFRGB: np.uint8,
        }
        
        actual_format = self.format & 0x7F
        dtype = dtype_map.get(actual_format, np.uint8)
        element_size = np.dtype(dtype).itemsize
        
        if actual_format == HipsFormat.PFRGB:
            pixel_size = 3 * element_size
            total_size = self.width * self.height * pixel_size
            data = f.read(total_size)
            if len(data) < total_size:
                raise EOFError("Unexpected end of file while reading RGB data")
            self._pixels = np.frombuffer(data, dtype=dtype).reshape(self.height, self.width, 3)
        else:
            band_size = self.width * self.height * element_size
            self._pixels = np.zeros((self.height, self.width, self.bands), dtype=dtype)
            
            for b in range(self.bands):
                data = f.read(band_size)
                if len(data) < band_size:
                    raise EOFError(f"Unexpected end of file while reading band {b}")
                self._pixels[:, :, b] = np.frombuffer(data, dtype=dtype).reshape(self.height, self.width)

    @classmethod
    def read(cls, path: str) -> 'HipsImage':
        """Reads a HIPS file (header and prepares for lazy pixel loading)."""
        img = cls.read_header(path)
        img._path = path
        return img

    @classmethod
    def read_header(cls, path: str) -> 'HipsImage':
        """Reads the HIPS header from a file."""
        with open(path, 'rb') as f:
            def read_next_val():
                while True:
                    line = f.readline().decode('ascii', errors='replace').strip()
                    if line:
                        return line

            line = f.readline().decode('ascii').strip()
            if "HIPS" not in line:
                raise ValueError(f"File {path} is not a valid HIPS image.")
            
            f.readline() # onm
            f.readline() # snm
            frames = int(read_next_val())
            f.readline() # odt
            
            height = int(read_next_val())
            width = int(read_next_val())
            
            roi_height = int(read_next_val())
            roi_width = int(read_next_val())
            roi_y = int(read_next_val())
            roi_x = int(read_next_val())
            
            pixel_format = HipsFormat(int(read_next_val()))
            colors = int(read_next_val())
            bands = frames if (frames > colors or pixel_format == HipsFormat.PFRGB) else colors
            
            img = cls(
                width=width, height=height, bands=bands, format=pixel_format,
                roi_height=roi_height, roi_width=roi_width, roi_y=roi_y, roi_x=roi_x
            )
            img._path = path
            
            szhist = int(read_next_val())
            history_bytes = f.read(szhist)
            img.history = history_bytes.decode('utf-8', errors='replace').rstrip('\n\r\0')
            
            szdesc = int(read_next_val())
            description_bytes = f.read(szdesc)
            img.description = description_bytes.decode('utf-8', errors='replace').rstrip('\n\r\0')
            
            # Extended Parameters
            img._read_x_params(f)
            img._data_offset = f.tell()
            return img

    def _read_x_params(self, f):
        def read_next_val():
            while True:
                line = f.readline().decode('ascii', errors='replace').strip()
                if line:
                    return line
        
        try:
            line = read_next_val()
        except:
            return
            
        n_param = int(line)
        x_params = []
        
        for _ in range(n_param):
            headerline = f.readline().decode('ascii', errors='replace').strip()
            if not headerline: # Handle empty lines in xparam headers if any
                headerline = read_next_val()
                
            parts = headerline.split(' ', 3)
            if len(parts) < 4:
                continue
                
            x_params.append({
                'name': parts[0],
                'format': parts[1][0],
                'count': int(parts[2]),
                'val_or_offset': parts[3]
            })
            
        line = read_next_val()
        byte_offset_total = int(line)
        binary_start = f.tell()
        
        for xp in x_params:
            name = xp['name']
            fmt = xp['format']
            count = xp['count']
            
            if count == 1:
                self._set_single_x_param(name, fmt, xp['val_or_offset'])
            else:
                offset = int(xp['val_or_offset'])
                f.seek(binary_start + offset)
                data_size = self._get_format_size(fmt) * count
                padded_size = (data_size + 3) & ~3
                data = f.read(padded_size)
                self._set_array_x_param(name, fmt, count, data[:data_size])
                
        f.seek(binary_start + byte_offset_total)

    def _get_format_size(self, fmt_char: str) -> int:
        return {'b': 1, 's': 2, 'i': 4, 'f': 4, 'd': 8, 'c': 1}.get(fmt_char, 1)

    def _parse_quantization(self, val: str, is_legacy: bool):
        """Helper to parse XML quantization parameters."""
        try:
            # Handle possible UTF-16 declaration in a string already decoded as UTF-8/ASCII
            if 'encoding="utf-16"' in val:
                val = val.replace('encoding="utf-16"', 'encoding="utf-8"')
            
            root = ET.fromstring(val)
            if is_legacy:
                # Note spelling mistake in legacy XML tag: QuantificationParamaters
                qp_node = root if "QuantificationParamaters" in root.tag else root.find(".//QuantificationParamaters")
                if qp_node is not None:
                    qp = QuantificationParameters(
                        Q=int(float(qp_node.find('Q').text)),
                        Q_Min=float(qp_node.find('Q_Min').text),
                        Q_Max=float(qp_node.find('Q_Max').text)
                    )
                    self._quantization_parameters = [qp] * self.bands
                    
                    orig_fmt_node = qp_node.find('OriginalFormat')
                    if orig_fmt_node is not None:
                        fmt_map = {"BytePixel": 0, "Int16Pixel": 1, "Int32Pixel": 2, "FloatPixel": 3, "DoublePixel": 6, "ByteRGBPixel": 35}
                        self._original_format = fmt_map.get(orig_fmt_node.text)
            else:
                params = []
                for qp_node in root.findall('.//QuantificationParameters'):
                    qp = QuantificationParameters(
                        Q=int(qp_node.find('Q').text),
                        Q_Min=float(qp_node.find('Q_Min').text),
                        Q_Max=float(qp_node.find('Q_Max').text)
                    )
                    params.append(qp)
                if params:
                    self._quantization_parameters = params
        except Exception as e:
            # print(f"DEBUG: XML Parse error: {e}")
            pass

    def _set_single_x_param(self, name: str, fmt: str, val: str):
        self._x_params_raw[name] = val
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
            try:
                idx = int(name[len("BandName"):])
                while len(self.band_names) <= idx:
                    self.band_names.append("")
                self.band_names[idx] = val
            except: pass
        elif name.startswith("ExtraData_"):
            self.extra_data[name[len("ExtraData_"):]] = float(val)
        elif name.startswith("ExtraDataInt_"):
            self.extra_data_int[name[len("ExtraDataInt_"):]] = int(val)
        elif name.startswith("ExtraDataString_"):
            self.extra_data_string[name[len("ExtraDataString_"):]] = val
        elif name == "BandQuantification":
            self._parse_quantization(val, is_legacy=False)
        elif name == "Quantification":
            self._parse_quantization(val, is_legacy=True)
        elif name == "OriginalFormat":
            fmt_map = {"BytePixel": 0, "Int16Pixel": 1, "Int32Pixel": 2, "FloatPixel": 3, "DoublePixel": 6, "ByteRGBPixel": 35}
            self._original_format = fmt_map.get(val)

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
            arr = data.decode('ascii', errors='replace').rstrip('\0')
        else:
            return

        self._x_params_raw[name] = arr

        if name == "BandWaveLength":
            self.wavelengths = arr.copy()
        elif name == "BandStrobeTime":
            self.strobe_times = arr.astype(np.int32).copy()
        elif name == "BandStrobeTimesUniversal":
            self.strobe_times_universal = arr.copy()
        elif name == "BandIllumination":
            self.illumination = arr.astype(np.int32).copy()
        elif name.startswith("BandName"):
            try:
                idx = int(name[len("BandName"):])
                while len(self.band_names) <= idx:
                    self.band_names.append("")
                self.band_names[idx] = str(arr)
            except: pass
        elif name.startswith("ExtraData_"):
            val = float(arr[0]) if hasattr(arr, '__len__') and len(arr)>0 else (float(arr) if not isinstance(arr, str) else 0.0)
            self.extra_data[name[len("ExtraData_"):]] = val
        elif name.startswith("ExtraDataInt_"):
            val = int(arr[0]) if hasattr(arr, '__len__') and len(arr)>0 else (int(arr) if not isinstance(arr, str) else 0)
            self.extra_data_int[name[len("ExtraDataInt_"):]] = val
        elif name.startswith("ExtraDataString_"):
            self.extra_data_string[name[len("ExtraDataString_"):]] = str(arr)
        elif name == "BandQuantification":
            self._parse_quantization(str(arr), is_legacy=False)
        elif name == "Quantification":
            self._parse_quantization(str(arr), is_legacy=True)

    def _write_to_handle(self, f):
        """Internal helper to write header to an open file handle."""
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
        
        hist_bytes = self.history.encode('utf-8') + b'\n'
        f.write(f"{len(hist_bytes)}\n".encode('ascii'))
        f.write(hist_bytes)
        
        desc_bytes = self.description.encode('utf-8') + b'\n'
        f.write(f"{len(desc_bytes)}\n".encode('ascii'))
        f.write(desc_bytes)
        
        self._write_x_params(f)

    def write_header(self, path: str):
        """Writes only the header to a file."""
        with open(path, 'wb') as f:
            self._write_to_handle(f)

    def write(self, path: str, compression: Optional[str] = None):
        """
        Writes the HIPS image (header and data) to a file.
        If compression is None, it attempts to use the current self.format.
        """
        if self._pixels is None:
            self.load_pixels()
            
        if compression is not None:
            preset = COMPRESSION_PRESETS.get(compression)
            if not preset:
                raise ValueError(f"Invalid compression preset '{compression}'")
                
            self.format = preset["format"]
            is_quantized = preset["quantize"]
            
            if is_quantized:
                q_val = preset["Q"]
                q_min = preset["Q_Min"]
                q_max = preset["Q_Max"]
                self._quantization_parameters = [
                    QuantificationParameters(Q=q_val, Q_Min=q_min, Q_Max=q_max) 
                    for _ in range(self.bands)
                ]
                self._original_format = HipsFormat.PFFLOAT
            else:
                self._quantization_parameters = None
                self._original_format = None
        else:
            # If no compression preset specified, but we are writing float32 pixels to a PFBYTE format, 
            # we should probably default to PFFLOAT to avoid data loss.
            if self.format == HipsFormat.PFBYTE and self._pixels.dtype == np.float32:
                self.format = HipsFormat.PFFLOAT

        is_gz = bool(self.format & 0x80)
        is_jpg = bool(self.format & 0x100)
        is_png = bool(self.format & 0x200)
        is_chunked = is_gz or is_jpg or is_png
        is_quantized = self._quantization_parameters is not None
        
        if is_png:
            encoder = PngEncoder()
        elif is_jpg:
            encoder = JpegEncoder()
        elif is_gz:
            dtype = np.uint8 if is_quantized else (np.float32 if (self.format & 0x7F) == HipsFormat.PFFLOAT else np.uint8)
            encoder = GzipEncoder(dtype)
        else:
            dtype = np.uint8 if is_quantized else (np.float32 if (self.format & 0x7F) == HipsFormat.PFFLOAT else np.uint8)
            encoder = RawEncoder(dtype)

        with open(path, 'wb') as f:
            self._write_to_handle(f)
            
            if is_chunked:
                for b in range(self.bands):
                    band_data = self._pixels[:, :, b]
                    if is_quantized:
                        band_data = self._quantize_band(band_data, self._quantization_parameters[b])
                    
                    encoded_bytes = encoder.encode_band(band_data)
                    # Interleaved: write size, then data
                    f.write(struct.pack('<I', len(encoded_bytes)))
                    f.write(encoded_bytes)
            else:
                # RAW
                for b in range(self.bands):
                    band_data = self._pixels[:, :, b]
                    if is_quantized:
                        band_data = self._quantize_band(band_data, self._quantization_parameters[b])
                    f.write(encoder.encode_band(band_data))

    def _quantize_band(self, band_data: np.ndarray, qp: QuantificationParameters) -> np.ndarray:
        max_q = float(2**qp.Q - 1)
        range_val = qp.Q_Max - qp.Q_Min
        if range_val == 0:
            quantized = np.zeros_like(band_data)
        else:
            quantized = (band_data - qp.Q_Min) / range_val * max_q
            
        dtype = np.uint8 if qp.Q <= 8 else np.int16
        return np.round(quantized).clip(0, max_q).astype(dtype)

    def _write_x_params(self, f):
        x_params = []
        binary_data = b""
        curr_offset = 0
        
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
            
        if self._quantization_parameters:
            root = ET.Element("ArrayOfQuantificationParameters")
            for qp in self._quantization_parameters:
                qp_node = ET.SubElement(root, "QuantificationParameters")
                ET.SubElement(qp_node, "Q").text = str(qp.Q)
                ET.SubElement(qp_node, "Q_Min").text = str(qp.Q_Min)
                ET.SubElement(qp_node, "Q_Max").text = str(qp.Q_Max)
            xml_str = ET.tostring(root, encoding='unicode')
            x_params.append(f"BandQuantification c 1 {xml_str}")
            
        if self._original_format is not None:
            fmt_names = {0: "BytePixel", 1: "Int16Pixel", 2: "Int32Pixel", 3: "FloatPixel", 6: "DoublePixel", 35: "ByteRGBPixel"}
            fmt_name = fmt_names.get(self._original_format, "FloatPixel")
            x_params.append(f"OriginalFormat c 1 {fmt_name}")

        for i, name in enumerate(self.band_names):
            if name:
                x_params.append(f"BandName{i} c 1 {name}")
                
        for k, v in self.extra_data.items():
            x_params.append(f"ExtraData_{k} f 1 {v}")
        for k, v in self.extra_data_int.items():
            x_params.append(f"ExtraDataInt_{k} i 1 {v}")
        for k, v in self.extra_data_string.items():
            x_params.append(f"ExtraDataString_{k} c 1 {v}")
                
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
        add_array("BandStrobeTime", self.strobe_times, 'f')
        add_array("BandStrobeTimesUniversal", self.strobe_times_universal, 'f')
        add_array("BandIllumination", self.illumination, 'f')
            
        f.write(f"{len(x_params)}\n".encode('ascii'))
        for p in x_params:
            f.write(f"{p}\n".encode('ascii'))
            
        f.write(f"{len(binary_data)}\n".encode('ascii'))
        f.write(binary_data)
        
    def __str__(self) -> str:
        hist_short = self.history.replace('\n', ' ')[:100]
        desc_short = self.description.replace('\n', ' ')[:100]
        lines = [
            f"HIPS Image Summary:",
            f"  Dimensions: {self.width}x{self.height} pixels",
            f"  Bands: {self.bands}",
            f"  Format: {self.format.name} ({int(self.format)})",
            f"  MmPixel: {self.mm_pixel:.6f}",
            f"  ROI: {self.roi_width}x{self.roi_height} at ({self.roi_x}, {self.roi_y})",
            f"  History: {hist_short}..." if len(self.history) > 100 else f"  History: {self.history}",
            f"  Description: {desc_short}..." if len(self.description) > 100 else f"  Description: {self.description}",
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

def write(image: Union[HipsImage, np.ndarray], path: str, compression: Optional[str] = None):
    """
    Convenience function to write an image to a HIPS file.
    image can be a HipsImage object or a numpy array.
    """
    if isinstance(image, np.ndarray):
        img_obj = HipsImage()
        img_obj.pixels = image
        img_obj.write(path, compression)
    else:
        image.write(path, compression)

def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Inspect HIPS file header information.")
    parser.add_argument("path", help="Path to the .hips file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all parameters and full history")
    parser.add_argument("--history", action="store_true", help="Show full history and description")
    args = parser.parse_args()
    if not os.path.exists(args.path):
        print(f"Error: File {args.path} not found.")
        sys.exit(1)
    try:
        img = HipsImage.read_header(args.path)
        if args.verbose or args.history:
            print(f"HIPS Image: {args.path}")
            print(f"  Dimensions: {img.width}x{img.height} pixels")
            print(f"  Bands: {img.bands}")
            print(f"  Format: {img.format.name} ({int(img.format)})")
            print(f"  MmPixel: {img.mm_pixel:.6f}")
            print(f"  ROI: {img.roi_width}x{img.roi_height} at ({img.roi_x}, {img.roi_y})")
            print(f"\n--- History ---\n{img.history}")
            print(f"\n--- Description ---\n{img.description}")
            if args.verbose:
                print("\n--- Extended Parameters ---")
                if len(img.wavelengths) > 0:
                    print(f"  Wavelengths: {img.wavelengths.tolist()}")
                if img.band_names:
                    print(f"  Band Names: {img.band_names}")
                if len(img.strobe_times) > 0:
                    print(f"  Strobe Times: {img.strobe_times.tolist()}")
                if len(img.illumination) > 0:
                    print(f"  Illumination: {img.illumination.tolist()}")
                if img.extra_data:
                    print(f"  Extra Data: {img.extra_data}")
                if img.extra_data_int:
                    print(f"  Extra Data (Int): {img.extra_data_int}")
                if img.extra_data_string:
                    print(f"  Extra Data (String): {img.extra_data_string}")
                if img.camera_temperature != 0.0:
                    print(f"  Camera Temperature: {img.camera_temperature}")
                if img.id:
                    print(f"  ID: {img.id}")
                if img.freehand_layers_xml:
                    print(f"  Freehand Layers XML: {len(img.freehand_layers_xml)} bytes")
        else:
            print(img)
    except Exception as e:
        print(f"Error reading HIPS file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
