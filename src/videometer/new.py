import struct
import numpy as np
import zlib
import io
import xml.etree.ElementTree as ET
from PIL import Image

# Enum mapping for code readability
class CompressionMode:
    RAW = 0
    GZ = 1
    PNG = 2
    JPEG = 3

class HipsImage:
    def __init__(self, filepath):
        self.filepath = filepath
        self.metadata = {}
        self.extended_params = {}
        self.image_data = None
        self.quantification = []  # List of dicts per band
        self.compression_mode = CompressionMode.RAW
        self._load()

    def _readline(self, f):
        return f.readline().decode('utf-8').strip()

    def _load(self):
        with open(self.filepath, 'rb') as f:
            # --- 1. Header Parsing (Same as before) ---
            if self._readline(f) != "HIPS": raise ValueError("Invalid HIPS")
            self._readline(f); self._readline(f) # Skip onm, snm
            
            frames = int(self._readline(f))
            self._readline(f) # odt
            self.height = int(self._readline(f))
            self.width = int(self._readline(f))
            
            # ROI (Region of Interest)
            self.roi_h = int(self._readline(f))
            self.roi_w = int(self._readline(f))
            self.roi_y = int(self._readline(f))
            self.roi_x = int(self._readline(f))
            
            self.format_int = int(self._readline(f))
            self.bands = int(self._readline(f))
            if frames > self.bands: self.bands = frames

            # History & Description
            hist_len = int(self._readline(f))
            self.metadata['history'] = f.read(hist_len).decode('utf-8').strip()
            if f.peek(1) == b'\n': f.read(1)
            desc_len = int(self._readline(f))
            self.metadata['description'] = f.read(desc_len).decode('utf-8').strip()
            if f.peek(1) == b'\n': f.read(1)

            # --- 2. Extended Parameters ---
            self._read_extended_params(f)

            # --- 3. Detect Compression & Quantification ---
            # Parse Quantification XML if present
            self._parse_quantification()
            
            # Heuristic: If Quantification is present or specific Format Ints are used, 
            # we assume chunks. (Note: Real implementation would decode format_int fully)
            # Users might need to set this manually if format_int mapping is unknown.
            if "BandQuantification" in self.extended_params:
                # If quantified, it usually implies PNG or GZ compression in VideometerLab
                # We default to PNG if not explicitly known, or attempt to detect.
                # Ideally, we read this from format_int using a helper we don't have.
                # For this reader, we assume PNG if Q param exists, as per 'ParameterHelper.cs' presets.
                self.compression_mode = CompressionMode.PNG 
                
            # Override if history suggests otherwise or manual override (omitted)

            # --- 4. Read Image Data ---
            if self.compression_mode == CompressionMode.RAW and not self.quantification:
                self._load_raw_data(f)
            else:
                self._load_compressed_chunks(f)

    def _read_extended_params(self, f):
        num_params = int(self._readline(f))
        x_param_defs = []
        for _ in range(num_params):
            line = self._readline(f)
            parts = line.split(' ', 3)
            name = parts[0]
            fmt_char = parts[1][0]
            count = int(parts[2])
            val_str = parts[3] if len(parts) > 3 else ""

            if count == 1:
                self.extended_params[name] = self._parse_scalar(fmt_char, val_str)
            else:
                offset = int(val_str)
                x_param_defs.append({'name': name, 'fmt': fmt_char, 'count': count, 'offset': offset})

        binary_size = int(self._readline(f))
        while f.peek(1) == b' ': f.read(1)
        if f.peek(1) == b'\n': f.read(1)

        if num_params > 0 and binary_size > 0:
            bin_start = f.tell()
            for xp in x_param_defs:
                f.seek(bin_start + xp['offset'])
                dtype = self._map_format_to_struct(xp['fmt'])
                calc_size = struct.calcsize(dtype) * xp['count']
                data = f.read(calc_size)
                unpacked = struct.unpack(f"{xp['count']}{dtype}", data)
                
                # Special handling for Strings (format 'c')
                if xp['fmt'] == 'c':
                    self.extended_params[xp['name']] = b''.join(unpacked).decode('ascii')
                else:
                    self.extended_params[xp['name']] = unpacked
            f.seek(bin_start + binary_size)

    def _parse_quantification(self):
        """Parses the BandQuantification XML string into a list of dicts."""
        xml_str = self.extended_params.get("BandQuantification")
        if not xml_str:
            return

        try:
            # It's an array of params, usually <ArrayOfQuantificationParameters>
            root = ET.fromstring(xml_str)
            for item in root.findall("QuantificationParameters"):
                q_params = {
                    'Q_Min': float(item.get('Q_Min')),
                    'Q_Max': float(item.get('Q_Max')),
                    'Q': int(item.get('Q'))
                }
                self.quantification.append(q_params)
        except ET.ParseError:
            print("Warning: Failed to parse BandQuantification XML")

    def _load_compressed_chunks(self, f):
        """Reads band-sequential chunks."""
        bands_data = []

        for b in range(self.bands):
            # Read chunk size (4 bytes int)
            chunk_size_bytes = f.read(4)
            if not chunk_size_bytes: break
            chunk_size = struct.unpack('<i', chunk_size_bytes)[0]

            # Read compressed data
            compressed_data = f.read(chunk_size)

            # Decompress
            band_arr = self._decompress_chunk(compressed_data)

            # De-Quantify (Scale to Float)
            if self.quantification and b < len(self.quantification):
                qp = self.quantification[b]
                band_arr = self._dequantify(band_arr, qp)

            bands_data.append(band_arr)

        if bands_data:
            self.image_data = np.stack(bands_data)

    def _decompress_chunk(self, data):
        """Decompresses GZ, PNG, or JPEG data into a numpy array."""
        
        # Helper to check signature if mode is ambiguous, 
        # but here we rely on self.compression_mode
        
        if self.compression_mode == CompressionMode.GZ:
            # GZip: raw bytes, packed (no stride)
            raw = zlib.decompress(data)
            # Assume 32-bit float if GZ and no quantification, 
            # but if quantification exists, it depends on Q. 
            # Simple heuristic:
            dtype = np.float32
            if self.quantification:
                # If quantified, GZ stores the INT/Byte representation
                q = self.quantification[0]['Q'] # Assume uniform Q for dtype guess
                dtype = np.uint8 if q <= 8 else np.uint16
            
            arr = np.frombuffer(raw, dtype=dtype)
            return arr.reshape((self.height, self.width))

        elif self.compression_mode in [CompressionMode.PNG, CompressionMode.JPEG]:
            # Use Pillow for PNG/JPEG
            with io.BytesIO(data) as bio:
                img = Image.open(bio)
                # Ensure we read data
                img.load() 
                arr = np.array(img)
                # PIL might return (H, W) or (H, W, C). HIPS bands are mono.
                return arr

        return np.array([])

    def _dequantify(self, arr, qp):
        """
        Converts integer array to float using quantification parameters.
        Val = (IntVal / Factor) + Min
        """
        Q = qp['Q']
        Q_Min = qp['Q_Min']
        Q_Max = qp['Q_Max']
        
        # Factor calculation from ImageQuantification.cs
        # float maxQ = (float)Math.Pow(2.0, (double)Q) - 1.0f;
        # float factor = maxQ / range;
        
        max_q = (2.0 ** Q) - 1.0
        val_range = Q_Max - Q_Min
        factor = max_q / val_range

        # Convert to float
        arr_f = arr.astype(np.float32)
        
        # Apply formula: IppNative.DivScalar(dst, factor) -> dst / factor
        arr_f /= factor
        
        # Apply formula: IppNative.AddScalar(dst, min) -> dst + min
        arr_f += Q_Min
        
        return arr_f

    def _load_raw_data(self, f):
        dtype, pixel_bytes = self._get_pixel_format(self.format_int)

        # 1. Calculate the size of the data block on disk
        current_pos = f.tell()
        f.seek(0, 2) # Seek to end
        end_pos = f.tell()
        f.seek(current_pos) # Seek back to data start
        
        total_data_bytes = end_pos - current_pos
        
        # 2. Deduce the Line Length (Stride) used by the C# writer
        # Total Bytes = Bands * Height * LineLength
        bytes_per_band = total_data_bytes // self.bands
        line_length = bytes_per_band // self.height
        
        # Validation
        min_line_length = self.width * pixel_bytes
        if line_length < min_line_length:
             raise ValueError(f"Calculated stride ({line_length}) is smaller than image width ({min_line_length}). File might be truncated or format is wrong.")
        
        # 3. Read the full data block
        raw_data = f.read(total_data_bytes)
        
        # 4. Load into Numpy with the detected stride
        # Shape: (Bands, Height, Stride_Width)
        img_buffer = np.frombuffer(raw_data, dtype=np.uint8)
        
        try:
            img_buffer = img_buffer.reshape((self.bands, self.height, line_length))
        except ValueError:
            raise ValueError(f"Data size mismatch. Expected {self.bands}x{self.height}x{line_length}={self.bands*self.height*line_length}, got {len(raw_data)} bytes.")

        # 5. Crop padding and convert to actual type
        valid_row_bytes = self.width * pixel_bytes
        img_buffer = img_buffer[:, :, :valid_row_bytes]
        
        # View as correct dtype
        # Note: We must make a copy/contiguous array if the stride is not a multiple of the itemsize
        # but usually cropping to valid_row_bytes handles the view logic correctly.
        self.image_data = img_buffer.view(dtype).reshape((self.bands, self.height, self.width))

    def _parse_scalar(self, fmt, val):
        if fmt == 'i' or fmt == 's' or fmt == 'b': return int(val)
        if fmt == 'f' or fmt == 'd': return float(val)
        return val

    def _map_format_to_struct(self, fmt):
        mapping = {'c': 'c', 'b': 'B', 's': 'h', 'i': 'i', 'f': 'f', 'd': 'd'}
        return mapping.get(fmt, 'B')

    def _get_pixel_format(self, fmt_int):
        # Mappings based on VMImage / HipsIO logic
        # You may need to refine these integer keys based on specific Videometer Enums
        if fmt_int == 0: return np.uint8, 1   # BytePixel
        if fmt_int == 1: return np.uint16, 2  # Int16Pixel / Short (unsigned in VMImage)
        if fmt_int == 2: return np.uint32, 4 # Int32Pixel
        if fmt_int == 3: return np.float32, 4 # FloatPixel
        if fmt_int == 6: return np.double, 8 # DoublePixel
        if fmt_int == 35: return np.uint8, 1   # ByteRGB (handled as 3 bands)
        return np.uint8, 1

    # Helpers
    def _parse_scalar(self, fmt, val):
        if fmt in ['i', 's', 'b']: return int(val)
        if fmt in ['f', 'd']: return float(val)
        return val

    def _map_format_to_struct(self, fmt):
        mapping = {'c': 'c', 'b': 'B', 's': 'h', 'i': 'i', 'f': 'f', 'd': 'd'}
        return mapping.get(fmt, 'B')