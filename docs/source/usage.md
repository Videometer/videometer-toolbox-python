# Usage Guide

## Backend Selection

The `videometer` package supports two backends:
- `clr`: Legacy implementation requiring `pythonnet` and Videometer DLLs.
- `python`: Pure Python implementation using `hips_core`.

You can set the backend globally:

```python
import videometer
videometer.set_backend("python")
```

Or via the environment variable `VIDEOMETER_BACKEND`.

## Reading HIPS Images

```python
from videometer import hips

# Read an image
img = hips.read("path/to/image.hips")

# Access pixel data
pixels = img.PixelValues
print(f"Dimensions: {img.Width}x{img.Height} with {img.Bands} bands")
```

## Writing HIPS Images

```python
from videometer import hips
import numpy as np

# Create some dummy data (Height, Width, Bands)
data = np.random.rand(100, 100, 3).astype(np.float32)

# Write to HIPS
hips.write(data, "output.hips", compression="Uncompressed")
```

## Visualizing Images

```python
from videometer import hips

img = hips.read("image.hips")

# Show all bands
hips.show(img)

# Show RGB representation (requires clr backend for now)
hips.showRGB(img)
```
