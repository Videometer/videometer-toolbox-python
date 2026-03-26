"""
Configuration module for the Videometer toolbox.
This module manages global settings, such as the active backend.
"""

import os

# Backend selection: 'clr' or 'python'
# Default to 'clr' for backward compatibility
_BACKEND = os.environ.get("VIDEOMETER_BACKEND", "clr")

def set_backend(backend):
    """Sets the active backend for HIPS file operations.

    Args:
        backend (str): The backend to use. Must be either 'clr' or 'python'.
            'clr' uses the legacy C# implementation via pythonnet.
            'python' uses the pure Python implementation (hips_core).

    Raises:
        ValueError: If an invalid backend name is provided.
    """
    global _BACKEND
    if backend not in ["clr", "python"]:
        raise ValueError("Backend must be 'clr' or 'python'")
    _BACKEND = backend

def get_backend():
    """Returns the currently active backend.

    Returns:
        str: The active backend ('clr' or 'python').
    """
    return _BACKEND
