import os

# Backend selection: 'clr' or 'python'
# Default to 'clr' for backward compatibility
_BACKEND = os.environ.get("VIDEOMETER_BACKEND", "clr")

def set_backend(backend):
    global _BACKEND
    if backend not in ["clr", "python"]:
        raise ValueError("Backend must be 'clr' or 'python'")
    _BACKEND = backend

def get_backend():
    return _BACKEND
