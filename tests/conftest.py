"""Shared pytest setup for the CLR backend.

Previously every CLR-dependent test module repeated the same block to compute the DLL paths,
prepend them to PATH, load the pythonnet ``coreclr`` runtime, and ``clr.AddReference`` the VM
assemblies. That boilerplate lived in 6 files and had drifted out of sync. It now lives here.

This module executes at import time -- pytest imports ``conftest.py`` before collecting the test
modules in this directory -- so module-level ``clr.AddReference(...)`` calls in the tests still
work. The DLLs themselves are provisioned from ``dlls.lock.json`` (a no-op once present), so a
fresh checkout can run the suite without any manual DLL setup.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

REPO_ROOT = Path(__file__).resolve().parent.parent
DLL_PATH = REPO_ROOT / "src" / "videometer" / "DLLs" / "VM"

# 1. Ensure the DLLs described by dlls.lock.json are present (no-op if already provisioned).
sys.path.insert(0, str(REPO_ROOT / "tools"))
try:
    import fetch_dlls

    fetch_dlls.ensure_dlls()
except SystemExit as exc:
    # e.g. the .NET SDK is unavailable; CLR-dependent tests will then fail with a clear error.
    print(f"[conftest] could not provision DLLs: {exc}")

# 2. Make the assembled DLLs discoverable, then load the CLR runtime once.
os.environ["PATH"] = str(DLL_PATH) + os.pathsep + os.environ["PATH"]
sys.path.append(str(DLL_PATH))
if sys.platform == "win32" and hasattr(os, "add_dll_directory") and DLL_PATH.is_dir():
    os.add_dll_directory(str(DLL_PATH))

import pythonnet

if pythonnet.get_runtime_info() is None:
    pythonnet.load("coreclr")
