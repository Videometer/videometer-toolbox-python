"""Download the C# DLLs on first use of the clr backend.

The published wheel is small and does not contain the DLLs. The first time the clr backend is
used, ``ensure_runtime_dlls()`` downloads the versioned bundle recorded in ``dlls.lock.json``
(``runtime_bundle.url``), verifies its SHA256, and extracts it into ``DLLs/VM``. Subsequent runs
are a no-op.

This is the end-user counterpart to ``tools/fetch_dlls.py``: it needs only ``requests`` and a
public download URL -- no .NET SDK and no access to the internal NuGet feed. If a developer has
already assembled the DLLs via ``tools/fetch_dlls.py``, the download is skipped (the existing
DLLs are detected by hash).
"""

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
LOCK_PATH = PKG_DIR / "dlls.lock.json"
DLLS_DIR = PKG_DIR / "DLLs"
VM_DIR = DLLS_DIR / "VM"
STAMP_PATH = VM_DIR / ".installed.json"

_KEY_ASSEMBLY = "VM.Image.dll"          # representative managed assembly
_KEY_NATIVE = "mkl_core.1.dll"          # representative vendored native lib


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_lock():
    with open(LOCK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _expected_assembly_hash(lock, filename):
    for asm in lock.get("nuget", {}).get("managed_assemblies", []):
        if asm["file"] == filename:
            return asm["sha256"]
    return None


def is_installed(lock=None):
    """True if a valid, matching DLL set is already present (downloaded or dev-assembled)."""
    if not (VM_DIR / _KEY_ASSEMBLY).is_file() or not (VM_DIR / _KEY_NATIVE).is_file():
        return False
    lock = lock or _load_lock()
    expected = _expected_assembly_hash(lock, _KEY_ASSEMBLY)
    # If the lock pins the key assembly's hash, require a match; otherwise presence is enough.
    if expected is None:
        return True
    return _sha256(VM_DIR / _KEY_ASSEMBLY) == expected


def ensure_runtime_dlls():
    """Download + extract the DLL bundle if it is not already present. No-op when installed."""
    lock = _load_lock()
    if is_installed(lock):
        return

    bundle = lock.get("runtime_bundle", {})
    url = bundle.get("url")
    expected_sha = bundle.get("sha256")
    if not url or not expected_sha:
        raise RuntimeError(
            "dlls.lock.json has no runtime_bundle url/sha256. For a release this must be set by "
            "tools/package_dll_bundle.py; in a source checkout run 'python tools/fetch_dlls.py' "
            "to assemble the DLLs locally instead."
        )

    import requests  # declared dependency; imported lazily so the python backend stays light

    DLLS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vm_dll_dl_") as tmp:
        tmp_zip = Path(tmp) / bundle.get("file", "videometer-dlls.zip")
        print(f"Fetching Videometer DLLs from {url} ...", flush=True)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp_zip, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    fh.write(chunk)

        actual_sha = _sha256(tmp_zip)
        if actual_sha != expected_sha:
            raise RuntimeError(
                "Downloaded DLL bundle failed integrity check.\n"
                f"  url      {url}\n  expected {expected_sha}\n  actual   {actual_sha}"
            )

        # Replace any existing VM dir, then extract (entries are stored as "VM/<name>").
        if VM_DIR.exists():
            shutil.rmtree(VM_DIR)
        with zipfile.ZipFile(tmp_zip, "r") as z:
            z.extractall(DLLS_DIR)

    print("Videometer DLLs ready.", flush=True)
