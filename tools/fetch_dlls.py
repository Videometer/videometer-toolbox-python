"""Provision the C# DLLs that the ``clr`` backend loads, from a single source of truth.

This is the only supported way to populate ``src/videometer/DLLs/VM``. It reads
``src/videometer/dlls.lock.json`` and assembles that folder from two sources:

* **Vendored zip** (``src/videometer/DLLs_vendored.zip``, committed in git) -- the native
  Intel libraries (ipp*/mkl*/libiomp5md) and the .NET desktop-runtime framework assemblies
  (PresentationCore, WindowsBase, System.IO.Packaging, System.Drawing.Common). These are not
  available as loose redistributable DLLs on NuGet and change very rarely.
* **NuGet feed** -- the VM.* managed assemblies (+ SixLabors.ImageSharp), restored at the exact
  versions pinned in the lock file via ``dotnet``.

The assembled folder is verified against the SHA256 hashes in the lock file, so what ends up on
disk is exactly what the lock describes. A stamp file (``DLLs/VM/.installed.json``) records which
lock produced the current install; if it already matches, the script is a fast no-op. Bumping a
version in the lock therefore forces a clean re-provision automatically -- no manual cleanup.

Usage::

    python tools/fetch_dlls.py            # provision if missing/stale
    python tools/fetch_dlls.py --force    # re-provision even if up to date
    python tools/fetch_dlls.py --check    # verify only; non-zero exit if missing/stale/corrupt

Requires the .NET SDK (``dotnet``) on PATH for the NuGet restore.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# tools/fetch_dlls.py -> repo root is the parent of tools/
REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "src" / "videometer"
LOCK_PATH = PKG_DIR / "dlls.lock.json"
NUGET_CONFIG = REPO_ROOT / "nuget.config"
DLLS_DIR = PKG_DIR / "DLLs"
VM_DIR = DLLS_DIR / "VM"
STAMP_PATH = VM_DIR / ".installed.json"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_lock():
    with open(LOCK_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def lock_fingerprint(lock):
    """A stable hash of the parts of the lock that determine the installed bytes.

    Comments and formatting are ignored, so reflowing the file does not trigger a re-provision.
    """
    relevant = {
        "framework": lock.get("framework"),
        "vendored_zip": lock.get("vendored_zip", {}).get("sha256"),
        "packages": sorted(
            (p["id"], p["version"]) for p in lock["nuget"]["packages"]
        ),
        "managed_assemblies": sorted(
            (a["file"], a["sha256"]) for a in lock["nuget"]["managed_assemblies"]
        ),
    }
    blob = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def is_up_to_date(lock):
    if not STAMP_PATH.exists():
        return False
    try:
        with open(STAMP_PATH, "r", encoding="utf-8") as f:
            stamp = json.load(f)
    except (OSError, ValueError):
        return False
    return stamp.get("lock_fingerprint") == lock_fingerprint(lock)


def verify_installed(lock):
    """Return a list of problems with the currently installed DLLs (empty == healthy)."""
    problems = []
    for asm in lock["nuget"]["managed_assemblies"]:
        dest = VM_DIR / asm["file"]
        if not dest.exists():
            problems.append(f"missing {asm['file']}")
        elif _sha256(dest) != asm["sha256"]:
            problems.append(f"sha256 mismatch {asm['file']}")
    # The vendored framework/native libs are validated as a set via the zip hash at install time;
    # here we just confirm a representative native lib is present.
    if not (VM_DIR / "mkl_core.1.dll").exists():
        problems.append("vendored native libraries not extracted (mkl_core.1.dll missing)")
    return problems


def _extract_vendored_zip(lock):
    zip_path = PKG_DIR / lock["vendored_zip"]["file"]
    if not zip_path.exists():
        raise SystemExit(f"ERROR: vendored zip not found: {zip_path}")
    actual = _sha256(zip_path)
    expected = lock["vendored_zip"]["sha256"]
    if actual != expected:
        raise SystemExit(
            f"ERROR: {zip_path.name} sha256 mismatch\n  expected {expected}\n  actual   {actual}"
        )
    # Entries are stored as "VM/<name>.dll"; extracting at DLLS_DIR lands them in DLLs/VM/.
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DLLS_DIR)
    print(f"  extracted vendored libraries from {zip_path.name}")


def _restore_managed(lock, staging):
    """dotnet-publish the pinned packages into ``staging`` and return that dir."""
    if shutil.which("dotnet") is None:
        raise SystemExit(
            "ERROR: 'dotnet' (the .NET SDK) is required to restore VM.* DLLs but was not found "
            "on PATH. Install the .NET SDK or run on a machine that has it."
        )
    if not NUGET_CONFIG.exists():
        raise SystemExit(
            f"ERROR: {NUGET_CONFIG.name} not found. The internal feed URL is not committed to "
            "this public repo. Copy nuget.config.template to nuget.config and set the real "
            "Videometer feed URL, then re-run."
        )
    proj_dir = staging / "proj"
    proj_dir.mkdir(parents=True, exist_ok=True)
    refs = "\n".join(
        f'    <PackageReference Include="{p["id"]}" Version="{p["version"]}" />'
        for p in lock["nuget"]["packages"]
    )
    csproj = f"""<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Library</OutputType>
    <TargetFramework>{lock["framework"]}</TargetFramework>
    <UseWindowsForms>true</UseWindowsForms>
    <UseWPF>true</UseWPF>
    <Nullable>disable</Nullable>
    <ImplicitUsings>disable</ImplicitUsings>
  </PropertyGroup>
  <ItemGroup>
{refs}
  </ItemGroup>
</Project>
"""
    (proj_dir / "fetch.csproj").write_text(csproj, encoding="utf-8")
    out = staging / "pub"
    cmd = [
        "dotnet", "publish", str(proj_dir / "fetch.csproj"),
        "-c", "Release", "-o", str(out),
        "--configfile", str(NUGET_CONFIG),
    ]
    print("  restoring VM.* packages via dotnet publish ...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stderr.write(res.stdout + "\n" + res.stderr + "\n")
        raise SystemExit("ERROR: dotnet publish failed (see output above)")
    return out


def _copy_and_verify_managed(lock, staging_out):
    for asm in lock["nuget"]["managed_assemblies"]:
        src = staging_out / asm["file"]
        if not src.exists():
            raise SystemExit(f"ERROR: expected assembly not produced by restore: {asm['file']}")
        actual = _sha256(src)
        if actual != asm["sha256"]:
            raise SystemExit(
                f"ERROR: {asm['file']} sha256 mismatch after restore\n"
                f"  expected {asm['sha256']}\n  actual   {actual}\n"
                f"  The feed returned different bytes than the lock pins. Update dlls.lock.json "
                f"if this version change is intended."
            )
        shutil.copy2(src, VM_DIR / asm["file"])
    print(f"  copied + verified {len(lock['nuget']['managed_assemblies'])} managed assemblies")


def _write_stamp(lock):
    stamp = {
        "lock_fingerprint": lock_fingerprint(lock),
        "framework": lock["framework"],
        "vendored_zip": lock["vendored_zip"],
        "packages": lock["nuget"]["packages"],
    }
    with open(STAMP_PATH, "w", encoding="utf-8") as f:
        json.dump(stamp, f, indent=2)


def provision(force=False):
    lock = _load_lock()
    if not force and is_up_to_date(lock) and not verify_installed(lock):
        print("DLLs already up to date with dlls.lock.json - nothing to do.")
        return
    print(f"Provisioning DLLs into {VM_DIR} ...")
    if VM_DIR.exists():
        shutil.rmtree(VM_DIR)
    VM_DIR.mkdir(parents=True, exist_ok=True)
    _extract_vendored_zip(lock)
    with tempfile.TemporaryDirectory(prefix="vm_fetch_dlls_") as tmp:
        staging_out = _restore_managed(lock, Path(tmp))
        _copy_and_verify_managed(lock, staging_out)
    _write_stamp(lock)
    print("Done. DLLs assembled and verified against dlls.lock.json.")


def ensure_dlls():
    """Provision DLLs if missing/stale. Safe to call from build hooks and test setup."""
    lock = _load_lock()
    if is_up_to_date(lock) and not verify_installed(lock):
        return
    provision(force=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Provision VM C# DLLs from dlls.lock.json")
    parser.add_argument("--force", action="store_true", help="Re-provision even if up to date")
    parser.add_argument(
        "--check", action="store_true",
        help="Verify only; exit non-zero if missing, stale, or corrupt (no changes)",
    )
    args = parser.parse_args(argv)

    lock = _load_lock()
    if args.check:
        stale = not is_up_to_date(lock)
        problems = verify_installed(lock)
        if stale:
            problems.append("install stamp does not match dlls.lock.json")
        if problems:
            print("DLLs NOT in sync with dlls.lock.json:")
            for p in problems:
                print(f"  - {p}")
            return 1
        print("DLLs are in sync with dlls.lock.json.")
        return 0

    provision(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
