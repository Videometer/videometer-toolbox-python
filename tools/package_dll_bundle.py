"""Package the assembled DLLs into the release bundle that end users download.

The wheel published to PyPI is tiny and does NOT contain the DLLs. Instead, on first use of the
clr backend, ``videometer/dll_provision.py`` downloads a single ``videometer-dlls-<version>.zip``
from the matching GitHub release and verifies it against the ``runtime_bundle.sha256`` recorded in
``dlls.lock.json``.

This tool builds that bundle from the already-assembled ``src/videometer/DLLs/VM`` (run
``tools/fetch_dlls.py`` first) and writes ``runtime_bundle`` (file, url, sha256) back into the lock
file so the wheel ships the correct download metadata.

Usage:
    python tools/package_dll_bundle.py                 # version from pyproject
    python tools/package_dll_bundle.py --version 2.0.3

After running: upload the produced dist/videometer-dlls-<version>.zip to the GitHub release
tagged v<version>, then build + publish the wheel.
"""

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "src" / "videometer"
VM_DIR = PKG_DIR / "DLLs" / "VM"
LOCK_PATH = PKG_DIR / "dlls.lock.json"
PYPROJECT = REPO_ROOT / "pyproject.toml"
DIST = REPO_ROOT / "dist"

URL_TEMPLATE = (
    "https://github.com/Videometer/videometer-toolbox-python/"
    "releases/download/v{version}/videometer-dlls-{version}.zip"
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _project_version():
    text = PYPROJECT.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise SystemExit("could not find version in pyproject.toml")
    return m.group(1)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the runtime DLL bundle + update the lock")
    parser.add_argument("--version", help="Release version (defaults to pyproject version)")
    args = parser.parse_args(argv)

    version = args.version or _project_version()

    dlls = sorted(VM_DIR.glob("*.dll"))
    if not dlls:
        raise SystemExit(
            f"No DLLs found in {VM_DIR}. Run 'python tools/fetch_dlls.py' first."
        )

    DIST.mkdir(exist_ok=True)
    bundle_name = f"videometer-dlls-{version}.zip"
    bundle_path = DIST / bundle_name

    # Build the zip deterministically (sorted entries, fixed timestamps) so the same DLLs always
    # produce the same sha256 - a rebuilt bundle then still matches the committed lock.
    members = [(f"VM/{d.name}", d) for d in dlls]
    stamp = VM_DIR / ".installed.json"
    if stamp.exists():
        members.append(("VM/.installed.json", stamp))
    members.sort(key=lambda m: m[0])

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for arcname, src in members:
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, src.read_bytes())

    digest = _sha256(bundle_path)
    size_mb = bundle_path.stat().st_size / (1024 * 1024)

    # Write runtime_bundle back into the lock.
    with open(LOCK_PATH, encoding="utf-8") as f:
        lock = json.load(f)
    lock.setdefault("runtime_bundle", {})
    lock["runtime_bundle"]["file"] = bundle_name
    lock["runtime_bundle"]["url"] = URL_TEMPLATE.format(version=version)
    lock["runtime_bundle"]["sha256"] = digest
    with open(LOCK_PATH, "w", encoding="utf-8") as f:
        json.dump(lock, f, indent=2)
        f.write("\n")

    print(f"Bundled {len(dlls)} DLLs -> dist/{bundle_name}  ({size_mb:.1f} MB)")
    print(f"sha256: {digest}")
    print("Updated dlls.lock.json -> runtime_bundle.{file,url,sha256}")
    print("\nNext:")
    print(f"  1. Tag + create GitHub release v{version}")
    print(f"  2. Upload dist/{bundle_name} as a release asset at:")
    print(f"       {lock['runtime_bundle']['url']}")
    print("  3. Commit the updated dlls.lock.json, then build + publish the wheel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
