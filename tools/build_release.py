"""Prepare a release: assemble DLLs, build the download bundle, build the (small) wheel.

Distribution model: the wheel published to PyPI is small and contains only ``dlls.lock.json``.
The DLLs are downloaded on first use of the clr backend from a GitHub release asset, verified
against ``runtime_bundle.sha256`` in the lock (see ``videometer/dll_provision.py``).

This script runs the maintainer side of that flow and verifies it is internally consistent:
  1. assemble DLLs from the feed + vendored zip      (tools/fetch_dlls.py)
  2. verify they match the lock                       (--check)
  3. build the download bundle + update the lock      (tools/package_dll_bundle.py)
  4. build the wheel + sdist                          (python -m build)
  5. verify the wheel is small (no DLLs) and that the lock it ships has runtime_bundle.sha256

It cannot create the GitHub release for you; it prints the remaining manual steps.

Usage:
    python tools/build_release.py
    python tools/build_release.py --version 2.0.3
"""

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PKG_DIR = REPO_ROOT / "src" / "videometer"
LOCK_PATH = PKG_DIR / "dlls.lock.json"
DIST = REPO_ROOT / "dist"
WHEEL_SANITY_LIMIT_MB = 5  # the wheel must NOT contain DLLs; it should be tiny.

sys.path.insert(0, str(REPO_ROOT / "tools"))
import fetch_dlls  # noqa: E402
import package_dll_bundle  # noqa: E402


def _run(cmd):
    print("+ " + " ".join(str(c) for c in cmd))
    if subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode != 0:
        raise SystemExit(f"command failed: {' '.join(str(c) for c in cmd)}")


def _clean_build_artifacts():
    """Remove stale build/ and *.egg-info so setuptools does not repackage old contents."""
    for path in [REPO_ROOT / "build", *(REPO_ROOT / "src").glob("*.egg-info")]:
        if path.exists():
            print(f"  cleaning {path.relative_to(REPO_ROOT)}")
            shutil.rmtree(path)


def verify_wheel(wheel_path):
    with zipfile.ZipFile(wheel_path) as z:
        members = z.namelist()
        dlls = [m for m in members if m.lower().endswith(".dll")]
        if dlls:
            raise SystemExit(
                f"Wheel unexpectedly contains {len(dlls)} DLL(s); it should ship only the lock. "
                "Check pyproject package-data."
            )
        lock_member = next((m for m in members if m.endswith("dlls.lock.json")), None)
        if not lock_member:
            raise SystemExit("Wheel does not contain dlls.lock.json")
        shipped_lock = json.loads(z.read(lock_member))

    if not shipped_lock.get("runtime_bundle", {}).get("sha256"):
        raise SystemExit("Lock shipped in the wheel has no runtime_bundle.sha256")

    size_mb = wheel_path.stat().st_size / (1024 * 1024)
    print(f"\nOK: wheel is {size_mb:.2f} MB, ships dlls.lock.json, contains no DLLs.")
    if size_mb > WHEEL_SANITY_LIMIT_MB:
        print(f"!! Wheel is larger than expected ({size_mb:.1f} MB) - did DLLs leak in?")
    return shipped_lock


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build a release (bundle + small wheel)")
    parser.add_argument("--version", help="Release version (defaults to pyproject version)")
    args = parser.parse_args(argv)

    print("== 1. assemble DLLs from the feed + vendored zip ==")
    fetch_dlls.provision(force=False)

    print("\n== 2. verify DLLs match the lock ==")
    if fetch_dlls.main(["--check"]) != 0:
        raise SystemExit("DLLs are not in sync with dlls.lock.json; aborting.")

    print("\n== 3. build download bundle + update lock ==")
    bundle_argv = ["--version", args.version] if args.version else []
    package_dll_bundle.main(bundle_argv)

    print("\n== 4. build wheel + sdist ==")
    _clean_build_artifacts()
    _run([sys.executable, "-m", "build"])

    print("\n== 5. verify built wheel ==")
    wheels = sorted(DIST.glob("*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise SystemExit("no wheel produced in dist/")
    shipped_lock = verify_wheel(wheels[-1])

    rb = shipped_lock["runtime_bundle"]
    print("\n== remaining manual steps ==")
    print(f"  1. Commit the updated dlls.lock.json (runtime_bundle.sha256 = {rb['sha256'][:16]}...).")
    print(f"  2. Create GitHub release and upload the bundle as an asset:")
    print(f"       gh release create v{_ver(args)} dist/{rb['file']} \\")
    print(f"         --title v{_ver(args)} --notes 'DLL bundle for videometer {_ver(args)}'")
    print(f"     (asset must resolve to {rb['url']})")
    print(f"  3. Publish the wheel:  twine upload {wheels[-1].relative_to(REPO_ROOT)}")
    return 0


def _ver(args):
    return args.version or package_dll_bundle._project_version()


if __name__ == "__main__":
    raise SystemExit(main())
