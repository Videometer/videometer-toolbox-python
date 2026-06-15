import argparse
import json
import pathlib
import shutil

PKG_DIR = pathlib.Path(__file__).parent.resolve()
DLL_FOLDER = PKG_DIR / "DLLs"
STAMP_PATH = DLL_FOLDER / "VM" / ".installed.json"
LOCK_PATH = PKG_DIR / "dlls.lock.json"


def clean_dlls():
    # Locates the 'DLLs' folder relative to this file
    dll_folder = DLL_FOLDER

    if dll_folder.exists() and dll_folder.is_dir():
        for file in dll_folder.iterdir():
            try:
                if file.is_file():
                    file.unlink()
                elif file.is_dir():
                    shutil.rmtree(file)
            except Exception as e:
                print(f"Failed to delete {file}: {e}")
        print(f"Cleaned: {dll_folder}")
    else:
        print("DLLs folder not found.")


def dll_info():
    """Print which DLLs are currently installed (versions + provenance)."""
    if not STAMP_PATH.exists():
        print("No DLLs installed (no install stamp at " + str(STAMP_PATH) + ").")
        if LOCK_PATH.exists():
            print("Run 'python tools/fetch_dlls.py' to provision them from dlls.lock.json.")
        return

    with open(STAMP_PATH, "r", encoding="utf-8") as f:
        stamp = json.load(f)

    print(f"Installed DLLs (from {STAMP_PATH}):")
    print(f"  framework : {stamp.get('framework')}")
    vz = stamp.get("vendored_zip", {})
    print(f"  vendored  : {vz.get('file')} (sha256 {str(vz.get('sha256'))[:16]}...) "
          f"- native Intel + .NET framework runtime libs")
    print("  NuGet packages:")
    for p in stamp.get("packages", []):
        print(f"    - {p['id']} {p['version']}")


def main():
    parser = argparse.ArgumentParser(prog="videometer")
    parser.add_argument("--clean-dll", action="store_true", help="Clear the DLLs folder")
    parser.add_argument(
        "--dll-info", action="store_true",
        help="Show which DLL versions are currently installed",
    )

    args = parser.parse_args()

    if args.clean_dll:
        clean_dlls()
    elif args.dll_info:
        dll_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
