import argparse
import pathlib
import shutil

def clean_dlls():
    # Locates the 'DLLs' folder relative to this file
    pkg_dir = pathlib.Path(__file__).parent.resolve()
    dll_folder = pkg_dir / "DLLs"

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

def main():
    parser = argparse.ArgumentParser(prog="videometer")
    parser.add_argument("--clean-dll", action="store_true", help="Clear the DLLs folder")
    
    args = parser.parse_args()

    if args.clean_dll:
        clean_dlls()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()