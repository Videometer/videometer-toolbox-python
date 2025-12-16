import os
import requests
from zipfile import ZipFile

PATHVM = os.path.abspath(os.path.join(os.path.dirname(__file__)))


def unblockDLLs(path2dlls):
    """Unblock DLLs"""
    for filename in os.listdir(path2dlls):
        if filename.endswith(".dll"):
            try:
                os.remove(os.path.join(path2dlls, filename + ":" + "Zone.Identifier"))
            except:
                pass


def setupDlls():
    # Get the zip of DLLs
    path2zip = os.path.join(PATHVM, "DLLs.zip")
    path2GithubZip = r"https://github.com/Videometer/videometer-toolbox-python/raw/main/src/videometer/DLLs.zip?download="

    print("\nFetching DLLs... ", end=" ")

    r = requests.get(path2GithubZip)
    with open(path2zip, "wb") as code:
        code.write(r.content)
    print("Complete!")

    print("Unzipping... ", end="")
    # Unzip
    with ZipFile(path2zip, "r") as zObject:
        zObject.extractall(path=PATHVM)
    print("Complete!")

    print("Deleting Zip file... ", end="")
    # Delete zip file
    os.remove(path2zip)
    print("Complete!")

    print("Unblocking Dlls... ", end="")
    # Unlock all zip files
    path2dlls = os.path.join(PATHVM, "DLLs")
    intelDlls = os.path.join(path2dlls, "IPP2019Update1", "intel64")
    videometerDlls = os.path.join(path2dlls, "VM")

    unblockDLLs(intelDlls)
    unblockDLLs(videometerDlls)

    print("Complete!")
