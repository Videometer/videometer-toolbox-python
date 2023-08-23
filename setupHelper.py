import os


PATHVM = os.path.abspath(os.path.join(os.path.dirname(__file__)))

intelDlls = os.path.join(PATHVM,"IPP2019Update1","intel64")
videometerDlls = os.path.join(PATHVM,"VM")

unblockDLLs(intelDlls)
unblockDLLs(videometerDlls)


def unblockDLLs(path2dlls):
    """Unblock DLLs"""
    for filename in os.listdir(path2dlls):
        if filename.endswith(".dll"):
            try:
                os.remove(os.path.join(pathDLL,filename+":"+"Zone.Identifier"))
            except:
                pass


