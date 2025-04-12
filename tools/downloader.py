import os 
import subprocess as sub

from tools.macros import LIB_LINK
from tools.macros import _DOWNLOAD_DIR

def download_library(project):
    root = os.getcwd()
    os.chdir(_DOWNLOAD_DIR)

    if not os.path.exists(f"{_DOWNLOAD_DIR}/{project}"):
        command = f"git clone {LIB_LINK[project]}"
        print(command)
        sub.run(command, shell=True)
    else:
        print(f"{project} is existed")
    
    os.chdir(root)