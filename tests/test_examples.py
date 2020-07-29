import os, sys, glob
import subprocess
from subprocess import DEVNULL, STDOUT


files = glob.glob("../examples/*/*.py")

for file_path in files:
    file_name = str(file_path.rsplit("/", maxsplit=1)[1])
    try:
        subprocess.check_call(["python", file_path], stdout=DEVNULL, stderr=STDOUT)
    except subprocess.CalledProcessError:
        print("\033[0;31;40m Error in", file_name)
    else:
        print("\033[0;32;40m", file_name, "is correct")
