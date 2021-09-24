import os, glob
import subprocess
from subprocess import DEVNULL, STDOUT


abspath = os.path.abspath(__file__)
dir_ = os.path.dirname(abspath)

files = glob.glob(dir_ + "/_progress_board_tests/_test_progress_board_*.py")

for file_path in files:
    file_name = str(file_path.rsplit("/", maxsplit=1)[1])

    try:
        print("\033[0;33;40m Testing", file_name, end="...\r")
        subprocess.check_call(["pytest", file_path], stdout=DEVNULL, stderr=STDOUT)
    except subprocess.CalledProcessError:
        print("\033[0;31;40m Error in", file_name)
    else:
        print("\033[0;32;40m", file_name, "is correct")
