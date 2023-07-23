import os
import shutil
import time


def reset_folder(destination):
    # synchrnously and recursively delete the destination folder and all its contents, donot return until done
    if os.path.isdir(destination):
        shutil.rmtree(destination)
        while os.path.isdir(destination):
            time.sleep(4)
    os.mkdir(destination)
    while not os.path.isdir(destination):
        time.sleep(4)
