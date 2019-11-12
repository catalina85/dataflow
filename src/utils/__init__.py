import os, sys

import numpy as np


def envs():
    for each in sys.path:
        print(each)

    cwd = os.getcwd()

    for dir in os.listdir(cwd):
        print(dir)

    try:
        import tensorflow as tf
        print(tf.__version__)

    except:
        pass

    finally:
        print(cwd)
        print(sys.version)


def Run():
    if (len(sys.argv) != 1):
        print("args error ...")
        sys.exit(0)

    envs()


if __name__ == "__main__":
    Run()
