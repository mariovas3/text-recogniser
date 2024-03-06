import os
from contextlib import contextmanager


@contextmanager
def change_wd(to_dir):
    curdir = os.getcwd()
    try:
        os.chdir(to_dir)
        yield
    finally:
        os.chdir(curdir)
