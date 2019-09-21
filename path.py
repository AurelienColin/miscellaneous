import os

def get_local_file(src_file, dest_file):
    return os.path.join(os.path.split(os.path.abspath(src_file))[0], dest_file)
