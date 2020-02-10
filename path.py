import os


def get_local_file(src_file, dest_file):
    return os.path.join(os.path.split(os.path.abspath(src_file))[0], dest_file)


def create_path(path):
    path, ext = os.path.splitext(path)
    if not path:
        return
    if ext:
        create_path(os.path.split(path)[0])
    else:
        root, folder = os.path.split(path)
        if not os.path.exists(root):
            create_path(root)
        os.makedirs(os.path.join(path), exist_ok=True)
