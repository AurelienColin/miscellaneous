import os
import functools

try:
    import win32com.client
except ImportError:
    print('Will not be able to translate links')


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


@functools.lru_cache(maxsize=None)
def convert_link(link):
    shell = win32com.client.Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(link)
    return shortcut.Targetpath


def list_dir(root):
    folders = [folder for folder in os.listdir(root) if
               os.path.isdir(os.path.join(root, folder)) or folder.endswith('.lnk')]
    print('l30', folders)
    for i, folder in enumerate(folders):
        if folder.endswith('.lnk'):
            folders[i] = convert_link(os.path.join(root, folder))
        else:
            folders[i] = os.path.join(root, folder)
    print('l35', folders)
    return folders
