import os

from .assert_utils import assert_argument_types


@assert_argument_types
def get_local_path(
        local_filename: str,
        filename: str
) -> str:
    return os.path.join(os.path.dirname(local_filename), filename)


def listdir(folder: str, extensions: (None, list, tuple) = None) -> list:
    filenames = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    if extensions is not None:
        filenames = [filename for filename in filenames if os.path.splitext(filename)[1] in extensions]
    filenames = sorted(filenames)
    return filenames


@assert_argument_types
def get_parent_folder(
        path: str,
        level: int = 1
) -> str:
    folder, filename = os.path.split(path)
    if level == 1:
        return folder
    elif level < 1:
        return filename
    else:
        return get_parent_folder(folder, level=level - 1)
