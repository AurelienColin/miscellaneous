import os
from typing import List, Optional, Tuple, Union

from .assert_utils import assert_argument_types


@assert_argument_types
def get_local_path(
        local_filename: str,
        filename: str
) -> str:
    return os.path.join(os.path.dirname(local_filename), filename)


# No @assert_argument_types for listdir as its signature is complex for the basic decorator
def listdir(folder: str, extensions: Optional[Union[List[str], Tuple[str, ...]]] = None) -> List[str]:
    # extensions can be None, or a list/tuple of strings like ['.txt', '.py']
    
    filenames_in_dir: List[str] = [os.path.join(folder, filename) for filename in os.listdir(folder)]
    
    if extensions is not None:
        # Ensure extensions is a tuple for the 'in' operator, if it's a list.
        # This is good practice for consistency, though 'in' works with lists too.
        valid_extensions: Union[List[str], Tuple[str, ...]] = extensions 
        filenames_in_dir = [filename for filename in filenames_in_dir if os.path.splitext(filename)[1] in valid_extensions]
        
    filenames_in_dir = sorted(filenames_in_dir)
    return filenames_in_dir


@assert_argument_types
def get_parent_folder(
        path: str,
        level: int = 1
) -> str:
    folder: str
    filename: str # Though filename is not used if level == 1 or level > 1 initially
    folder, filename = os.path.split(path)
    if level == 1:
        return folder
    elif level < 1: # This case seems odd, returning filename if level is 0 or negative
        return filename 
    else:
        # Recursive call, type checker will infer return type is str
        return get_parent_folder(folder, level=level - 1)
