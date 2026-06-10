import os
from typing import List

from . import config
from .assert_utils import assert_argument_types
from .logging_utils import logger


@assert_argument_types
def safe_file_replacement(
        filename: str,
        lines: List[str], 
        encoding: str = config.ENCODING
) -> None:
    new_filename: str = filename + '.new'
    old_filename: str = filename + '.old'

    # Assuming new_file.writelines expects a list of strings, which lines is.
    with open(new_filename, 'w', encoding=encoding) as new_file:
        new_file.writelines(lines) # writelines typically takes Iterable[str]

    if os.path.exists(old_filename):
        os.remove(old_filename)
    if os.path.exists(filename):
        os.rename(filename, old_filename)
    os.rename(new_filename, filename)


@assert_argument_types
def get_lines(
        filename: str,
        encoding: str = config.ENCODING
) -> List[str]: 
    lines_read: List[str] = [] 

    if os.path.exists(filename):
        with open(filename, 'r', encoding=encoding) as file_handle: # file is a keyword
            for line in file_handle: 
                lines_read.append(line.replace('\n', '')) 
    else:
        # logger is called with an f-string, which is fine.
        # The problem statement does not specify logger's type, assume it's a callable taking str.
        logger(f"{filename} not found. Return void list.")

    return lines_read
