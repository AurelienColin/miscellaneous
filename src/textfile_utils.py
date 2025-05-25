import os

import rignak.local_config as config
from rignak.assert_utils import assert_argument_types
from rignak.logging_utils import logger


@assert_argument_types
def safe_file_replacement(
        filename: str,
        lines: list,
        encoding: str = config.ENCODING
) -> None:
    new_filename = filename + '.new'
    old_filename = filename + '.old'

    with open(new_filename, 'w', encoding=encoding) as new_file:
        new_file.writelines(lines)
    if os.path.exists(old_filename):
        os.remove(old_filename)
    if os.path.exists(filename):
        os.rename(filename, old_filename)
    os.rename(new_filename, filename)


@assert_argument_types
def get_lines(
        filename: str,
        encoding: str = config.ENCODING
) -> list:
    lines = []

    if os.path.exists(filename):
        with open(filename, 'r', encoding=encoding) as file:
            for line in file:
                lines.append(line.replace('\n', ''))
    else:
        logger(f"{filename} not found. Return void list.")

    return lines
