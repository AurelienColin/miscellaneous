import os
import zipfile
from typing import Tuple, List # Added List for os.walk variables

def backup_py_files(
        repository_path: str,
        extensions: Tuple[str, ...] = ('.py', '.txt') # Changed from tuple to Tuple[str, ...]
) -> None:
    backup_filename: str = os.path.join(repository_path, os.path.basename(repository_path) + '.zip')
    with zipfile.ZipFile(backup_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder_name, _, current_filenames_in_walk in os.walk(repository_path):
            current_folder: str = folder_name
            for filename in current_filenames_in_walk:
                if os.path.splitext(filename)[1] in extensions:
                    full_path: str = os.path.join(current_folder, filename)
                    zipf.write(full_path, os.path.relpath(full_path, repository_path))
