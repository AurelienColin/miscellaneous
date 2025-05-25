import os
import zipfile


def backup_py_files(
        repository_path: str,
        extensions: tuple = ('.py', '.txt')
) -> None:
    backup_filename = os.path.join(repository_path, os.path.basename(repository_path) + '.zip')
    with zipfile.ZipFile(backup_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder, _, filenames in os.walk(repository_path):
            for filename in filenames:
                if os.path.splitext(filename)[1] in extensions:
                    full_path = os.path.join(folder, filename)
                    zipf.write(full_path, os.path.relpath(full_path, repository_path))
