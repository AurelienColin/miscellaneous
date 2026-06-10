import os
from typing import Callable

try:
    from .logging_utils import Logger

    logger = Logger("logging") # Assuming Logger is appropriately typed in its own file or is Any
except ImportError:
    from logging import Logger # This is the standard library Logger

    logger = Logger("logging") 
    logger.error("ImportError on 'Logger'.")

try:
    from .assert_utils import ExistingFilename 
    # ExistingFilename would be typed as str or a specific TypeVar if defined as such in assert_utils
except ImportError:
    logger.error("ImportError on 'ExistingFilename'.")
    ExistingFilename = str # Fallback is str, so hints using ExistingFilename can use str

try:
    from .assert_utils import assert_argument_types
except ImportError:
    logger.error("ImportError on 'assert_argument_types'.")

    # Fallback decorator definition
    def assert_argument_types(function: Callable) -> Callable: # Changed from callable to Callable
        return function

if not os.environ.get('DISABLE_RIGNAK_BACKUP', 'False') == "True":
    try:
        import threading
        # These are imported for use, their own type hints are in their respective files.
        from .backup import backup_py_files 
        from .path import get_parent_folder

        # folder will be str as per get_parent_folder's return type hint
        folder: str = get_parent_folder(__file__, level=2) 
        # args=(folder,) is correct for backup_py_files(repository_path: str, ...)
        thread = threading.Thread(target=backup_py_files, args=(folder,))
        thread.start()

    except ImportError:
        logger.error("ImportError on 'backup_py_files'.")
