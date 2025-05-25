import os

try:
    from rignak.logging_utils import Logger

    logger = Logger("logging")
except ImportError:
    from logging import Logger

    logger = Logger("logging")
    logger.error("ImportError on 'Logger'.")

try:
    from rignak.assert_utils import ExistingFilename
except ImportError:
    logger.error("ImportError on 'ExistingFilename'.")
    ExistingFilename = str

try:
    from rignak.assert_utils import assert_argument_types
except ImportError:
    logger.error("ImportError on 'assert_argument_types'.")


    def assert_argument_types(function: callable) -> callable:
        return function

if not os.environ.get('DISABLE_rignak_BACKUP', 'False') == "True":
    try:
        import threading
        from rignak.backup import backup_py_files
        from rignak.path import get_parent_folder

        folder = get_parent_folder(__file__, level=2)
        thread = threading.Thread(target=backup_py_files, args=(folder,))
        thread.start()

    except ImportError:
        logger.error("ImportError on 'backup_py_files'.")
