import pytest
import os
import shutil
import numpy as np
from src.path import get_local_path, listdir, get_parent_folder

# Setup a temporary directory structure for testing
TEST_DIR = "temp_test_path_dir"
SUB_DIR = os.path.join(TEST_DIR, "subdir")
FILE_IN_TEST_DIR = os.path.join(TEST_DIR, "file1.txt")
FILE_IN_SUB_DIR = os.path.join(SUB_DIR, "file2.py")
FILE_LOCAL_TO_SUB_DIR = os.path.join(SUB_DIR, "local.txt")


@pytest.fixture(scope="module", autouse=True)
def setup_teardown_test_dir():
    # Setup
    os.makedirs(SUB_DIR, exist_ok=True)
    with open(FILE_IN_TEST_DIR, "w") as f:
        f.write("test1")
    with open(FILE_IN_SUB_DIR, "w") as f:
        f.write("test2")
    with open(FILE_LOCAL_TO_SUB_DIR, "w") as f:
        f.write("local_test")

    yield

    # Teardown
    shutil.rmtree(TEST_DIR)


def test_get_local_path():
    # Assuming __file__ for test_path_utils.py is test/test_path_utils.py
    # For robustness, we use a known file within our test structure
    expected = os.path.join(SUB_DIR, "file2.py")
    # We pass FILE_IN_SUB_DIR as the 'local_filename' argument to get_local_path
    assert get_local_path("file2.py", FILE_IN_SUB_DIR) == expected


def test_listdir_no_extension():
    files = listdir(TEST_DIR)
    # Normalizes paths for comparison across OS
    normalized_files = sorted([os.path.normpath(f) for f in files])
    expected_files = sorted([os.path.normpath(f) for f in [FILE_IN_TEST_DIR, SUB_DIR]])
    print(f"{normalized_files=}")
    print(f"{expected_files=}")
    assert normalized_files == expected_files


def test_listdir_with_py_extension():
    files = listdir(SUB_DIR, extensions=[".py"])
    normalized_files = sorted([os.path.normpath(f) for f in files])
    expected_files = sorted([os.path.normpath(FILE_IN_SUB_DIR)])
    assert normalized_files == expected_files


def test_listdir_with_txt_extension():
    files = listdir(TEST_DIR, extensions=[".txt"])
    normalized_files = sorted([os.path.normpath(f) for f in files])
    expected_files = sorted([os.path.normpath(FILE_IN_TEST_DIR)])
    assert normalized_files == expected_files


def test_listdir_empty_dir():
    empty_subdir = os.path.join(TEST_DIR, "empty_subdir")
    os.makedirs(empty_subdir, exist_ok=True)
    assert listdir(empty_subdir) == []
    os.rmdir(empty_subdir)


def test_get_parent_folder():
    assert os.path.normpath(get_parent_folder(FILE_IN_SUB_DIR)) == os.path.normpath(SUB_DIR)
    assert os.path.normpath(get_parent_folder(FILE_IN_SUB_DIR, level=1)) == os.path.normpath(SUB_DIR)
    assert os.path.normpath(get_parent_folder(FILE_IN_SUB_DIR, level=2)) == os.path.normpath(TEST_DIR)


def test_get_parent_folder_level_zero_returns_filename():
    # Level 0 should return the filename itself
    assert os.path.normpath(get_parent_folder(FILE_IN_SUB_DIR, level=0)) == os.path.normpath(
        os.path.basename(FILE_IN_SUB_DIR))


def test_get_parent_folder_level_too_high():
    # Test with level higher than actual depth
    # This should return the root part of the path, often an empty string or '.' depending on os.path.split behavior
    # For 'temp_test_path_dir/subdir/file2.py', level 3 would be above 'temp_test_path_dir'
    # os.path.split('temp_test_path_dir') -> ('', 'temp_test_path_dir')
    # os.path.split('') -> ('', '')
    # So, get_parent_folder(TEST_DIR, level=1) is '', get_parent_folder(TEST_DIR, level=2) is ''
    path_level_3 = get_parent_folder(FILE_IN_SUB_DIR, level=3)
    path_level_4 = get_parent_folder(FILE_IN_SUB_DIR, level=4)  # even higher

    # The exact behavior for very high levels might depend on how os.path.split handles paths like "" or "."
    # A common expectation is it might return an empty string or the current directory indicator
    # For this library, let's assume it should effectively be the "highest" it can go, which is often an empty string
    # representing the relative root if the path itself was relative.
    root_of_relative_path = get_parent_folder(TEST_DIR,
                                              level=1)  # This should be "" if TEST_DIR is "temp_test_path_dir"

    assert path_level_3 == root_of_relative_path
    assert path_level_4 == root_of_relative_path


def test_listdir_order():
    folder = f"{TEST_DIR}/listdir_order"
    os.makedirs(folder, exist_ok=True)
    filenames = [f"{folder}/{i}.txt" for i in np.random.randint(0, 1E5, 5)]
    for filename in filenames:
        with open(filename, 'w') as file:
            pass

    ordered = sorted(filenames)

    normalized_files = [os.path.normpath(filename) for filename in ordered]
    expected_files = [os.path.normpath(filename) for filename in listdir(folder)]
    assert normalized_files == expected_files
