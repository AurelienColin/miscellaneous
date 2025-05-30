import pytest
import os
import shutil
from rignak.textfile_utils import get_lines, safe_file_replacement
from rignak.config import ENCODING # Assuming ENCODING is accessible

TEST_DIR_TEXT = "temp_test_text_utils_dir"
TEST_FILE = os.path.join(TEST_DIR_TEXT, "test_file.txt")

@pytest.fixture(scope="module", autouse=True)
def setup_teardown_test_files():
    # Setup
    os.makedirs(TEST_DIR_TEXT, exist_ok=True)
    # Create initial file for get_lines and safe_file_replacement
    initial_content = ["line1\n", "line2\n", "line3\n"]
    with open(TEST_FILE, "w", encoding=ENCODING) as f:
        f.writelines(initial_content)
        
    yield # This is where the testing happens

    # Teardown
    shutil.rmtree(TEST_DIR_TEXT)

def test_get_lines_existing_file():
    lines = get_lines(TEST_FILE)
    #writelines adds \n, get_lines removes them.
    assert lines == ["line1", "line2", "line3"]

def test_get_lines_non_existing_file():
    lines = get_lines(os.path.join(TEST_DIR_TEXT,"non_existent.txt"))
    assert lines == []

def test_safe_file_replacement_normal():
    new_lines_content = ["newline1", "newline2"] # content for writelines should end with \n
    new_lines_for_write = [line + "\n" for line in new_lines_content]

    safe_file_replacement(TEST_FILE, new_lines_for_write)
    
    # Verify content
    assert get_lines(TEST_FILE) == new_lines_content
    
    # Verify backup file exists
    old_file = TEST_FILE + ".old"
    assert os.path.exists(old_file)
    # Verify content of backup, should be original content
    # Original content was ["line1\n", "line2\n", "line3\n"], so get_lines returns them without \n
    assert get_lines(old_file) == ["line1", "line2", "line3"] 

def test_safe_file_replacement_creates_new():
    new_file_path = os.path.join(TEST_DIR_TEXT, "newly_created_file.txt")
    new_lines_content = ["new_file_line1", "new_file_line2"]
    new_lines_for_write = [line + "\n" for line in new_lines_content]

    # Ensure file does not exist initially
    if os.path.exists(new_file_path):
        os.remove(new_file_path)
    
    safe_file_replacement(new_file_path, new_lines_for_write)
    
    assert get_lines(new_file_path) == new_lines_content
    # .old file should not exist as original file didn't exist
    assert not os.path.exists(new_file_path + ".old")

def test_safe_file_replacement_existing_old_file():
    # Simulate an existing .old file
    old_file_path = TEST_FILE + ".old"
    # Re-create initial state for TEST_FILE before this specific test
    initial_content_for_test = ["line1_alt\n", "line2_alt\n"]
    with open(TEST_FILE, "w", encoding=ENCODING) as f:
        f.writelines(initial_content_for_test)

    # Create a dummy .old file
    with open(old_file_path, "w", encoding=ENCODING) as f:
        f.write("dummy_old_content\n")
        
    new_lines_content = ["final_line1", "final_line2"]
    new_lines_for_write = [line + "\n" for line in new_lines_content]
    
    safe_file_replacement(TEST_FILE, new_lines_for_write)
    
    assert get_lines(TEST_FILE) == new_lines_content
    # The .old file should be overwritten with the content of TEST_FILE before replacement
    assert get_lines(old_file_path) == ["line1_alt", "line2_alt"]
