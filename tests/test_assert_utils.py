import pytest
import os
from src.assert_utils import assert_type, ExistingFilename, assert_argument_types

def test_assert_type_correct():
    assert_type("variable", 10, int)
    assert_type("variable", "hello", str)
    assert_type("variable", None, type(None))
    assert_type("variable", lambda x: x, callable)
    assert_type("variable", (1, 2), tuple)
    assert_type("variable", [1, 2], list)
    assert_type("variable", {1, 2}, set)
    assert_type("variable", {"a":1}, dict)


def test_assert_type_incorrect():
    with pytest.raises(AssertionError):
        assert_type("variable", 10, str)
    with pytest.raises(AssertionError):
        assert_type("variable", "hello", int)
    with pytest.raises(AssertionError):
        assert_type("variable", None, int)
    with pytest.raises(AssertionError):
        assert_type("variable", 123, callable)

def test_existing_filename():
    # Create a dummy file
    dummy_file_name = "test_dummy_file.txt"
    with open(dummy_file_name, "w") as f:
        f.write("test")
    
    instance = ExistingFilename(dummy_file_name)
    assert instance == dummy_file_name
    os.remove(dummy_file_name)

def test_existing_filename_not_found():
    with pytest.raises(AssertionError):
        ExistingFilename("non_existent_file.txt")

@assert_argument_types
def sample_function_for_decorator(a: int, b: str) -> str:
    return f"{a}{b}"

def test_assert_argument_types_decorator_correct():
    result = sample_function_for_decorator(10, "hello")
    assert result == "10hello"

def test_assert_argument_types_decorator_incorrect():
    with pytest.raises(AssertionError):
        sample_function_for_decorator("wrong", "hello") # a should be int
    with pytest.raises(AssertionError):
        sample_function_for_decorator(10, 20) # b should be str
        
# Test for None type hint
@assert_argument_types
def function_with_none_type(a: None) -> None:
    assert a is None

def test_assert_argument_types_with_none_correct():
    function_with_none_type(None)

def test_assert_argument_types_with_none_incorrect():
    with pytest.raises(AssertionError):
        function_with_none_type("not none")

# Test for tuple of types
@assert_argument_types
def function_with_tuple_type(a: (int, str)) -> None:
    assert isinstance(a, (int, str))

def test_assert_argument_types_with_tuple_correct():
    function_with_tuple_type(10)
    function_with_tuple_type("hello")

def test_assert_argument_types_with_tuple_incorrect():
    with pytest.raises(AssertionError):
        function_with_tuple_type(10.5) # float, not int or str
        
# Test for ExistingFilename in decorator
@assert_argument_types
def function_with_existing_filename(f: ExistingFilename) -> str:
    return str(f)

def test_assert_argument_types_with_existing_filename_correct():
    dummy_file_name = "test_dummy_decorator.txt"
    with open(dummy_file_name, "w") as f:
        f.write("test")
    
    result = function_with_existing_filename(dummy_file_name)
    assert result == dummy_file_name
    os.remove(dummy_file_name)

def test_assert_argument_types_with_existing_filename_incorrect():
    with pytest.raises(AssertionError):
        function_with_existing_filename("non_existent_decorator.txt")
