import pytest
import os
import typing
from rignak.assert_utils import assert_type, ExistingFilename, assert_argument_types

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

    dummy_file_name = ExistingFilename(dummy_file_name)
    result = function_with_existing_filename(dummy_file_name)
    assert result == dummy_file_name
    os.remove(dummy_file_name)

def test_assert_argument_types_with_existing_filename_incorrect():
    with pytest.raises(AssertionError):
        dummy_file_name = ExistingFilename("non_existent_decorator.txt")
        function_with_existing_filename(dummy_file_name)

# --- Tests for assert_type with typing library ---

def test_assert_type_typing_list():
    assert_type("var_list_correct1", [1, 2], typing.List)
    assert_type("var_list_correct2", [1, 2], typing.List[int]) # Checks origin
    with pytest.raises(AssertionError):
        assert_type("var_list_incorrect", {1, 2}, typing.List)
    with pytest.raises(AssertionError):
        assert_type("var_list_incorrect2", {1, 2}, typing.List[int])

def test_assert_type_typing_dict():
    assert_type("var_dict_correct1", {"a": 1}, typing.Dict)
    assert_type("var_dict_correct2", {"a": 1}, typing.Dict[str, int]) # Checks origin
    with pytest.raises(AssertionError):
        assert_type("var_dict_incorrect", [1, 2], typing.Dict)
    with pytest.raises(AssertionError):
        assert_type("var_dict_incorrect2", [1, 2], typing.Dict[str, int])

def test_assert_type_typing_tuple():
    assert_type("var_tuple_correct1", (1, "a"), typing.Tuple)
    assert_type("var_tuple_correct2", (1, "a"), typing.Tuple[int, str]) # Checks origin
    with pytest.raises(AssertionError):
        assert_type("var_tuple_incorrect", [1, "a"], typing.Tuple)
    with pytest.raises(AssertionError):
        assert_type("var_tuple_incorrect2", [1, "a"], typing.Tuple[int, str])

def test_assert_type_typing_set():
    assert_type("var_set_correct1", {1, 2}, typing.Set)
    assert_type("var_set_correct2", {1, 2}, typing.Set[int]) # Checks origin
    with pytest.raises(AssertionError):
        assert_type("var_set_incorrect", [1, 2], typing.Set)
    with pytest.raises(AssertionError):
        assert_type("var_set_incorrect2", [1, 2], typing.Set[int])

def test_assert_type_typing_optional():
    assert_type("var_optional_correct1", "hello", typing.Optional[str])
    assert_type("var_optional_correct2", None, typing.Optional[str])
    with pytest.raises(AssertionError):
        assert_type("var_optional_incorrect", 123, typing.Optional[str])

def test_assert_type_typing_union():
    assert_type("var_union_correct1", "hello", typing.Union[str, int])
    assert_type("var_union_correct2", 123, typing.Union[str, int])
    assert_type("var_union_correct3", None, typing.Union[str, int, None])
    # Test with Optional as part of Union, which is common
    assert_type("var_union_optional_correct", "hello", typing.Union[str, typing.Optional[int]])
    assert_type("var_union_optional_correct2", None, typing.Union[str, typing.Optional[int]]) # None matches Optional[int] part
    assert_type("var_union_optional_correct3", 123, typing.Union[str, typing.Optional[int]])

    with pytest.raises(AssertionError):
        assert_type("var_union_incorrect", 12.3, typing.Union[str, int])
    with pytest.raises(AssertionError):
        assert_type("var_union_optional_incorrect", 12.3, typing.Union[str, typing.Optional[int]])


def test_assert_type_typing_any():
    assert_type("var_any_correct1", "hello", typing.Any)
    assert_type("var_any_correct2", 123, typing.Any)
    assert_type("var_any_correct3", None, typing.Any)
    assert_type("var_any_correct4", [1, 2], typing.Any)
    assert_type("var_any_correct5", {"a":1}, typing.Any)

def test_assert_type_typing_callable():
    assert_type("var_callable_correct", lambda x: x, typing.Callable)
    # Test with collections.abc.Callable as well, as assert_type handles it
    from collections.abc import Callable as AbcCallable
    assert_type("var_abc_callable_correct", lambda x: x, AbcCallable)
    with pytest.raises(AssertionError):
        assert_type("var_callable_incorrect", 123, typing.Callable)
    with pytest.raises(AssertionError):
        assert_type("var_abc_callable_incorrect", 123, AbcCallable)

# --- Tests for assert_argument_types decorator with typing library ---

@assert_argument_types
def func_with_typing_collections(a: typing.List[int], b: typing.Dict[str, float]):
    return True

def test_decorator_typing_collections_correct():
    func_with_typing_collections([1, 2], {"pi": 3.14})
    func_with_typing_collections([], {}) # Empty collections are fine

def test_decorator_typing_collections_incorrect():
    with pytest.raises(AssertionError): # Incorrect type for list
        func_with_typing_collections({1, 2}, {"pi": 3.14})
    with pytest.raises(AssertionError): # Incorrect type for dict
        func_with_typing_collections([1, 2], ["pi", 3.14])
    # Note: content type checks (e.g. List[str] for List[int]) are not performed by current assert_type

@assert_argument_types
def func_with_typing_optional_union(a: typing.Optional[str], b: typing.Union[int, float]):
    return True

def test_decorator_typing_optional_union_correct():
    func_with_typing_optional_union("hello", 10)
    func_with_typing_optional_union(None, 20.5)
    func_with_typing_optional_union("world", 30.3)
    func_with_typing_optional_union(None, 40)

def test_decorator_typing_optional_union_incorrect():
    with pytest.raises(AssertionError): # Incorrect type for Optional[str]
        func_with_typing_optional_union(123, 10)
    with pytest.raises(AssertionError): # Incorrect type for Union[int, float]
        func_with_typing_optional_union("hello", "world")
    with pytest.raises(AssertionError): # Incorrect type for Union[int, float]
        func_with_typing_optional_union(None, "None")


@assert_argument_types
def func_with_typing_any(a: typing.Any, b: int):
    return True

def test_decorator_typing_any_correct():
    func_with_typing_any("string", 10)
    func_with_typing_any(123.45, 20)
    func_with_typing_any(None, 30)
    func_with_typing_any([1,2,3], 40)
    func_with_typing_any({"key":"value"}, 50)
    func_with_typing_any(lambda x: x, 60)

def test_decorator_typing_any_incorrect():
    with pytest.raises(AssertionError): # Incorrect type for b (int)
        func_with_typing_any("whatever", "not_an_int")


@assert_argument_types
def func_with_typing_callable(a: typing.Callable[[int], str], b: int):
    # The signature [[int], str] is for documentation; assert_type checks isinstance callable.
    return a(b)

def my_callable_example(x: int) -> str:
    return f"Number: {x}"

def not_my_callable_example(x: str) -> str: # different signature for testing
    return f"String: {x}"


def test_decorator_typing_callable_correct():
    result = func_with_typing_callable(my_callable_example, 10)
    assert result == "Number: 10"
    
    result_lambda = func_with_typing_callable(lambda y: f"Lambda: {y}", 20)
    assert result_lambda == "Lambda: 20"

def test_decorator_typing_callable_incorrect():
    with pytest.raises(AssertionError): # a is not callable
        func_with_typing_callable("not_a_callable", 10)
    
    # This case will pass the assert_type check because it only checks if 'a' is callable,
    # not the signature of the callable. This is consistent with assert_type's current behavior.
    # If detailed signature checking were part of assert_type for Callable, this would fail.
    # For now, we test that a non-callable fails.
    # func_with_typing_callable(not_my_callable_example, 10) # This would likely cause a TypeError inside the function if types are incompatible
                                                          # but not an AssertionError from the decorator for the callable type itself.
