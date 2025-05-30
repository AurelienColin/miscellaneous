import functools
import os
from typing import get_origin, get_args, Any, Union, Tuple as TypingTuple, Type, Optional, List, Callable # Added Callable
from collections.abc import Callable as AbcCallable

from .logging_utils import logger # Assuming logger is typed in its module


class ExistingFilename(str):
    def __new__(cls: Type['ExistingFilename'], string: str, message: Optional[str] = None) -> 'ExistingFilename':
        if message is None:
            message = f"`{string}` not found."
        assert os.path.exists(string), message
        instance = super().__new__(cls, string)
        return instance

# --- Private Helper Functions for assert_type ---

def _handle_any(variable_name: str, variable: Any, type_hint: Type[Any]) -> None:
    """Handles typing.Any. Always passes."""
    pass

def _handle_none_type(variable_name: str, variable: Any) -> None:
    """Asserts that the variable is None."""
    if variable is not None:
        # Using type(None).__name__ for consistency, though "None" is also fine.
        raise AssertionError(
            f"'{variable_name}' should be '{type(None).__name__}' but is '{type(variable).__name__}'."
        )

def _handle_union_type(variable_name: str, variable: Any, type_hint_repr: Any, union_args: TypingTuple[Any, ...]) -> None:
    """Handles typing.Union types, including Optional."""
    processed_union_args = []
    for arg in union_args:
        if arg is Any: 
            return 
        processed_union_args.append(type(None) if arg is None else arg)
    
    match_found_in_union = False
    for t_arg in processed_union_args:
        try:
            assert_type(variable_name, variable, t_arg) 
            match_found_in_union = True
            break
        except AssertionError:
            continue
    
    if not match_found_in_union:
        type_reprs = [arg.__name__ if hasattr(arg, '__name__') else str(arg) for arg in processed_union_args]
        expected_types_str = ", ".join(type_reprs)
        # Use type_hint_repr for the "Expected one of: {type_hint_repr}" part for better user message
        raise AssertionError(
            f"'{variable_name}' of type '{type(variable).__name__}' does not match any type in Union[{expected_types_str}]."
            f" Expected one of: {type_hint_repr}."
        )

def _handle_abc_callable_type(variable_name: str, variable: Any, type_hint: Type[AbcCallable]) -> None:
    """Handles typing.Callable and collections.abc.Callable."""
    if not isinstance(variable, AbcCallable):
        type_hint_repr = type_hint.__name__ if hasattr(type_hint, '__name__') else str(type_hint)
        raise AssertionError(
            f"'{variable_name}' should be callable (expected '{type_hint_repr}'), but is '{type(variable).__name__}'."
        )

def _handle_generic_origin(variable_name: str, variable: Any, type_hint: Any, origin_type: Type[Any]) -> None:
    """Handles generic types from typing (List, Dict, etc.) by checking their origin."""
    if not isinstance(variable, origin_type):
        origin_name = origin_type.__name__ if hasattr(origin_type, '__name__') else str(origin_type)
        type_hint_repr = str(type_hint) # Full representation like List[int]
        raise AssertionError(
            f"'{variable_name}' should be an instance of '{origin_name}' (for typing hint '{type_hint_repr}') "
            f"but is '{type(variable).__name__}'."
        )

def _handle_tuple_of_types(variable_name: str, variable: Any, type_tuple: TypingTuple[Type[Any], ...]) -> None:
    """Handles tuples of types, e.g., assert_type(var, (int, str))."""
    # This function is for when type_ itself is a tuple of types, e.g. assert_type(var, (int, str))
    # It is NOT for typing.Tuple[int, str] which is handled by _handle_generic_origin (origin is tuple)
    
    if Any in type_tuple:
        return

    # Replace None with type(None) in the tuple for isinstance compatibility if we were to use it directly
    # However, recursive calls to assert_type are more robust.
    # processed_tuple_elements = [type(None) if t_element is None else t_element for t_element in type_tuple]

    match_found_in_tuple = False
    # We iterate through the original `type_` tuple to preserve complex typing hints for recursion
    for t_element in type_tuple: 
        try:
            assert_type(variable_name, variable, t_element) 
            match_found_in_tuple = True
            break 
        except AssertionError:
            continue
    
    if not match_found_in_tuple:
        type_reprs = [te.__name__ if hasattr(te, '__name__') else str(te) for te in type_tuple]
        expected_types_str = ", ".join(type_reprs)
        raise AssertionError(
            f"'{variable_name}' of type '{type(variable).__name__}' does not match any type in the tuple ({expected_types_str})."
            f" Expected one of: {type_tuple}."
        )

def _handle_builtin_callable_keyword(variable_name: str, variable: Any) -> None:
    """Handles the built-in `callable` keyword used as a type hint."""
    if not isinstance(variable, AbcCallable):
         raise AssertionError(
            f"'{variable_name}' should be callable (expected keyword 'callable'), but is '{type(variable).__name__}'."
        )

def _handle_standard_type(variable_name: str, variable: Any, type_hint: Type[Any]) -> None:
    """Handles standard Python types, custom classes, and ExistingFilename."""
    if not isinstance(variable, type_hint):
        expected_type_name = type_hint.__name__ if hasattr(type_hint, '__name__') else str(type_hint)
        actual_type_name = type(variable).__name__ if hasattr(type(variable), '__name__') else str(type(variable))
        raise AssertionError(
            f"'{variable_name}' should be '{expected_type_name}' but is '{actual_type_name}'."
        )

# --- Main assert_type function (Dispatcher) ---
def assert_type(
        variable_name: str,
        variable: Any,
        type_: Any 
) -> None:
    """
    Assert that the variable is of a required type. Dispatches to helper functions.
    Args:
        variable_name:  The name to display in the error message.
        variable: The variable whose type we check.
        type_: The required type (or tuple of types).
    Returns:
        Nothing.
    """
    if type_ is Any:
        _handle_any(variable_name, variable, type_)
        return

    if type_ is None or type_ is type(None): # Handle both None and type(None) as hints for NoneType
        _handle_none_type(variable_name, variable)
        return

    # Must check for tuple instance before get_origin for types like (int, str)
    if isinstance(type_, tuple): 
        _handle_tuple_of_types(variable_name, variable, type_)
        return

    origin_type = get_origin(type_)
    type_args = get_args(type_) # Safe to call even if no args, returns ()

    if origin_type is Union:
        _handle_union_type(variable_name, variable, type_, type_args)
        return
    
    # Check for typing.Callable or collections.abc.Callable
    # type_ is AbcCallable handles `collections.abc.Callable` directly
    # origin_type is AbcCallable handles `typing.Callable[...]`
    if type_ is AbcCallable or origin_type is AbcCallable:
        _handle_abc_callable_type(variable_name, variable, type_)
        return

    if origin_type is not None: 
        _handle_generic_origin(variable_name, variable, type_, origin_type)
        return
    
    if type_ is callable: 
        _handle_builtin_callable_keyword(variable_name, variable)
        return
    
    # Also handles cases where type_ is a Type[Any] directly.
    _handle_standard_type(variable_name, variable, type_)


def assert_isin(
        value: Any, 
        iterable: Any 
) -> None:
    assert value in iterable, f"'{value}' not found in {repr(iterable)}."


def assert_callable( 
        variable_name: str,
        variable: Any
) -> None:
    # This function seems redundant if assert_type handles `callable` keyword.
    # However, it might be used directly elsewhere. Keeping it.
    if not isinstance(variable, AbcCallable): # Check against AbcCallable for consistency
        raise AssertionError(f"{variable_name} should be a callable function, but is '{type(variable).__name__}'.")


def assert_argument_types(function: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(function)
    def asserted_function(*args: Any, **kwargs: Any) -> Any:
        try:
            to_check: List[Tuple[str, Any, Any]] = [] # (name, value, required_type)
            
            # Get argument names from function code object
            arg_names = function.__code__.co_varnames[:function.__code__.co_argcount]

            # function.__annotations__ includes return type, so filter if needed or slice args.
            # zip will stop at the shorter of args or annotations relevant to args.
            
            for i, arg_value in enumerate(args):
                if i < len(arg_names): # Should always be true unless *args is empty and arg_names not
                    arg_name = arg_names[i]
                    # Skip 'self' if it's a method and type hint is a string (forward ref)
                    if arg_name == 'self' and isinstance(function.__annotations__.get(arg_name), str):
                        continue
                    if arg_name in function.__annotations__:
                        required_type = function.__annotations__[arg_name]
                        to_check.append((arg_name, arg_value, required_type))
            
            for kw_name, kw_value in kwargs.items():
                if kw_name in function.__annotations__:
                    required_type = function.__annotations__[kw_name]
                    to_check.append((kw_name, kw_value, required_type))
                # Note: Original `assert_isin(kw_name, list(function.__annotations__.keys()))` was removed
                # as it would prevent passing extra kwargs to functions, which is standard Python behavior.
                # If strict checking of only annotated kwargs is desired, it could be re-added.

            for arg_name_to_check, arg_val_to_check, req_type_to_check in to_check:
                assert_type(arg_name_to_check, arg_val_to_check, req_type_to_check)
            
            result: Any = function(*args, **kwargs)
        except AssertionError as error: 
            logger.error(str(error)) 
            raise
        return result
    return asserted_function
