import functools
import os

from rignak.logging_utils import logger


class ExistingFilename(str):
    def __new__(cls, string, message=None):
        if message is None:
            message = f"`{string}` not found."
        assert os.path.exists(string), message
        instance = super().__new__(cls, string)
        return instance


def assert_type(
        variable_name: str,
        variable,
        type_: (type, tuple)
) -> None:
    """
    Assert that the variable is of a required type.

    Args:
        variable_name:  The name to display in the error message.
        variable: The variable whose type we check.
        type_: The required type (or tuple of types). Special cases are `callable`, `None` and `ExistingFilename`.

    Returns:
        Nothing.
    """
    # Switch `None` to `Nonetype`
    if type_ is None:
        type_ = type(None)
    elif isinstance(type_, tuple):
        type_ = tuple([type(None) if x is None else x for x in type_])

    if type_ is callable:  # Check if function
        assert_callable(variable_name, variable)
    else:
        assert isinstance(variable, type_), f"'{variable_name}' should be '{type_}' but is '{type(variable)}."


def assert_isin(
        value,
        iterable
) -> None:
    """
    Assert that the iterable contains a given value.

    Args:
         value: The value that is required to be in the iterable.
         iterable: The iterable that should contain the value.

    Returns:
        Nothing.
    """
    assert value in iterable, f"'{value}' not found in {repr(iterable)}."


def assert_callable(
        variable_name: str,
        variable
) -> None:
    """
    Assert that the variable can be called.

    Args:
        variable_name: The name to display in the error message.
        variable: The variable we want to be a callable.

    Returns:
        Nothing.
    """
    assert callable(variable), f"{variable_name} should be a callable function."


def assert_argument_types(function):
    """
    Ensure a hard-typing behaviour.
    At each call of the decorated function, the type of the arguments will be checked against those of the type hints.
    Discrepancies betwen the call and the type hint will immediatly raise an error.
    """

    @functools.wraps(function)
    def asserted_function(*args, **kwargs):
        try:
            to_check = []
            for argument, (argument_name, required_type) in zip(args, function.__annotations__.items()):
                if argument_name == 'self' and isinstance(required_type, str):
                    continue  # it's just the `self` argument of a class method.
                to_check.append((argument_name, argument, required_type))
                assert_type(argument_name, argument, required_type)

            keys = list(function.__annotations__.keys())
            for argument_name, argument in kwargs.items():
                assert_isin(argument_name, keys)
                to_check.append((argument_name, argument, function.__annotations__[argument_name]))

            for argument_name, argument, required_type in to_check:
                assert_type(argument_name, argument, required_type)
            result = function(*args, **kwargs)
        except AssertionError as error:
            logger.error(error)
            raise
        return result

    return asserted_function
