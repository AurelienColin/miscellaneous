from typing import Callable, Any, Optional, TypeVar, Generic

_T = TypeVar('_T')
_R = TypeVar('_R')

class LazyProperty(Generic[_T, _R]):
    """
    The LazyProperty is used to be less verbose. Eg., the following code:
    It implements lazy computation (i.e. computing variables only when needed)
    and run a sanity check (non-None) to ensure when do get stuck in a loop.


    @dataclass
    class Foo:
        _bar: typing.Optional[int] = None # Note: Original example used typing.Optional

        def bar(self) -> int:
            if self._bar is None:
                self._bar = ... # type: ignore
            assert self._bar is not None
            return self._bar

    is replaced by:

    @dataclass
    class Foo:
        _bar: typing.Optional[int] = None # Note: Original example used typing.Optional

        @LazyProperty
        def bar(self) -> int:
            return ... # type: ignore
    """

    def __init__(self, func: Callable[[_T], _R]) -> None:
        self.func: Callable[[_T], _R] = func
        self.attr_name: str = f"_{func.__name__}"

    def __get__(self, instance: Optional[_T], owner: Optional[type] = None) -> _R: # owner can also be Type[_T]
        if instance is None:
            # When accessed on the class, return the descriptor itself
            # Type check would expect _R, but returning self (LazyProperty) is standard descriptor protocol.
            # This is a known tricky point in typing descriptors.
            # For practical purposes, if this path is taken, the type checker might complain
            # if the context expects _R. However, typical usage is via instance.
            return self # type: ignore 
        
        # This will raise AttributeError if self.attr_name (e.g., "_another_lazy_prop")
        # was not defined in instance.__init__ (as in SampleClassNoInit from test).
        # This matches the expectation of test_lazy_property_without_explicit_init_in_constructor.
        current_value: Optional[_R] = getattr(instance, self.attr_name) 
        
        if current_value is None: # It existed on the instance, but was initialized to None (signifying not yet computed)
            computed_value: _R = self.func(instance)
            setattr(instance, self.attr_name, computed_value)
            
            # The assertion ensures that the computation itself does not return None.
            # If self.func can legitimately return None and that should be cached, this assert needs adjustment.
            # Given the original assert, it seems None results from self.func were not expected to be cached as valid.
            assert computed_value is not None, \
                f"Lazy computation for '{self.attr_name}' returned None, which is not allowed by this LazyProperty."
            return computed_value
        else: # It existed and was not None (already computed and cached)
            # current_value here is known to be _R because it's not None (initial Optional[_R] was for the None case)
            return current_value # type: ignore
