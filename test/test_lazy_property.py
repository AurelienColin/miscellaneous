import pytest
from src.lazy_property import LazyProperty

class SampleClass:
    def __init__(self):
        self._counter = 0
        self._expensive_computation = None # Explicitly initialize to None

    @LazyProperty
    def expensive_computation(self) -> int:
        self._counter += 1
        # Simulate an expensive computation
        self._expensive_computation = 42
        return self._expensive_computation

def test_lazy_property_computes_once():
    instance = SampleClass()
    assert instance._counter == 0
    
    # Access for the first time
    val1 = instance.expensive_computation
    assert val1 == 42
    assert instance._counter == 1
    assert instance._expensive_computation == 42

    # Access for the second time
    val2 = instance.expensive_computation
    assert val2 == 42
    assert instance._counter == 1 # Counter should not increment again
    assert instance._expensive_computation == 42 # Value should be cached
    
class SampleClassNoInit:
    def __init__(self):
        self._counter = 0
        # _another_lazy_prop is not initialized here

    @LazyProperty
    def another_lazy_prop(self) -> str:
        self._counter +=1
        return "computed"

def test_lazy_property_without_explicit_init_in_constructor():
    instance = SampleClassNoInit()
    with pytest.raises(AttributeError):
        _ = instance.another_lazy_prop