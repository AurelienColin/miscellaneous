import threading
import time
from typing import List, Callable, Any, Tuple, Dict, Union, Hashable, FrozenSet, TypeVar, Iterable, Optional

from . import config # Assuming config attributes are appropriately typed or Any
from .thread_with_return_value import ThreadWithReturnValue
# Assuming logger is an instance of the Logger class defined in logging_utils
try:
    from ..logging_utils import logger # Try relative import
except ImportError:
    # Fallback or placeholder if direct relative import fails (e.g. running script directly)
    # This section might need adjustment based on actual project structure and how logger is provided.
    print("Warning: Could not import logger via '..logging_utils'. Using a placeholder.")
    class PlaceholderLogger:
        def set_iterator(self, *args: Any, **kwargs: Any) -> None: pass
        def iterate(self, *args: Any, **kwargs: Any) -> None: pass
    logger = PlaceholderLogger() # type: ignore


_K = TypeVar('_K', bound=Hashable)
_V = TypeVar('_V')
_R = TypeVar('_R') 

# Define a more specific return type for freeze if possible.
# Since it's recursive and can handle nested structures, using 'Any' for elements is pragmatic.
# This FrozenType describes the possible output structures of freeze.
FrozenType = Union[Tuple[Any, ...], FrozenSet[Tuple[Any, Any]], Hashable]

def freeze(mutable: Union[Dict[_K, _V], List[_V], _V]) -> FrozenType:
    """
    Recursively freezes mutable dicts and lists into hashable frozensets and tuples.
    Other types are returned as is (assuming they are already hashable or base types).
    """
    if isinstance(mutable, dict):
        # Keys are already Hashable (_K). Values are recursively frozen.
        return frozenset((key, freeze(value)) for key, value in mutable.items())
    elif isinstance(mutable, list):
        return tuple(freeze(value) for value in mutable)
    # Assuming other types passed are already hashable (e.g., int, str, float, or already frozen)
    return mutable # type: ignore 


def routing(
        function: Callable[..., _R],
        kwargs_list: List[Dict[str, Any]],
        thread_limit: int = config.PARALLEL_THREADS_LIMIT,
        seconds_between_threads: Union[float, int] = config.SECONDS_BETWEEN_THREADS,
        thread_timeout: Union[float, int] = config.THREAD_TIMEOUT,
        seconds_after_completion: Union[float, int] = config.SECONDS_AFTER_COMPLETION
) -> Dict[FrozenSet[Tuple[str, Any]], Optional[_R]]:
    
    # This behavior means the actual number of concurrent threads from this routing call
    # plus existing threads should not exceed thread_limit + current_threads.
    # It might be simpler to interpret thread_limit as the max for *this batch* of tasks.
    # For now, sticking to the original logic:
    effective_max_threads: int = thread_limit + threading.active_count() 
    
    threads_info: List[Tuple[Dict[str, Any], ThreadWithReturnValue[_R]]] = []

    # Assuming logger has set_iterator and iterate methods.
    # If logger type is unknown, these calls might be flagged by a strict type checker.
    logger.set_iterator(len(kwargs_list), percentage_threshold=1) # type: ignore

    for kwarg_item in kwargs_list:
        logger.iterate() # type: ignore
        
        while threading.active_count() >= effective_max_threads: # Use >= for clarity
            time.sleep(0.01)
            
        # Create and start the thread. Pass the individual kwarg_item to the target function.
        # ThreadWithReturnValue expects 'target' and 'kwargs' (and optionally 'args').
        thread = ThreadWithReturnValue(target=function, kwargs=kwarg_item, args=()) # Pass empty args
        thread.start()
        threads_info.append((kwarg_item, thread))

        time.sleep(float(seconds_between_threads))
    
    time.sleep(float(seconds_after_completion))

    results: Dict[FrozenSet[Tuple[str, Any]], Optional[_R]] = {}
    for original_kwargs, thread_instance in threads_info:
        # Freeze the original kwargs to use as dict key.
        # The type of key for 'results' dict is FrozenSet[Tuple[str, Any]]
        # freeze(original_kwargs) should produce something compatible.
        # original_kwargs is Dict[str, Any]. freeze will turn it into FrozenSet[Tuple[str, FrozenType]]
        frozen_key_intermediate = freeze(original_kwargs)
        
        # Ensure the frozen_key_intermediate is of the correct type for the dictionary key.
        # If freeze returns a single Hashable for a non-dict/list input, this needs adjustment.
        # Assuming original_kwargs is always a dict here.
        if not isinstance(frozen_key_intermediate, frozenset):
            # This case should ideally not happen if original_kwargs are dicts.
            # Handle or raise error if the key structure is not as expected.
            # For now, converting to a frozenset of a tuple to fit Dict key type.
            # This part is tricky and depends on the exact structure of what freeze returns for dicts.
            # Assuming freeze(Dict) returns FrozenSet[Tuple[Hashable, FrozenType]]
            frozen_key = frozenset((("unexpected_frozen_key_structure", frozen_key_intermediate),))
        else:
            frozen_key = frozen_key_intermediate
            
        # Join the thread and get its return value (which is Optional[_R])
        result_value: Optional[_R] = thread_instance.join(timeout=float(thread_timeout))
        results[frozen_key] = result_value # type: ignore
        
    return results
