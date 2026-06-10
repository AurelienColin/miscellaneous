import threading
import time
import typing
from . import config
from .thread_with_return_value import ThreadWithReturnValue

# Assuming logger is an instance of the Logger class defined in logging_utils
try:
    from ..logging_utils import logger  # Try relative import
except ImportError:
    print("Warning: Could not import logger via '..logging_utils'. Using a placeholder.")


    class PlaceholderLogger:
        def set_iterator(self, *args: typing.Any, **kwargs: typing.Any) -> None: pass

        def iterate(self, *args: typing.Any, **kwargs: typing.Any) -> None: pass


    logger = PlaceholderLogger()  # type: ignore

_K = typing.TypeVar('_K', bound=typing.Hashable)
_V = typing.TypeVar('_V')
_R = typing.TypeVar('_R')

FrozenType = typing.Union[
    typing.Tuple[typing.Any, ...],
    typing.FrozenSet[typing.Tuple[typing.Any, typing.Any]], typing.Hashable
]


def freeze(mutable: typing.Union[typing.Dict[_K, _V], typing.List[_V], _V]) -> FrozenType:
    """
    Recursively freezes mutable dicts and lists into hashable frozensets and tuples.
    Other types are returned as is (assuming they are already hashable or base types).
    """
    if isinstance(mutable, dict):
        # Keys are already Hashable (_K). Values are recursively frozen.
        return frozenset((key, freeze(value)) for key, value in mutable.items())
    elif isinstance(mutable, list):
        return tuple(freeze(value) for value in mutable)
    return mutable


def routing(
        function: typing.Callable[..., _R],
        kwargs_list: typing.List[typing.Dict[str, typing.Any]],
        thread_limit: int = config.PARALLEL_THREADS_LIMIT,
        seconds_between_threads: typing.Union[float, int] = config.SECONDS_BETWEEN_THREADS,
        thread_timeout: typing.Union[float, int] = config.THREAD_TIMEOUT,
        seconds_after_completion: typing.Union[float, int] = config.SECONDS_AFTER_COMPLETION
) -> typing.Dict[typing.FrozenSet[typing.Tuple[str, typing.Any]], typing.Optional[_R]]:
    effective_max_threads: int = thread_limit + threading.active_count()

    threads: typing.List[typing.Tuple[typing.Dict[str, typing.Any], ThreadWithReturnValue[_R]]] = []
    logger.set_iterator(len(kwargs_list), percentage_threshold=1)

    for kwarg_item in kwargs_list:
        logger.iterate()  # type: ignore

        while threading.active_count() >= effective_max_threads:
            time.sleep(0.01)

        thread = ThreadWithReturnValue(target=function, kwargs=kwarg_item)
        thread.start()
        threads.append((kwarg_item, thread))

        time.sleep(float(seconds_between_threads))

    time.sleep(float(seconds_after_completion))

    results = {freeze(kwargs): thread.join(timeout=thread_timeout) for kwargs, thread in threads}

    return results
