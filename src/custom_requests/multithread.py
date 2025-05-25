import threading
import time

from .custom_requests import config as config
from .custom_requests.thread_with_return_value import ThreadWithReturnValue


def freeze(mutable: (list, dict)) -> (tuple, frozenset):
    if isinstance(mutable, dict):
        return frozenset((key, freeze(value)) for key, value in mutable.items())
    elif isinstance(mutable, list):
        return tuple(freeze(value) for value in mutable)
    return mutable


def routing(
        function: callable,
        kwargs: list,
        thread_limit: int = config.PARALLEL_THREADS_LIMIT,
        seconds_between_threads: (float, int) = config.SECONDS_BETWEEN_THREADS,
        thread_timeout: (float, int) = config.THREAD_TIMEOUT,
        seconds_after_completion: (float, int) = config.SECONDS_AFTER_COMPLETION
) -> dict:
    thread_limit += threading.active_count()
    threads = []

    for kwarg in kwargs:
        while threading.active_count() > thread_limit:
            time.sleep(0.01)
        thread = ThreadWithReturnValue(target=function, kwargs=kwarg)
        thread.start()
        threads.append((kwarg, thread))

        time.sleep(seconds_between_threads)
    time.sleep(seconds_after_completion)

    results = {freeze(kwargs): thread.join(timeout=thread_timeout) for kwargs, thread in threads}
    return results
