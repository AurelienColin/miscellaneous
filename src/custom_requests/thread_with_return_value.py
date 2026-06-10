from threading import Thread
from typing import Optional, Callable, Any, TypeVar, Tuple, Dict, Union
from rignak.logging_utils import logger

_R = TypeVar('_R')

class ThreadWithReturnValue(Thread):
    def __init__(
            self: 'ThreadWithReturnValue', 
            target: Callable[..., _R], 
            kwargs: Dict[str, Any], 
            args: Tuple[Any, ...] = () #
    ) -> None:
        super().__init__(target=target, args=args, kwargs=kwargs)
        self._return: Optional[_R] = None

    def run(self: 'ThreadWithReturnValue') -> None:
        if self._target:
            try:
                self._return = self._target(*self._args, **self._kwargs)
            except Exception as e:
                logger(f"Exception in thread {self.name}: {e}")


    def join(self: 'ThreadWithReturnValue', timeout: Optional[float] = None) -> Optional[_R]:
        super().join(timeout=timeout)
        return self._return
