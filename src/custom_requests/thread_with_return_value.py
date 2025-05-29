from threading import Thread
from typing import Optional, Callable, Any, TypeVar, Tuple, Dict, Union

_R = TypeVar('_R')

class ThreadWithReturnValue(Thread):
    def __init__(
            self: 'ThreadWithReturnValue', 
            target: Callable[..., _R], 
            kwargs: Dict[str, Any], 
            # To match Thread's run behavior which uses self._args, we should accept args too.
            # For this exercise, sticking to the original parameters but ensuring super call is robust.
            args: Tuple[Any, ...] = () # Adding args to match superclass and self._args usage in run
    ) -> None:
        super().__init__(target=target, args=args, kwargs=kwargs)
        self._return: Optional[_R] = None

    def run(self: 'ThreadWithReturnValue') -> None:
        # self._target, self._args, self._kwargs are set by Thread's __init__
        if self._target: 
            try:
                self._return = self._target(*self._args, **self._kwargs)
            except Exception as e:
                # Optionally handle exceptions from the target, e.g., log them
                # For now, the thread will terminate and _return might remain None or an exception could be stored.
                # Depending on requirements, self._return could store the exception instance.
                # Raising e here would make the thread crash, which is default Thread behavior.
                # print(f"Exception in thread {self.name}: {e}") # Example logging
                pass # Allow _return to remain None or be partially set if target has side effects before exception


    def join(self: 'ThreadWithReturnValue', timeout: Optional[float] = None) -> Optional[_R]:
        # The original 'timeout: (None, float, int) = False' is problematic.
        # 'False' is not a valid timeout value (should be float or None).
        # Assuming the intent was Optional float.
        super().join(timeout=timeout) 
        return self._return
