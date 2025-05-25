from threading import Thread

class ThreadWithReturnValue(Thread):
    def __init__(self: "ThreadWithReturnValue", target: callable, kwargs: dict) -> None:
        super().__init__(target=target, kwargs=kwargs)

        self._return = None

    def run(self: "ThreadWithReturnValue") -> None:
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self: "ThreadWithReturnValue", timeout: (None, float, int) = False):
        if timeout:
            Thread.join(self, timeout=timeout)
        else:
            Thread.join(self)
        return self._return
