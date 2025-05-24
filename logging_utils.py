import logging
import os
from datetime import datetime
import typing

class Logger(logging.Logger):
    def __init__(
            self: "Logger",
            name: str = "logging",
            level: int = logging.DEBUG,
            default_level: str = 'info'
    ):
        """
        Get a logger instance for logging messages.

        Args:
            name: First argument of `logging.Logger`
            level: Second argument of `logging.Logger`
            default_level: Indicate the default function used to log the messages.

        Returns:
            Initialize the Logger object.
        """
        super().__init__(name, level)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        formatter = logging.Formatter("\r%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)

        self.default_level:int = default_level
        self.current_time: typing.Optional[datetime.datetime] = None
        self.start_time: typing.Optional[datetime.datetime] = None
        self.iterations: typing.Optional[int] = None
        self.percentage_threshold: float = 0
        self.current_percentage: float = 0
        self.current_iteration: int = 0
        self.indent: int = 0

    def get_log_function(
            self: "Logger",
            level: (str, None) = None
    ) -> callable:
        """
        Retrieve the logging function used to display the messages.

        Args:
            level: Optional. Name of the logging function to use. If None, use the default function.

        Returns:
            The logging function.
        """
        log_function = getattr(self, self.default_level if level is None else level)
        return log_function

    def __call__(
            self: "Logger",
            message: str,
            level: typing.Optional[str] = None,
            indent: int = 0
    ) -> None:
        """
        Display a message.

        Args:
            message: The message to display.
            level: Optional. Name of the logging function to use. If None, use the default function.

        Returns:
            Nothing, but display a message.
        """
        if indent < 0:
            self.indent += indent
        self.get_log_function(level)('\t'*self.indent + message)
        if indent > 0:
            self.indent += indent

    def iterate(
            self: "Logger",
            message: str = '',
            level: (str, None) = None,
            display: bool = True
    ) -> None:
        """
        Display a message and the Estimate Time of Arrival (ETA) of the current loop.

        Args:
            message: The message to display.
            level: Optional. Name of the logging function to use. If None, use the default function.
            display: Wheter we print a message or only update the iterator.

        Returns:
            Nothing, but display a message with an Estimated Time of Arrival.
        """
        self.current_time = datetime.now()

        remaining_time = (self.current_time - self.start_time) / max(1, self.current_iteration) * self.iterations
        termination_time = self.start_time + remaining_time
        termination_time = termination_time.strftime("%Y-%m-%d %H:%M")

        current_percentage = self.current_iteration / self.iterations * 100

        if display and current_percentage - self.current_percentage > self.percentage_threshold:
            self.current_percentage = current_percentage
            message = f"ETA: {termination_time} ({self.current_iteration}/{self.iterations}) - " + message
            self.get_log_function(level)(message)

        self.current_iteration += 1

    def set_iterator(
            self: "Logger",
            iterations: int,
            percentage_threshold: (int, float) = 0
    ) -> None:
        """
        Set the parameters of the internal iterator to be able to display an ETA when looping.

        Args:
            iterations: The length of the iterator to browse.
            percentage_threshold: The logger will only display message if the process advanced by the given percentage.

        Returns:
            Nothing, but set `start_time`, `current_iteration` and `iterations` to compute the ETA.
        """
        self.start_time = datetime.now()
        self.current_iteration = 0
        self.current_percentage = 0
        self.iterations = iterations
        self.percentage_threshold = percentage_threshold

    def positional_logger_decorator(
            self: "Logger",
            filename: (str, None) = None,
            level: (str, None) = None
    ) -> callable:
        """
        This decorator aims at printing the file containing the function on which it is used.
        It also prints the start and termination times.

        Args:
            filename: Should be the name of the file containing the decorated function.
            level: String like "info", "debug", to choose the level of the messages.

        Returns:
            A decorator to display the name of either the function or a string.
        """

        def logger_decorator(function: callable) -> callable:
            """
            A decorates can't have any other argument than the function it decorates.
            Therefore, we had to use a parent function to store the name of the function in its namespace.

            Args:
                function: The decorated function.
            """
            printed_name = filename if filename is not None else function.__name__

            def logged_function(*args, **kwargs):
                """
                The decorated function. It just prints start and termination time and forward both kwargs and kwargs.
                """
                self.get_log_function(level)(f"'{printed_name}': Start")
                result = function(*args, **kwargs)
                self.get_log_function(level)(f"'{printed_name}': End\n")
                return result

            return logged_function

        if filename is not None:
            filename = os.path.basename(filename)
        return logger_decorator


logger = Logger()
