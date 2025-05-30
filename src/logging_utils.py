import logging
import os
from datetime import datetime, timedelta
from typing import Optional, Callable, Any, Union, TypeVar 
from dataclasses import dataclass, field 
from functools import wraps 

@dataclass(eq=False) # eq=False because logging.Logger has its own comparison logic
class Logger(logging.Logger):
    name: str = "logging"
    level: int = logging.DEBUG # e.g., logging.INFO, logging.DEBUG
    default_level: str = 'info' # Name of the default logging method (e.g., 'info', 'debug')

    current_time: Optional[datetime] = None
    start_time: Optional[datetime] = None
    iterations: Optional[int] = None
    percentage_threshold: float = 0.0
    # Initializing current_percentage to -1.0 (as in set_iterator) ensures the first message prints
    # if percentage_threshold is 0.0, because (0 - (-1.0)) >= 0.0 is true.
    current_percentage: float = field(default=-1.0, repr=False) # repr=False as it's internal state
    current_iteration: int = 0
    indent: int = 0
    
    def __post_init__(self: 'Logger') -> None:
        """
        Post-initialization for the dataclass.
        This is where we call super().__init__ and set up the handler.
        """
        super().__init__(self.name, self.level)

        console_handler: logging.StreamHandler = logging.StreamHandler()
        console_handler.setLevel(self.level) 
        formatter: logging.Formatter = logging.Formatter("\r%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)
        
    def get_log_function(
            self: 'Logger', 
            level: Optional[str] = None
    ) -> Callable[[str], None]:
        log_function_name = self.default_level if level is None else level
        log_function: Callable[[str], None] = getattr(self, log_function_name)
        return log_function

    def __call__(
            self: 'Logger',
            message: str,
            level: Optional[str] = None,
            indent: int = 0
    ) -> None:
        if indent < 0:
            self.indent += indent
        
        effective_message = ('\t' * self.indent) + message
        self.get_log_function(level)(effective_message)
        
        if indent > 0:
            self.indent += indent

    def iterate(
            self: 'Logger',
            message: str = '',
            level: Optional[str] = None,
            display: bool = True
    ) -> None:
        if self.start_time is None or self.iterations is None or self.iterations == 0:
            if display:
                self.get_log_function(level)(message if message else "Iterator not properly set or zero iterations.")
            # Increment iteration count only if configured and not a zero-iteration loop
            if self.iterations is not None and self.iterations > 0:
                 self.current_iteration += 1
            return

        self.current_time = datetime.now()
        termination_time_str: str = "N/A"

        # current_iteration must be positive for meaningful ETA
        if self.current_time and self.start_time and self.current_iteration > 0:
            time_elapsed: timedelta = self.current_time - self.start_time
            avg_time_per_iteration: timedelta = time_elapsed / self.current_iteration
            remaining_iterations: int = self.iterations - self.current_iteration
            
            if remaining_iterations >= 0:
                total_estimated_duration: timedelta = avg_time_per_iteration * self.iterations
                termination_time_dt: datetime = self.start_time + total_estimated_duration
                termination_time_str = termination_time_dt.strftime("%Y-%m-%d %H:%M")
            else: 
                termination_time_str = "Overrun"
        
        current_percentage_val: float = (self.current_iteration / self.iterations) * 100.0

        # Display if threshold met or it's the very first data point (current_iteration will be 1 after increment)
        if display and (current_percentage_val - self.current_percentage >= self.percentage_threshold or self.current_iteration == 1):
            self.current_percentage = current_percentage_val
            display_percentage = min(current_percentage_val, 100.0)
            display_iteration = min(self.current_iteration, self.iterations)
            
            display_message: str = f"ETA: {termination_time_str} ({display_iteration}/{self.iterations} - {display_percentage:.2f}%) - {message}"
            self.get_log_function(level)(display_message)

        self.current_iteration += 1


    def set_iterator(
            self: 'Logger',
            iterations: int,
            percentage_threshold: Union[int, float] = 0.0
    ) -> None:
        self.start_time = datetime.now()
        self.current_iteration = 0 
        self.current_percentage = -1.0 # Ensure first message prints if threshold is 0
        self.iterations = iterations
        self.percentage_threshold = float(percentage_threshold)


    def positional_logger_decorator(
            self: 'Logger',
            filename: Optional[str] = None,
            level: Optional[str] = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def logger_decorator(function: Callable[..., Any]) -> Callable[..., Any]:
            printed_name: str = os.path.basename(filename) if filename is not None else function.__name__
            @wraps(function)
            def logged_function(*args: Any, **kwargs: Any) -> Any:
                self.get_log_function(level)(f"'{printed_name}': Start")
                result: Any = function(*args, **kwargs)
                self.get_log_function(level)(f"'{printed_name}': End\n")
                return result
            return logged_function
        return logger_decorator

logger: Logger = Logger()
