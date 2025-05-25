# Rignak Python Utilities

Rignak is a collection of Python utility modules designed to simplify common programming tasks and provide reusable components.

## Features

The library is organized into several modules, each providing distinct functionalities:

*   **`assert_utils`**: Offers tools for runtime assertions and type checking, including a decorator (`@assert_argument_types`) to enforce type hints for function arguments and a special `ExistingFilename` type that verifies file existence.
*   **`backup`**: Provides a utility function (`backup_py_files`) to create zip archives of Python and text files within a specified repository path.
*   **`custom_display`**: A comprehensive module for generating various types of plots and visualizations using `matplotlib` and `seaborn`. It supports line plots, scatter plots, heatmaps, histograms, geographical maps (via Basemap integration), and offers extensive customization through a `Kwargs` dataclass and a `Display` helper class.
*   **`custom_requests`**: A package for handling HTTP requests with advanced features:
    *   `local_config`: Configuration for request headers, cookies, retry limits, and multithreading.
    *   `multithread`: A `routing` function to execute a function with multiple argument sets in parallel threads.
    *   `request_utils`: Utilities for robust HTTP requests with retries, streaming, file downloads (including image conversion to JPG), and URL/filename conversions.
    *   `thread_with_return_value`: A custom `Thread` subclass enabling return values from threaded functions.
    *   `tor`: Functions for interacting with the Tor network, such as renewing Tor circuits (requires the `stem` library).
*   **`lazy_property`**: Contains a `LazyProperty` decorator that computes a property's value upon its first access and then caches it for subsequent calls.
*   **`logging_utils`**: Implements a custom `Logger` class with features like ETA calculation for loops, message indentation, and a decorator for logging function entry/exit points.
*   **`path`**: Includes utility functions for common path manipulations, such as resolving local paths, listing directory contents with extension filters, and retrieving parent directory paths.
*   **`textfile_utils`**: Provides functions for safely replacing file contents (`safe_file_replacement`) and reading lines from text files (`get_lines`).

## Installation

To install the Rignak library, you can use pip with the `setup.py` file:

```bash
pip install .
```

For Tor functionalities, install the 'tor' extra:

```bash
pip install .[tor]
```

For plotting functionalities, install the 'display' extra:
```bash
pip install .[display]
```

## Coding Style

This library aims to follow these coding principles:
*   Functions should be concise and modular.
*   Code should be self-documenting, minimizing the need for comments.
*   Function prototypes must include type hints for inputs and outputs.
*   Dataclasses and `LazyProperty` are preferred for data structures and computed properties, respectively.

```
