"""General utility functions."""

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer(message: str) -> Generator[dict[str, float], None, None]:
    """A context manager that prints and logs the time taken to execute the
    block of code."""
    result: dict[str, float] = {}
    start_time = time.perf_counter()
    yield result
    end_time = time.perf_counter()
    result["time"] = end_time - start_time
    print_and_log(f"{message}: {result['time']:.6f} seconds")


def print_and_log(
    message: str,
    log_file: str = "experiments/log.txt",
) -> None:
    """Print and log a message to a file."""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
