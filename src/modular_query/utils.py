"""General utility functions."""

import time
from contextlib import contextmanager
from typing import Generator

from modular_query.modules import Module


@contextmanager
def timer(
    message: str, verbose: bool = True
) -> Generator[dict[str, float], None, None]:
    """A context manager that prints and logs the time taken to execute the
    block of code."""
    result: dict[str, float] = {}
    start_time = time.perf_counter()
    yield result
    end_time = time.perf_counter()
    result["time"] = end_time - start_time
    if verbose:
        print_and_log(f"{message}: {result['time']:.6f} seconds")


def print_and_log(
    message: str,
    log_file: str = "experiments/log.txt",
) -> None:
    """Print and log a message to a file."""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")


def product_of_confidences(confidences: dict[Module, float]) -> float:
    """Compute the product of confidences."""
    product = 1.0
    for conf in confidences.values():
        product *= conf
    return product


def sum_of_uncertainties(confidences: dict[Module, float]) -> float:
    """Compute the sum of uncertainties."""
    # Uncertainty is 1 - confidence.
    uncertainty_sum = 0.0
    for conf in confidences.values():
        uncertainty_sum += 1.0 - conf
    return uncertainty_sum
