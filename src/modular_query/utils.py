"""General utility functions."""


def print_and_log(
    message: str,
    log_file: str = "experiments/log.txt",
) -> None:
    """Print and log a message to a file."""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
