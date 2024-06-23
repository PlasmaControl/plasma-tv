import functools
import time
import logging

def timing(func):
    """A decorator that logs the execution time of the function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Executing {func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def debug(func):
    """A decorator that logs function calls with their arguments and the result."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        result = func(*args, **kwargs)
        logging.info(f"Calling {func.__name__}({signature}) -> {result!r}")
        return result
    return wrapper

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    @timing
    @debug
    def example_function(x):
        time.sleep(1)
        return x * 2

    example_function(5)