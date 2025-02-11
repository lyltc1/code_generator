import time
from contextlib import contextmanager

@contextmanager
def timer(text='', do=True):
    if do:
        start = time.time()
        try:
            yield
        finally:
            print(f'{text}: {time.time() - start:.4}s')
    else:
        yield