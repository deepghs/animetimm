import logging
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import BoundedSemaphore
from typing import Iterable, Callable, Any, Optional

from tqdm import tqdm


class BoundedThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers=None, max_pending=None, **kwargs):
        super().__init__(max_workers=max_workers, **kwargs)
        if max_pending is not None:
            self._semaphore = BoundedSemaphore(max_pending)
        else:
            self._semaphore = None

    def submit(self, fn, *args, **kwargs):
        if self._semaphore:
            self._semaphore.acquire()
        try:
            future = super().submit(self._wrapper, fn, *args, **kwargs)
        except:
            if self._semaphore:
                self._semaphore.release()
            raise
        return future

    def _wrapper(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        finally:
            if self._semaphore:
                self._semaphore.release()


def parallel_call(iterable: Iterable, fn: Callable[[Any], None], total: Optional[int] = None,
                  desc: Optional[str] = None, max_workers: Optional[int] = None, max_pending: Optional[int] = None):
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            total = None

    pg = tqdm(total=total, desc=desc or f'Process with {fn!r}')
    if not max_workers:
        max_workers = min(os.cpu_count(), 16)
    tp = BoundedThreadPoolExecutor(max_workers=max_workers, max_pending=max_pending)

    def _fn(item):
        try:
            return fn(item)
        except Exception as err:
            logging.exception(f'Error when processing {item!r} - {err!r}')
            raise
        finally:
            pg.update()

    for item in iterable:
        tp.submit(_fn, item)

    tp.shutdown(wait=True)
