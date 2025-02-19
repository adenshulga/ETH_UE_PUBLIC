from itertools import accumulate
from collections import deque
import typing as tp


def get_last_elem_from_iter[T](iterable: tp.Iterable[T]) -> T:
    return deque(iterable, maxlen=1).pop()


def efficient_accumulate[T](
    iterable: tp.Iterable[T], operation: tp.Callable[[T, T], T]
) -> T:
    return get_last_elem_from_iter(accumulate(iterable, operation))
