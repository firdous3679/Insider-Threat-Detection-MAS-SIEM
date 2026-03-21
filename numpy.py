"""Minimal subset of NumPy used by this repository."""
from __future__ import annotations
import builtins
import math

class ndarray(list):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            rows, col = key
            subset = self[rows] if isinstance(rows, slice) else [list.__getitem__(self, int(i)) for i in rows]
            return ndarray([row[col] for row in subset])
        if isinstance(key, slice):
            return ndarray(list.__getitem__(self, key))
        if isinstance(key, list):
            if key and isinstance(key[0], bool):
                return ndarray([item for item, keep in zip(self, key) if keep])
            return ndarray([list.__getitem__(self, int(i)) for i in key])
        if hasattr(key, 'tolist'):
            vals = key.tolist()
            if vals and isinstance(vals[0], bool):
                return ndarray([item for item, keep in zip(self, vals) if keep])
            return ndarray([list.__getitem__(self, int(i)) for i in vals])
        return list.__getitem__(self, key)
    def sum(self):
        return builtins.sum(self)
    def mean(self):
        return mean(self)
    def tolist(self):
        return list(self)
    def _binop(self, other, fn):
        if isinstance(other, (list, ndarray)):
            return ndarray([fn(a, b) for a, b in zip(self, other)])
        return ndarray([fn(a, other) for a in self])
    def __eq__(self, other):
        return self._binop(other, lambda a, b: a == b)
    def __ne__(self, other):
        return self._binop(other, lambda a, b: a != b)
    def __gt__(self, other):
        return self._binop(other, lambda a, b: a > b)
    def __ge__(self, other):
        return self._binop(other, lambda a, b: a >= b)
    def __lt__(self, other):
        return self._binop(other, lambda a, b: a < b)
    def __le__(self, other):
        return self._binop(other, lambda a, b: a <= b)
    def __and__(self, other):
        return self._binop(other, lambda a, b: bool(a) and bool(b))
    def __or__(self, other):
        return self._binop(other, lambda a, b: bool(a) or bool(b))


def array(data):
    if isinstance(data, ndarray):
        return data
    return ndarray(list(data))


def mean(values):
    vals = list(values)
    return builtins.sum(vals) / len(vals) if vals else 0.0


def std(values):
    vals = list(values)
    if not vals:
        return 0.0
    m = mean(vals)
    return math.sqrt(builtins.sum((x - m) ** 2 for x in vals) / len(vals))


def exp(x):
    return math.exp(x)
