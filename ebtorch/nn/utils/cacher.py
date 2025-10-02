#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from collections.abc import Callable
from typing import Any

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = ["fromcache"]


# ~~ Utilities ~~ ──────────────────────────────────────────────────────────────
def _normargs(args, kwargs, kwpos: tuple[str, ...], kwdef: tuple) -> tuple:
    largs: int = len(args)
    n_args: list = list(args) + [None] * (len(kwpos) - largs)
    for i, argname in enumerate(kwpos[largs:], start=largs):
        if argname in kwargs:
            n_args[i] = kwargs[argname]
        else:
            n_args[i] = kwdef[i]
    return tuple(n_args)


def _args2keyer(kwpos: tuple[str, ...], kwdef: tuple) -> Callable:
    def _args2key(*args, **kwargs) -> tuple:
        return _normargs(args, kwargs, kwpos, kwdef)

    return _args2key


def _retlookup(key: tuple, dictionary: dict) -> Any | None:
    return dictionary[key] if key in dictionary else None


# ~~ Cache retrieval function ~~ ───────────────────────────────────────────────
def fromcache(
    func: Callable,
    *,
    kwpos: tuple[str, ...],
    kwdef: tuple,
    cache: dict,
    updateable: bool = True,
) -> Callable:
    def _cached_func(*args, **kwargs):
        key: tuple = _args2keyer(kwpos, kwdef)(*args, **kwargs)
        if (rlk := _retlookup(key, cache)) is not None:
            return rlk
        else:
            if updateable:
                funcret = func(*args, **kwargs)
                cache[key] = funcret
                return funcret
            else:
                raise ValueError("Cache miss, but cache is not updateable.")

    return _cached_func
