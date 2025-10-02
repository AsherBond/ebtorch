#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from collections.abc import Iterable
from functools import partial

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = ["variadic_attrs"]


# ~~ Utilities ~~ ──────────────────────────────────────────────────────────────
def _str_to_bool(s: str, onesym: bool = False) -> bool:
    osl: list[str] = ["t", "y", "1"]
    if onesym:
        return s.lower() in osl
    return s.lower() in (osl + ["true", "yes"])


def _any_to_bool(x, onesym: bool = False) -> bool:
    if isinstance(x, str):
        return _str_to_bool(x, onesym)
    return bool(x)


def _str_to_booltuple(s: str, sep: str | None = None) -> tuple[bool, ...]:
    if sep is not None:
        return tuple(map(_str_to_bool, s.split(sep)))
    return tuple(map(partial(_str_to_bool, onesym=True), [*s]))


def _any_to_booltuple(x: str | Iterable[str | bool], sep: str | None = None) -> tuple[bool, ...]:
    if isinstance(x, str):
        return _str_to_booltuple(x, sep)
    return tuple(map(_any_to_bool, x))


def variadic_attrs(
    selfobj,
    varsel: Iterable[str | bool] | None = None,
    insep: str | None = None,
    outsep: str = "_",
):
    odict: dict = selfobj.__getstate__()
    odkeys: tuple[str, ...] = tuple(odict.keys())
    lodk: int = len(odkeys)
    varsel: Iterable = varsel if varsel is not None else ([True] * lodk)
    bvsel: tuple[bool, ...] = _any_to_booltuple(varsel, insep)
    strtuple: tuple[str, ...] = tuple(str(odict[odkeys[i]]) if bvsel[i] else "" for i in range(lodk))
    return (outsep.join(strtuple)).strip().strip(outsep)
