#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
#
#  Copyright (c) 2020-2025 Emanuele Ballarin <emanuele@ballarin.cc>
#  Released under the terms of the MIT License
#  (see: https://url.ballarin.cc/mitlicense)
#
# ──────────────────────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: MIT
#
# ──────────────────────────────────────────────────────────────────────────────
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from torch import distributed as dist
from torch import Tensor

# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = ["reduce_accumulate_keepalive"]


def reduce_accumulate_keepalive(reduction_tensor: Tensor, accumulator: int | float):
    dist.barrier()
    dist.all_reduce(reduction_tensor, op=dist.ReduceOp.SUM)
    dist.barrier()
    accumulator += reduction_tensor.item()
    reduction_tensor.zero_()
    return accumulator
