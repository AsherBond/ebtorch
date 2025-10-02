#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ──────────────────────────────────────────────────────────────────────────────
# ~~ Imports ~~ ────────────────────────────────────────────────────────────────
from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

# ──────────────────────────────────────────────────────────────────────────────
# ~~ Exports ~~ ────────────────────────────────────────────────────────────────
__all__: list[str] = ["realnum", "strdev", "numlike", "tensorlike", "actvt"]
# ──────────────────────────────────────────────────────────────────────────────
realnum = int | float
strdev = str | int | torch.device
tensorlike = Tensor | np.ndarray
numlike = realnum | tensorlike
actvt = torch.nn.Module | Callable[[Tensor], Tensor]
# ──────────────────────────────────────────────────────────────────────────────
