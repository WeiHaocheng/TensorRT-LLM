# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which entry a nemotron4 target calls for a bf16-out dense GEMM of a few rows.

`lowm_direct_mm` and `lowm_splitk_mm` each certify the (k, n, m) rows where survey-gemm measured them faster
than `cublas_mm`; at every other (k, n, m) `cublas_mm` is the faster or the only certified choice. This is that
choice as one lookup, so a call site reads

    low_m = lowm_entry(m, k, n)
    y = low_m(x, weight_t) if low_m is not None else cublas_mm(x, weight_t)
"""

from __future__ import annotations

from typing import Optional

from .lowm_direct_mm import NEMOTRON4_TACTICS as _DIRECT
from .lowm_direct_mm import LowmDirectMm, lowm_direct_mm
from .lowm_splitk_mm import NEMOTRON4_TACTICS as _SPLITK
from .lowm_splitk_mm import LowmSplitkMm, lowm_splitk_mm


def lowm_entry(m: int, k: int, n: int) -> Optional[LowmDirectMm | LowmSplitkMm]:
    """The low-M entry certified for an [m, k] x [k, n] bf16 GEMM, or None where cublas_mm is the choice."""
    if (k, n, m) in _DIRECT:
        return lowm_direct_mm
    if (k, n, m) in _SPLITK:
        return lowm_splitk_mm
    return None
