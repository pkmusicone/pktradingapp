from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd


# -----------------------------
# Helpers: pivots, FVG, metrics
# -----------------------------

def compute_pivots(df: pd.DataFrame, left: int, right: int) -> pd.DataFrame:
    """
    Confirmed pivot highs/lows.
    Pivot high at i if High[i] is max over [i-left, i+right] and unique max.
    Similar for pivot low.

    IMPORTANT (no-lookahead): We "confirm" the pivot only after 'right' bars.
    This is implemented by shifting the raw pivot flags forward by 'right'.
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    n = len(df)

    ph = np.full(n, False)
    pl = np.full(n, False)

    for i in range(left, n - right):
        win_h = highs[i - left : i + right + 1]
        win_l = lows[i - left : i + right + 1]

        if highs[i] == win_h.max() and (win_h == highs[i]).sum() == 1:
            ph[i] = True
        if lows[i] == win_l.min() and (win_l == lows[i]).sum() == 1:
            pl[i] = True

    out = df.copy()
    out["pivot_high_raw"] = ph
    out["pivot_low_raw"] = pl

    # Confirm pivots only after 'right' bars (prevents lookahead bias)
    out["pivot_high"] = out["pivot_high_raw"].shift(right).fillna(False).astype(bool)
    out["pivot_low"] = out["pivot_low_raw"].shift(right).fillna(False).astype(bool)
    return out


def detect_fvg(df: pd.DataFrame, min_ticks: int, tick_size: float) -> List[Dict]:
    """
    3-candle FVG (gap) detection:

    Bullish FVG (at i, middle candle):
      high[i-1] < low[i+1]
      zone = (bottom=high[i-1], top=low[i+1])

    Bearish FVG:
      low[i-1] > high[i+1]
      zone = (bottom=high[i+1], top=low[i-1])

    min_ticks enforces minimum size in ticks.
    """
    zones: List[Dict] = []
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    idx = df.index

    min_size = float(min_ticks) * float(tick_size)

    for i
