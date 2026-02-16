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

    IMPORTANT (no-lookahead): We "confirm" the pivot only after 'right' bars.
    Implemented by shifting the raw pivot flags forward by 'right'.
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

    out["pivot_high"] = out["pivot_high_raw"].shift(right).fillna(False).astype(bool)
    out["pivot_low"] = out["pivot_low_raw"].shift(right).fillna(False).astype(bool)
    return out


def detect_fvg(df: pd.DataFrame, min_ticks: int, tick_size: float) -> List[Dict]:
    """
    3-candle FVG (gap) detection:

    Bullish FVG at i if high[i-1] < low[i+1]
      zone bottom = high[i-1], top = low[i+1]

    Bearish FVG at i if low[i-1] > high[i+1]
      zone bottom = high[i+1], top = low[i-1]
    """
    zones: List[Dict] = []
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    idx = df.index

    min_size = float(min_ticks) * float(tick_size)

    for i in range(1, len(df) - 1):
        # bullish gap
        if high[i - 1] < low[i + 1]:
            top = float(low[i + 1])
            bottom = float(high[i - 1])
            if (top - bottom) >= min_size:
                zones.append(
                    {
                        "type": "bull_fvg",
                        "created_i": i,
                        "created_time": idx[i],
                        "bottom": bottom,
                        "top": top,
                    }
                )

        # bearish gap
        if low[i - 1] > high[i + 1]:
            top = float(low[i - 1])
            bottom = float(high[i + 1])
            if (top - bottom) >= min_size:
                zones.append(
                    {
                        "type": "bear_fvg",
                        "created_i": i,
                        "created_time": idx[i],
                        "bottom": bottom,
                        "top": top,
                    }
                )

    return zones


def rr_to_price(entry: float, stop: float, rr: float, direction: str) -> float:
    risk = abs(entry - stop)
    if risk == 0:
        return entry
    return entry + rr * risk if direction == "long" else entry - rr * risk


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


# -----------------------------
# Backtest engine
# -----------------------------

@dataclass
class Trade:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    r_multiple: float
    bars_held: int
    reason: str


def resample_ohlc(df5: pd.DataFrame, rule: str) -> pd.DataFrame:
    o = df5["open"].resample(rule).first()
    h = df5["high"].resample(rule).max()
    l = df5["low"].resample(rule).min()
    c = df5["close"].resample(rule).last()

    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c})

    if "volume" in df5.columns:
        out["volume"] = df5["volume"].resample(rule).sum()

    return out.dropna()


def _ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        # Assumption: naive timestamps are UTC.
        return df.tz_localize("UTC")
    return df.tz_convert("UTC")


def apply_session_filter(df: pd.DataFrame, sessions_cfg: Dict) -> pd.DataFrame:
    """
    Fixed session filter using between_time(), avoids boolean-index inversion issues.
    """
    tz = sessions_cfg.get("timezone", "America/Chicago")
    mode = str(sessions_cfg.get("mode", "RTH")).upper()

    df = _ensure_utc_index(df)
    local = df.tz_convert(tz)

    if mode == "RTH":
        start = sessions_cfg["rth"]["start"]
        end = sessions_cfg["rth"]["end"]
        local = local.between_time(start, end, inclusive="both")
    elif mode == "ETH":
        # MVP: keep all
        pass
    else:
        # ALL / unknown
        pass

    for w in sessions_cfg.get("no_trade_windows", []):
        ns = w["start"]
        ne = w["end"]
        blocked = local.between_time(ns, ne, inclusive="both")
        local = local.drop(index=blocked.index)

    return local.tz_convert("UTC")


def backtest(df5: pd.DataFrame, cfg: Dict) -> Dict:
    # --- execution params
    tick_size = float(cfg["execution"]["tick_size"])
    slippage_ticks = int(cfg["execution"]["slippage_ticks"])
    commission_side = float(cfg["execution"]["commission_per_contract_per_side"])
    contracts = int(cfg["risk_model"]["position_sizing"]["contracts"])
    point_value = float(cfg["execution"].get("point_value", 2.0))

    # --- session filter
    df5 = df5.copy()
    df5 = apply_session_filter(df5, cfg.get("sessions", {}))
    df5 = df5.sort_index()

    # --- resample
    df4h = resample_ohlc(df5, "4h")

    # --- pivots
    s5 = cfg["structure_model"]["swing"]
    b4 = cfg["bias_model"]["swing"]
    df5p = compute_pivots(df5, int(s5["pivot_left"]), int(s5["pivot_right"]))
    df4p = compute_pivots(df4h, int(b4["pivot_left"]), int(b4["pivot_right"]))

    # --- fvg
    fvg_cfg = cfg["fvg_model"]
    zones_all = detect_fvg(df5, int(fvg_cfg["min_size_ticks"]), tick_size)

    # --- arrays
    times5 = df5.index
    close5 = df5["close"].to_numpy()
    high5 = df5["high"].to_numpy()
    low5 = df5["low"].to_numpy()

    # --- 4h bias tracker
    bos_requires_close_4h = bool(cfg["bias_model"]["trend_rule"].get("bos_requires_close", True))
    last_4h_swing_high = None
    last_4h_swing_low = None
    last_4h_bos = None
    no_bos_swings = 0
    consolidation_n = int(cfg["bias_model"]["consolidation_rule"]["n_swings"])

    # --- 5m structure tracker
    bos_requires_close_5m = bool(cfg["structure_model"]["bos"]["requires_close"])
    last_5m_swing_high = None
    last_5m_swing_low = None
    last_bos_i = None
    bos_now = None

    # --- entry/risk configs
    direction_mode = cfg["entry_model"]["direction_mode"]
    allow_counter = bool(cfg["entry_model"].get("allow_counter_bias", False))

    entry_price_cfg = cfg["entry_model"]["entry_price"]
    retrace_cfg = cfg["entry_model"]["retrace_rule"]
    max_bars_after_fvg = int(retrace_cfg["max_bars_after_fvg"])

    stop_cfg = cfg["risk_model"]["stop_loss"]
    stop_buffer_ticks = int(stop_cfg.get("buffer_ticks", 0))

    tp_cfg = cfg["risk_model"]["take_profit"]
    rr_targets = [float(x) for x in tp_cfg.get("rr_targets", [2.0])]

    time_stop_cfg = cfg["risk_model"].get("time_stop", {"enabled": False})
    time_stop_enabled = bool(time_stop_cfg.get("enabled", False))
    max_hold_bars = int(time_stop_cfg.get("max_hold_bars", 999999))

    max_dist_ticks = int(cfg.get("filters", {}).get("max_distance_from_fvg_ticks", 10**9))

    filter_impact = {
        "bias_mismatch": 0,
        "fvg_stale": 0,
        "fvg_not_within_bos_window": 0,
        "retrace_timeout": 0,
        "max_distance_from_fvg": 0,
    }

    trades: List[Trade] = []
    equity_r = [0.0]

    in_position = False
    pos: Dict = {}

    # setup states: 0 waiting BOS, 1 waiting FVG, 2 waiting retrace
    state = 0
    setup: Dict = {}

    def update_bias(current_time: pd.Timestamp) -> str:
        nonlocal last_4h_swing_high, last_4h_swing_low, last_4h_bos, no_bos_swings

        df4_sub = df4p.loc[df4p.index <= current_time]
        if len(df4_sub) < 5:
            return "unknown"

        last_row = df4_sub.iloc[-1]

        if bool(last_row["pivot_high"]):
            last_4h_swing_high = float(last_row["high"])
            no_bos_swings += 1
        if bool(last_row["pivot_low"]):
            last_4h_swing_low = float(last_row["low"])
            no_bos_swings += 1

        c = float(last_row["close"])

        if last_4h_swing_high is not None:
            broke = (c > last_4h_swing_high) if bos_requires_close_4h else (float(last_row["high"]) > last_4h_swing_high)
            if broke:
                last_4h_bos = "bull"
                no_bos_swings = 0

        if last_4h_swing_low is not None:
            broke = (c < last_4h_swing_low) if bos_requires_close_4h else (float(last_row["low"]) < last_4h_swing_low)
            if broke:
                last_4h_bos = "bear"
                no_bos_swings = 0

        if no_bos_swings >= consolidation_n:
            return "consolidation"
        if last_4h_bos == "bull":
            return "bullish"
        if last_4h_bos == "bear":
            return "bearish"
        return "unknown"

    for i in range(len(df5p)):
        t = times5[i]
        bias = update_bias(t)

        # swings update
        if bool(df5p.iloc[i]["pivot_high"]):
            last_5m_swing_high = float(df5p.iloc[i]["high"])
        if bool(df5p.iloc[i]["pivot_low"]):
            last_5m_swing_low = float(df5p.iloc[i]["low"])

        # BOS detection
        bos_now = None
        if last_5m_swing_high is not None:
            if (close5[i] > last_5m_swing_high) if bos_requires_close_5m else (high5[i] > last_5m_swing_high):
                bos_now = "bull"
        if last_5m_swing_low is not None:
            if (close5[i] < last_5m_swing_low) if bos_requires_close_5m else (low5[i] < last_5m_swing_low):
                bos_now = "bear"
        if bos_now is not None:
            last_bos_i = i

        # manage position
        if in_position:
            direction = pos["direction"]
            entry = pos["entry_price"]
            stop = pos["stop_price"]
            targets = pos["targets"]
            opened_i = pos["opened_i"]

            # time stop
            if time_stop_enabled and (i - opened_i) >= max_hold_bars:
                exit_px = close5[i] - (slippage_ticks * tick_size if direction == "long" else -slippage_ticks * tick_size)
                risk = abs(entry - stop)
                r = 0.0 if risk == 0 else ((exit_px - entry) / risk if direction == "long" else (entry - exit_px) / risk)
                risk_dollars = risk * point_value * contracts
                fees = 2.0 * commission_side * contracts
                if risk_dollars > 0:
                    r -= fees / risk_dollars
                trades.append(Trade(direction, pos["entry_time"], entry, t, exit_px, float(r), i - opened_i, "time_stop"))
                equity_r.append(equity_r[-1] + float(r))
                in_position = False
                pos = {}
                state = 0
                setup = {}
                continue

            bar_low = low5[i]
            bar_high = high5[i]

            hit_stop = (bar_low <= stop) if direction == "long" else (bar_high >= stop)

            hit_target = False
            target_px = None
            for tp in targets:
                if direction == "long" and bar_high >= tp:
                    hit_target = True
                    target_px = tp
                    break
                if direction == "short" and bar_low <= tp:
                    hit_target = True
                    target_px = tp
                    break

            if hit_stop:
                exit_px = stop - (slippage_ticks * tick_size if direction == "long" else -slippage_ticks * tick_size)
                r = -1.0
                risk = abs(entry - stop)
                risk_dollars = risk * point_value * contracts
                fees = 2.0 * commission_side * contracts
                if risk_dollars > 0:
                    r -= fees / risk_dollars
                trades.append(Trade(direction, pos["entry_time"], entry, t, exit_px, float(r), i - opened_i, "stop"))
                equity_r.append(equity_r[-1] + float(r))
                in_position = False
                pos = {}
                state = 0
                setup = {}
                continue

            if hit_target and target_px is not None:
                exit_px = target_px - (slippage_ticks * tick_size if direction == "long" else -slippage_ticks * tick_size)
                risk = abs(entry - stop)
                r = 0.0 if risk == 0 else ((exit_px - entry) / risk if direction == "long" else (entry - exit_px) / risk)
                risk_dollars = risk * point_value * contracts
                fees = 2.0 * commission_side * contracts
                if risk_dollars > 0:
                    r -= fees / risk_dollars
                trades.append(Trade(direction, pos["entry_time"], entry, t, exit_px, float(r), i - opened_i, "target"))
                equity_r.append(equity_r[-1] + float(r))
                in_position = False
                pos = {}
                state = 0
                setup = {}
                continue

        if in_position:
            continue

        # --- setup creation
        if state == 0 and bos_now is not None:
            if bias not in ("bullish", "bearish"):
                continue

            bias_dir = "long" if bias == "bullish" else "short"
            allowed_dirs = {bias_dir}
            if allow_counter or direction_mode in ("both", "counter_bias"):
                allowed_dirs.add("short" if bias_dir == "long" else "long")

            candidate_dir = bias_dir if direction_mode == "with_bias" else ("short" if bias_dir == "long" else "long")
            if candidate_dir not in allowed_dirs:
                filter_impact["bias_mismatch"] += 1
                continue

            needed_opposing_bos = "bear" if candidate_dir == "long" else "bull"
            if bos_now == needed_opposing_bos:
                setup = {"direction": candidate_dir, "bos_i": i, "bos_time": t}
                state = 1

        # --- wait FVG
        if state == 1:
            matching = [z for z in zones_all if z["created_i"] == i]
            if not matching:
                if last_bos_i is not None and (i - setup["bos_i"]) > int(fvg_cfg["must_form_within_n_bars_of_bos"]):
                    filter_impact["fvg_not_within_bos_window"] += 1
                    state = 0
                    setup = {}
                continue

            direction = setup["direction"]
            need = "bull_fvg" if direction == "long" else "bear_fvg"
            z = next((x for x in matching if x["type"] == need), None)
            if z is None:
                continue

            setup["fvg"] = z
            setup["fvg_i"] = i
            setup["fvg_time"] = t
            state = 2

        # --- wait retrace
        if state == 2:
            direction = setup["direction"]
            z = setup["fvg"]

            if (i - int(z["created_i"])) > int(fvg_cfg["max_age_bars"]):
                filter_impact["fvg_stale"] += 1
                state = 0
                setup = {}
                continue

            if (i - int(setup["fvg_i"])) > max_bars_after_fvg:
                filter_impact["retrace_timeout"] += 1
                state = 0
                setup = {}
                continue

            bottom, top = float(z["bottom"]), float(z["top"])
            mid = (bottom + top) / 2.0

            mode = entry_price_cfg["mode"]
            if mode == "fvg_anchor":
                anchor = entry_price_cfg["anchor"]
                base_px = bottom if anchor == "start" else (top if anchor == "end" else mid)
                offset = int(entry_price_cfg.get("custom_price_offset_ticks", 0)) * tick_size
                entry_px = base_px + (offset if direction == "long" else -offset)
            elif mode == "defined_price":
                entry_px = float(entry_price_cfg["price"])
            else:
                entry_px = mid

            nearest_edge = bottom if abs(entry_px - bottom) < abs(entry_px - top) else top
            dist_ticks = int(round(abs(entry_px - nearest_edge) / tick_size))
            if dist_ticks > max_dist_ticks:
                filter_impact["max_distance_from_fvg"] += 1
                state = 0
                setup = {}
                continue

            if direction == "long":
                touched = (low5[i] <= entry_px <= high5[i]) or (
                    low5[i] <= entry_px and retrace_cfg["touch_mode"] == "touch_or_trade_through"
                )
            else:
                touched = (low5[i] <= entry_px <= high5[i]) or (
                    high5[i] >= entry_px and retrace_cfg["touch_mode"] == "touch_or_trade_through"
                )

            if not touched:
                continue

            if direction == "long":
                if last_5m_swing_low is None:
                    continue
                stop_px = float(last_5m_swing_low) - (stop_buffer_ticks * tick_size)
            else:
                if last_5m_swing_high is None:
                    continue
                stop_px = float(last_5m_swing_high) + (stop_buffer_ticks * tick_size)

            entry_fill = entry_px + (slippage_ticks * tick_size if direction == "long" else -slippage_ticks * tick_size)
            targets = [rr_to_price(entry_fill, stop_px, rr, direction) for rr in rr_targets]

            in_position = True
            pos = {
                "direction": direction,
                "entry_time": t,
                "entry_price": entry_fill,
                "stop_price": stop_px,
                "targets": targets,
                "opened_i": i,
            }

            state = 0
            setup = {}

    trade_rows = [
        {
            "direction": tr.direction,
            "entry_time": tr.entry_time,
            "entry_price": tr.entry_price,
            "exit_time": tr.exit_time,
            "exit_price": tr.exit_price,
            "r_multiple": tr.r_multiple,
            "bars_held": tr.bars_held,
            "reason": tr.reason,
        }
        for tr in trades
    ]
    trades_df = pd.DataFrame(trade_rows)

    if len(trades_df) == 0:
        metrics = {
            "trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "expectancy_r": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_r": 0.0,
            "avg_hold_time_bars": 0.0,
            "filter_impact": filter_impact,
        }
        return {"metrics": metrics, "trades": trades_df, "equity_r": np.array(equity_r, dtype=float)}

    wins = trades_df.loc[trades_df["r_multiple"] > 0, "r_multiple"]
    losses = trades_df.loc[trades_df["r_multiple"] < 0, "r_multiple"]

    win_rate = float((trades_df["r_multiple"] > 0).mean())
    avg_r = float(trades_df["r_multiple"].mean())
    expectancy_r = avg_r

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0
    profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    eq = np.array(equity_r, dtype=float)
    mdd = max_drawdown(eq)

    metrics = {
        "trades": int(len(trades_df)),
        "win_rate": win_rate,
        "avg_r": avg_r,
        "expectancy_r": expectancy_r,
        "profit_factor": profit_factor,
        "max_drawdown_r": mdd,
        "avg_hold_time_bars": float(trades_df["bars_held"].mean()),
        "filter_impact": filter_impact,
    }

    return {"metrics": metrics, "trades": trades_df, "equity_r": eq}
