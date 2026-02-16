import json
import streamlit as st
import pandas as pd
from engine import backtest

st.set_page_config(page_title="MNQ Strategy Backtester (MVP)", layout="wide")

st.title("MNQ Strategy Backtester (MVP)")
st.caption("Load 5-minute MNQ data, load a strategy config, run a backtest.")

left, right = st.columns([1, 1])

with left:
    st.subheader("1) Upload MNQ 5m CSV")
    st.write("CSV must include columns: time, open, high, low, close (volume optional).")
    file = st.file_uploader("Upload CSV", type=["csv"])

with right:
    st.subheader("2) Strategy Config JSON")
    st.write("Paste your strategy config JSON here.")
    default_cfg = {
      "schema_version": "1.0",
      "strategy_id": "mnq_ict_fvg_v1",
      "name": "MNQ HTF Bias + 5m BOS → FVG → Retrace",
      "symbol": "MNQ",
      "timeframes": {"base": "5m", "context": ["15m"], "bias": "4h"},
      "sessions": {
        "mode": "RTH",
        "timezone": "America/Chicago",
        "rth": {"start": "08:30", "end": "15:00"},
        "eth": {"start": "17:00", "end": "16:00"},
        "no_trade_windows": [{"start": "08:30", "end": "08:35"}]
      },
      "execution": {
        "commission_per_contract_per_side": 0.52,
        "slippage_ticks": 1,
        "tick_size": 0.25,
        "point_value": 2.0,
        "fill_model": {"type": "touch", "limit_requires_trade_through": True},
        "max_positions": 1
      },
      "bias_model": {
        "timeframe": "4h",
        "swing": {"pivot_left": 3, "pivot_right": 3},
        "trend_rule": {"type": "structure", "bos_requires_close": True},
        "consolidation_rule": {"type": "no_bos_for_n_swings", "n_swings": 3}
      },
      "structure_model": {
        "timeframe": "5m",
        "swing": {"pivot_left": 2, "pivot_right": 2},
        "bos": {"requires_close": True, "use_body_break_only": False}
      },
      "fvg_model": {
        "timeframe": "5m",
        "min_size_ticks": 2,
        "max_age_bars": 30,
        "must_form_within_n_bars_of_bos": 8,
        "displacement_filter": {"enabled": False, "min_body_atr_mult": 1.0, "atr_period": 14}
      },
      "entry_model": {
        "direction_mode": "with_bias",
        "allow_counter_bias": False,
        "trigger_sequence": ["BOS_OPPOSING", "FVG_FORMS", "RETRACE_TO_FVG"],
        "entry_price": {"mode": "fvg_anchor", "anchor": "mid", "custom_price_offset_ticks": 0},
        "retrace_rule": {"touch_mode": "touch_or_trade_through", "must_touch_midpoint_if_anchor_mid": True, "max_bars_after_fvg": 20}
      },
      "risk_model": {
        "position_sizing": {"mode": "fixed_contracts", "contracts": 1},
        "stop_loss": {"mode": "swing_invalidation", "buffer_ticks": 2},
        "take_profit": {
          "mode": "fixed_rr_multiple",
          "rr_targets": [1.0, 2.0],
          "scale_out": [{"rr": 1.0, "close_pct": 0.5}, {"rr": 2.0, "close_pct": 0.5}]
        },
        "time_stop": {"enabled": True, "max_hold_bars": 48}
      },
      "filters": {"max_distance_from_fvg_ticks": 12, "one_trade_per_session": False},
      "reporting": {"metrics": ["win_rate","avg_r","expectancy_r","profit_factor","max_drawdown","avg_hold_time_bars","filter_impact"]}
    }

    cfg_text = st.text_area("Strategy JSON", value=json.dumps(default_cfg, indent=2), height=420)

run = st.button("Run Backtest", type="primary")

if file and run:
    df = pd.read_csv(file)

    # normalize columns
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"time", "open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        st.error(f"CSV missing required columns: {required}")
        st.stop()

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).set_index("time").sort_index()

    try:
        cfg = json.loads(cfg_text)
    except Exception as e:
        st.error(f"Invalid JSON config: {e}")
        st.stop()

    result = backtest(df, cfg)
    metrics = result["metrics"]
    trades = result["trades"]

    st.subheader("Metrics")
    mcols = st.columns(5)
    mcols[0].metric("Trades", metrics["trades"])
    mcols[1].metric("Win rate", f'{metrics["win_rate"]*100:.1f}%')
    mcols[2].metric("Avg R", f'{metrics["avg_r"]:.2f}')
    mcols[3].metric("Profit factor", f'{metrics["profit_factor"]:.2f}' if metrics["profit_factor"] != float("inf") else "∞")
    mcols[4].metric("Max DD (R)", f'{metrics["max_drawdown_r"]:.2f}')

    st.write("Filter impact (why trades didn’t happen):", metrics["filter_impact"])

    st.subheader("Trades")
    st.dataframe(trades, use_container_width=True)

    st.subheader("Equity Curve (R)")
    st.line_chart(result["equity_r"])
else:
    st.info("Upload your MNQ 5m CSV, then click **Run Backtest**.")
