"""
build_shortlist.py - builds data/shortlist.json for the trading agent.

Runs for free in GitHub Actions (see shortlist.yml). It uses free Yahoo Finance
data (yfinance) for daily bars of the S&P 500 plus a few core ETFs, computes the
trend / momentum / backdrop / volume part of the agent's score from COMPLETED
daily bars only, and saves the best candidates.

The agent still checks live quotes, earnings dates and headlines itself, and
re-verifies its final pick with Robinhood's own data before buying.

If the data looks bad (SPY/QQQ missing or under 90% of symbols loaded) the
script exits with an error and does NOT overwrite the last good file, so the
agent sees an old timestamp and refuses to use it.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from io import StringIO
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

CORE_ETFS = ["SPY", "QQQ", "VOO", "VTI", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU"]
BATCH_SIZE = 100
MIN_BARS = 205          # 200 for SMA200, 64 for ROC63, plus a little slack
TOP_N = 30              # candidates written to the file
MIN_PRE_SCORE = 65      # pre_score + up to 20 more (catalyst 15, spread 5) must reach 85
MIN_COVERAGE = 0.90     # fail the run if fewer than 90% of symbols loaded
OUT_PATH = "data/shortlist.json"
ET = ZoneInfo("America/New_York")


def get_universe():
    """S&P 500 constituents from Wikipedia plus the core ETFs."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (shortlist-bot)"}, timeout=30).text
    symbols = pd.read_html(StringIO(html))[0]["Symbol"].tolist()
    symbols = [str(s).strip().replace(".", "-") for s in symbols]
    return sorted(set(symbols + CORE_ETFS))


def download_batch(symbols, tries=3):
    """Download about a year of daily bars (split-adjusted, like Robinhood's default)."""
    for attempt in range(tries):
        try:
            df = yf.download(
                symbols, period="1y", interval="1d", auto_adjust=False,
                group_by="ticker", threads=True, progress=False,
            )
            if df is not None and not df.empty:
                return df
        except Exception as exc:  # network / rate-limit problems
            print(f"batch error (attempt {attempt + 1}): {exc}")
        time.sleep(5 * (attempt + 1))
    return None


def series_for(df, symbol):
    try:
        sub = df[symbol] if isinstance(df.columns, pd.MultiIndex) else df
        return sub[["Close", "Volume"]].dropna(subset=["Close"])
    except Exception:
        return None


def drop_partial_bar(sub):
    """Remove today's bar if the regular session is not finished yet."""
    if sub.empty:
        return sub
    now = datetime.now(ET)
    last = pd.Timestamp(sub.index[-1]).date()
    if last == now.date() and (now.hour, now.minute) < (16, 15):
        return sub.iloc[:-1]
    return sub


def compute_metrics(sub):
    close = sub["Close"].astype(float)
    if len(close) < MIN_BARS:
        return None
    price = float(close.iloc[-1])
    sma50 = float(close.iloc[-50:].mean())
    sma200 = float(close.iloc[-200:].mean())
    if min(price, sma50, sma200) <= 0:
        return None
    return {
        "price": price,
        "sma50": sma50,
        "sma200": sma200,
        "roc21": (price / float(close.iloc[-22]) - 1) * 100,
        "roc63": (price / float(close.iloc[-64]) - 1) * 100,
        "pct_above_sma50": (price / sma50 - 1) * 100,
        "avg_vol_50d": float(sub["Volume"].iloc[-50:].mean()),
        "as_of": str(pd.Timestamp(sub.index[-1]).date()),
    }


def trend_pts(m):
    if m["price"] > m["sma50"] and m["sma50"] > m["sma200"]:
        return 30
    if m["price"] > m["sma50"]:
        return 12
    return 0


def momentum_pts(m):
    if m["pct_above_sma50"] > 8:        # anti-chase rule
        return 0
    if m["roc21"] > 0 and m["roc63"] > 0:
        return 25 if 1 <= m["roc21"] <= 8 else 15
    return 0


def backdrop_pts(spy, qqq):
    spy_up = spy["price"] > spy["sma50"]
    qqq_up = qqq["price"] > qqq["sma50"]
    pts = 7 * int(spy_up) + 7 * int(qqq_up)
    if spy_up and qqq_up and spy["pct_above_sma50"] <= 5 and qqq["pct_above_sma50"] <= 5:
        pts += 6
    return pts


def main():
    universe = get_universe()
    print(f"Universe: {len(universe)} symbols")

    metrics = {}
    for i in range(0, len(universe), BATCH_SIZE):
        batch = universe[i:i + BATCH_SIZE]
        df = download_batch(batch)
        if df is None:
            print(f"Batch starting at {i} failed completely")
            continue
        for sym in batch:
            sub = series_for(df, sym)
            if sub is None or sub.empty:
                continue
            m = compute_metrics(drop_partial_bar(sub))
            if m:
                metrics[sym] = m
        time.sleep(2)

    if "SPY" not in metrics or "QQQ" not in metrics:
        print("ERROR: SPY or QQQ data missing; not writing a new file.")
        sys.exit(1)

    coverage = len(metrics) / len(universe)
    print(f"Coverage: {len(metrics)}/{len(universe)} ({coverage:.0%})")
    if coverage < MIN_COVERAGE:
        print("ERROR: coverage too low; not writing a new file.")
        sys.exit(1)

    market_date = metrics["SPY"]["as_of"]
    spy, qqq = metrics["SPY"], metrics["QQQ"]
    backdrop = backdrop_pts(spy, qqq)

    rows = []
    for sym, m in metrics.items():
        if m["as_of"] != market_date:      # stale or halted symbol
            continue
        t, mo = trend_pts(m), momentum_pts(m)
        vol = 5 if m["avg_vol_50d"] > 5_000_000 else 0
        pre = t + mo + backdrop + vol
        rows.append({
            "ticker": sym,
            "pre_score": pre,
            "trend": t,
            "momentum": mo,
            "volume_pts": vol,
            "price": round(m["price"], 2),
            "sma50": round(m["sma50"], 2),
            "sma200": round(m["sma200"], 2),
            "roc21_pct": round(m["roc21"], 2),
            "roc63_pct": round(m["roc63"], 2),
            "pct_above_sma50": round(m["pct_above_sma50"], 2),
            "avg_vol_50d": int(m["avg_vol_50d"]),
        })

    rows = [r for r in rows if r["pre_score"] >= MIN_PRE_SCORE]
    rows.sort(key=lambda r: (r["pre_score"], r["roc63_pct"]), reverse=True)

    out = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of_date": market_date,
        "status": "ok",
        "universe_size": len(universe),
        "symbols_scored": len(metrics),
        "rubric": ("pre_score = Trend(30) + Momentum(25) + Backdrop(20) + Volume(5), max 80. "
                   "Agent adds Catalyst(15) + Spread(5). Buy only if final score >= 85."),
        "market": {
            "backdrop_pts": backdrop,
            "SPY": {"price": round(spy["price"], 2), "sma50": round(spy["sma50"], 2),
                    "pct_above_sma50": round(spy["pct_above_sma50"], 2)},
            "QQQ": {"price": round(qqq["price"], 2), "sma50": round(qqq["sma50"], 2),
                    "pct_above_sma50": round(qqq["pct_above_sma50"], 2)},
        },
        "candidates": rows[:TOP_N],
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {OUT_PATH}: {len(out['candidates'])} candidates (as of {market_date})")


if __name__ == "__main__":
    main()
