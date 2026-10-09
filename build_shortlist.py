"""
Build data/shortlist.json for the trading agent.

GitHub Actions uses Finviz to build a candidate universe, then Yahoo Finance
daily bars to calculate trend, momentum, market-backdrop, and volume scores.
Only completed daily bars are used.

If Finviz fails or returns too few stocks, the script falls back to S&P 500
constituents from Wikipedia. It does not replace the existing JSON if required
market data is missing or data coverage is too low.

This script does not place trades or call Robinhood MCP. The agent should
independently verify live data and trade details before any order.
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


# ── Configuration ──────────────────────────────────────────────

CORE_ETFS = [
    "SPY", "QQQ", "VOO", "VTI",
    "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU",
]

BATCH_SIZE = 100
MIN_BARS = 205
TOP_N = 30
MIN_PRE_SCORE = 65
MIN_COVERAGE = 0.80
MIN_FINVIZ_SYMBOLS = 50

OUT_PATH = "data/shortlist.json"
ET = ZoneInfo("America/New_York")

# These names and values must match finvizfinance's accepted filter options.
FINVIZ_FILTERS = {
    "Market Cap.": "+Large (over $10bln)",
    "Average Volume": "Over 1M",
    "Price": "Over $5",
    "P/E": "Under 50",
    "Current Ratio": "Over 1",
    "Debt/Equity": "Under 1",
}


# ── Universe ────────────────────────────────────────────────────

def normalize_ticker(value):
    """Normalize a ticker for Yahoo Finance, e.g. BRK.B to BRK-B."""
    return str(value).strip().upper().replace(".", "-")


def get_finviz_universe():
    """Return Finviz-screened tickers, or None if the screen fails."""
    try:
        from finvizfinance.screener.overview import Overview

        screener = Overview()
        screener.set_filter(filters_dict=FINVIZ_FILTERS)
        frame = screener.screener_view(
            limit=2000,
            verbose=0,
            sleep_sec=1,
        )

        if frame is None or frame.empty or "Ticker" not in frame.columns:
            print("[Finviz] No usable ticker results; using Wikipedia fallback.")
            return None

        tickers = {
            normalize_ticker(ticker)
            for ticker in frame["Ticker"].dropna().tolist()
            if str(ticker).strip()
        }

        if len(tickers) < MIN_FINVIZ_SYMBOLS:
            print(
                f"[Finviz] Only {len(tickers)} stocks returned; "
                "using Wikipedia fallback."
            )
            return None

        universe = sorted(tickers | set(CORE_ETFS))
        print(f"[Finviz] Screen returned {len(tickers)} stocks.")
        return universe

    except Exception as exc:
        print(f"[Finviz] Screen failed: {exc}")
        print("[Finviz] Using Wikipedia S&P 500 fallback.")
        return None


def get_wikipedia_universe():
    """Return S&P 500 constituents from Wikipedia plus the core ETFs."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (shortlist-bot)"},
        timeout=30,
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))
    if not tables or "Symbol" not in tables[0].columns:
        raise RuntimeError(
            "Could not find the S&P 500 ticker table on Wikipedia."
        )

    tickers = {
        normalize_ticker(ticker)
        for ticker in tables[0]["Symbol"].dropna().tolist()
    }
    return sorted(tickers | set(CORE_ETFS))


def get_universe():
    """Use Finviz first, with an S&P 500 fallback."""
    universe = get_finviz_universe()
    if universe:
        return universe, "finviz"

    return get_wikipedia_universe(), "wikipedia_sp500_fallback"


# ── Download and indicator calculations ─────────────────────────

def download_batch(symbols, tries=3):
    """Download approximately one year of daily bars from Yahoo Finance."""
    for attempt in range(tries):
        try:
            frame = yf.download(
                symbols,
                period="1y",
                interval="1d",
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if frame is not None and not frame.empty:
                return frame

        except Exception as exc:
            print(
                f"[Yahoo Finance] Batch attempt {attempt + 1} failed: {exc}"
            )

        if attempt < tries - 1:
            time.sleep(5 * (attempt + 1))

    return None


def series_for(frame, symbol):
    """Extract Close and Volume for one ticker from a yfinance batch."""
    try:
        if isinstance(frame.columns, pd.MultiIndex):
            if symbol not in frame.columns.get_level_values(0):
                return None
            data = frame[symbol]
        else:
            data = frame

        if "Close" not in data.columns or "Volume" not in data.columns:
            return None

        return data[["Close", "Volume"]].dropna(subset=["Close"])

    except Exception:
        return None


def drop_partial_bar(data):
    """Drop today's daily bar if the regular US session is not finished."""
    if data is None or data.empty:
        return data

    now = datetime.now(ET)
    last_date = pd.Timestamp(data.index[-1]).date()

    if last_date == now.date() and (now.hour, now.minute) < (16, 15):
        return data.iloc[:-1]

    return data


def compute_metrics(data):
    """Calculate technical measures used by the shortlist score."""
    if data is None or data.empty:
        return None

    close = data["Close"].astype(float).dropna()
    volume = data["Volume"].astype(float)

    if len(close) < MIN_BARS:
        return None

    price = float(close.iloc[-1])
    sma50 = float(close.iloc[-50:].mean())
    sma200 = float(close.iloc[-200:].mean())

    if min(price, sma50, sma200) <= 0:
        return None

    close_22_days_ago = float(close.iloc[-22])
    close_64_days_ago = float(close.iloc[-64])

    if close_22_days_ago <= 0 or close_64_days_ago <= 0:
        return None

    return {
        "price": price,
        "sma50": sma50,
        "sma200": sma200,
        "roc21": (price / close_22_days_ago - 1) * 100,
        "roc63": (price / close_64_days_ago - 1) * 100,
        "pct_above_sma50": (price / sma50 - 1) * 100,
        "avg_vol_50d": float(volume.iloc[-50:].mean()),
        "as_of": str(pd.Timestamp(close.index[-1]).date()),
    }


# ── Scoring ─────────────────────────────────────────────────────

def trend_pts(metrics):
    if metrics["price"] > metrics["sma50"] > metrics["sma200"]:
        return 30
    if metrics["price"] > metrics["sma50"]:
        return 12
    return 0


def momentum_pts(metrics):
    # Anti-chase rule: do not award momentum points if price is too extended.
    if metrics["pct_above_sma50"] > 8:
        return 0

    if metrics["roc21"] > 0 and metrics["roc63"] > 0:
        return 25 if 1 <= metrics["roc21"] <= 8 else 15

    return 0


def backdrop_pts(spy, qqq):
    spy_up = spy["price"] > spy["sma50"]
    qqq_up = qqq["price"] > qqq["sma50"]

    points = 7 * int(spy_up) + 7 * int(qqq_up)

    if (
        spy_up
        and qqq_up
        and spy["pct_above_sma50"] <= 5
        and qqq["pct_above_sma50"] <= 5
    ):
        points += 6

    return points


# ── Build output ────────────────────────────────────────────────

def write_json_atomically(data):
    """Write a temporary JSON file, then replace the previous output."""
    output_dir = os.path.dirname(OUT_PATH)
    os.makedirs(output_dir, exist_ok=True)

    temp_path = f"{OUT_PATH}.tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")

    os.replace(temp_path, OUT_PATH)


def main():
    universe, universe_source = get_universe()

    if not universe:
        print("ERROR: The selected ticker universe is empty.")
        sys.exit(1)

    print(f"Universe source: {universe_source}")
    print(f"Universe size: {len(universe)} symbols")

    metrics_by_symbol = {}

    for start in range(0, len(universe), BATCH_SIZE):
        batch = universe[start:start + BATCH_SIZE]
        frame = download_batch(batch)

        if frame is None:
            print(f"[Yahoo Finance] Batch starting at {start} failed.")
            continue

        for symbol in batch:
            data = series_for(frame, symbol)
            data = drop_partial_bar(data)
            metrics = compute_metrics(data)

            if metrics:
                metrics_by_symbol[symbol] = metrics

        print(
            f"Scored data for {len(metrics_by_symbol)} symbols "
            f"after batch {start // BATCH_SIZE + 1}."
        )
        time.sleep(2)

    if "SPY" not in metrics_by_symbol or "QQQ" not in metrics_by_symbol:
        print(
            "ERROR: SPY or QQQ data is missing; "
            "leaving the previous JSON untouched."
        )
        sys.exit(1)

    coverage = len(metrics_by_symbol) / len(universe)
    print(
        f"Coverage: {len(metrics_by_symbol)}/{len(universe)} "
        f"({coverage:.0%})"
    )

    if coverage < MIN_COVERAGE:
        print(
            "ERROR: Coverage is below the minimum; "
            "leaving the previous JSON untouched."
        )
        sys.exit(1)

    market_date = metrics_by_symbol["SPY"]["as_of"]
    spy = metrics_by_symbol["SPY"]
    qqq = metrics_by_symbol["QQQ"]
    market_backdrop = backdrop_pts(spy, qqq)

    candidates = []

    for symbol, metrics in metrics_by_symbol.items():
        # Avoid mixing data dates older than the SPY market-data date.
        if metrics["as_of"] != market_date:
            continue

        trend = trend_pts(metrics)
        momentum = momentum_pts(metrics)
        volume_points = 5 if metrics["avg_vol_50d"] > 5_000_000 else 0
        pre_score = trend + momentum + market_backdrop + volume_points

        candidates.append({
            "ticker": symbol,
            "pre_score": pre_score,
            "trend": trend,
            "momentum": momentum,
            "backdrop": market_backdrop,
            "volume_pts": volume_points,
            "price": round(metrics["price"], 2),
            "sma50": round(metrics["sma50"], 2),
            "sma200": round(metrics["sma200"], 2),
            "roc21_pct": round(metrics["roc21"], 2),
            "roc63_pct": round(metrics["roc63"], 2),
            "pct_above_sma50": round(metrics["pct_above_sma50"], 2),
            "avg_vol_50d": int(metrics["avg_vol_50d"]),
        })

    candidates = [
        row for row in candidates
        if row["pre_score"] >= MIN_PRE_SCORE
    ]
    candidates.sort(
        key=lambda row: (row["pre_score"], row["roc63_pct"]),
        reverse=True,
    )

    output = {
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "as_of_date": market_date,
        "status": "ok",
        "universe_source": universe_source,
        "universe_size": len(universe),
        "symbols_scored": len(metrics_by_symbol),
        "coverage_pct": round(coverage * 100, 1),
        "finviz_filters": (
            FINVIZ_FILTERS if universe_source == "finviz" else None
        ),
        "rubric": (
            "pre_score = Trend(30) + Momentum(25) + Backdrop(20) + "
            "Volume(5), maximum 80. The agent must independently verify "
            "live quotes, spread, news, and earnings before considering a trade."
        ),
        "market": {
            "backdrop_pts": market_backdrop,
            "SPY": {
                "price": round(spy["price"], 2),
                "sma50": round(spy["sma50"], 2),
                "pct_above_sma50": round(spy["pct_above_sma50"], 2),
            },
            "QQQ": {
                "price": round(qqq["price"], 2),
                "sma50": round(qqq["sma50"], 2),
                "pct_above_sma50": round(qqq["pct_above_sma50"], 2),
            },
        },
        "candidate_count": min(len(candidates), TOP_N),
        "candidates": candidates[:TOP_N],
    }

    write_json_atomically(output)

    print(
        f"Wrote {OUT_PATH}: {len(output['candidates'])} candidates "
        f"(market data as of {market_date})."
    )


if __name__ == "__main__":
    main()
