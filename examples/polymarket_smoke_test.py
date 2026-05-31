#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEPTH_EVENT = 1
TRADE_EVENT = 2
DEPTH_CLEAR_EVENT = 3
DEPTH_SNAPSHOT_EVENT = 4
LOCAL_EVENT = 1 << 30
BUY_EVENT = 1 << 29
SELL_EVENT = 1 << 28


def event_kind(events: np.ndarray) -> np.ndarray:
    return events["ev"] & 0xff


def has_flag(ev: int, flag: int) -> bool:
    return (ev & flag) == flag


def summarize_events(data: np.ndarray) -> dict[str, float | int | None]:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    trade_events = 0
    trade_qty = 0.0
    last_trade_price = None
    last_valid_best_bid = None
    last_valid_best_ask = None
    last_valid_mid_price = None

    for row in data:
        ev = int(row["ev"])
        if not has_flag(ev, LOCAL_EVENT):
            continue

        kind = ev & 0xff
        px = float(row["px"])
        qty = float(row["qty"])

        if kind == DEPTH_CLEAR_EVENT:
            if has_flag(ev, BUY_EVENT):
                bids = {price: size for price, size in bids.items() if price > px}
            elif has_flag(ev, SELL_EVENT):
                asks = {price: size for price, size in asks.items() if price < px}
            else:
                bids.clear()
                asks.clear()
        elif kind in (DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT):
            book = bids if has_flag(ev, BUY_EVENT) else asks if has_flag(ev, SELL_EVENT) else None
            if book is not None:
                if qty <= 0:
                    book.pop(px, None)
                else:
                    book[px] = qty
        elif kind == TRADE_EVENT:
            trade_events += 1
            trade_qty += qty
            last_trade_price = px

        current_best_bid = max(bids) if bids else None
        current_best_ask = min(asks) if asks else None
        if current_best_bid is not None and current_best_ask is not None:
            last_valid_best_bid = current_best_bid
            last_valid_best_ask = current_best_ask
            last_valid_mid_price = (current_best_bid + current_best_ask) / 2.0

    best_bid = max(bids) if bids else None
    best_ask = min(asks) if asks else None
    mid_price = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
    kind = event_kind(data)

    return {
        "rows": len(data),
        "depth_events": int(np.sum((kind == DEPTH_EVENT) | (kind == DEPTH_SNAPSHOT_EVENT))),
        "trade_events": trade_events,
        "clear_events": int(np.sum(kind == DEPTH_CLEAR_EVENT)),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid_price,
        "last_valid_best_bid": last_valid_best_bid,
        "last_valid_best_ask": last_valid_best_ask,
        "last_valid_mid_price": last_valid_mid_price,
        "trade_qty": trade_qty,
        "last_trade_price": last_trade_price,
    }


def estimate_arrival_depths(data: np.ndarray, tick_size: float) -> np.ndarray:
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    depths: list[float] = []

    for row in data:
        ev = int(row["ev"])
        if not has_flag(ev, LOCAL_EVENT):
            continue

        kind = ev & 0xff
        px = float(row["px"])
        qty = float(row["qty"])

        if kind == DEPTH_CLEAR_EVENT:
            if has_flag(ev, BUY_EVENT):
                bids = {price: size for price, size in bids.items() if price > px}
            elif has_flag(ev, SELL_EVENT):
                asks = {price: size for price, size in asks.items() if price < px}
            else:
                bids.clear()
                asks.clear()
        elif kind in (DEPTH_EVENT, DEPTH_SNAPSHOT_EVENT):
            book = bids if has_flag(ev, BUY_EVENT) else asks if has_flag(ev, SELL_EVENT) else None
            if book is not None:
                if qty <= 0:
                    book.pop(px, None)
                else:
                    book[px] = qty
        elif kind == TRADE_EVENT and bids and asks:
            mid_price = (max(bids) + min(asks)) / 2.0
            if has_flag(ev, BUY_EVENT):
                depth = (px - mid_price) / tick_size
            elif has_flag(ev, SELL_EVENT):
                depth = (mid_price - px) / tick_size
            else:
                continue
            if depth >= 0:
                depths.append(depth)

    return np.array(depths, dtype=np.float64)


def measure_intensity(arrival_depths: np.ndarray, duration_seconds: float, max_ticks: int) -> tuple[np.ndarray, np.ndarray]:
    ticks = np.arange(1, max_ticks + 1, dtype=np.float64)
    lambda_ = np.array(
        [np.sum(arrival_depths >= tick) / duration_seconds for tick in ticks],
        dtype=np.float64,
    )
    positive = lambda_ > 0
    return ticks[positive], lambda_[positive]


def fit_exponential_intensity(ticks: np.ndarray, lambda_: np.ndarray) -> dict[str, float]:
    x = ticks.astype(np.float64)
    y = np.log(lambda_.astype(np.float64))
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    residual = np.sum((y - fitted) ** 2)
    total = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 if total == 0 else 1.0 - residual / total
    return {"A": float(np.exp(intercept)), "k": float(-slope), "r2": float(r2)}


def estimate_as_intensity(data: np.ndarray, tick_size: float, max_ticks: int) -> dict[str, float | int | np.ndarray]:
    arrival_depths = estimate_arrival_depths(data, tick_size)
    local_events = (data["ev"] & LOCAL_EVENT) == LOCAL_EVENT
    duration_seconds = (data["local_ts"][local_events].max() - data["local_ts"][local_events].min()) / 1_000_000_000
    ticks, lambda_ = measure_intensity(arrival_depths, duration_seconds, max_ticks)
    fit = fit_exponential_intensity(ticks, lambda_)
    return {
        "samples": int(len(arrival_depths)),
        "duration_seconds": float(duration_seconds),
        "tick_size": float(tick_size),
        "ticks": ticks,
        "lambda": lambda_,
        "A": fit["A"],
        "k": fit["k"],
        "r2": fit["r2"],
    }


def print_as_estimate(estimate: dict[str, float | int | np.ndarray]) -> None:
    print("AS/GLFT intensity estimate:")
    print(f"samples: {estimate['samples']}")
    print(f"duration seconds: {estimate['duration_seconds']}")
    print(f"tick size: {estimate['tick_size']}")
    print(f"A: {estimate['A']}")
    print(f"k: {estimate['k']}")
    print(f"r2: {estimate['r2']}")


def plot_intensity_fit(estimate: dict[str, float | int | np.ndarray], output: Path) -> None:
    import matplotlib.pyplot as plt

    ticks = estimate["ticks"]
    lambda_ = estimate["lambda"]
    fitted = estimate["A"] * np.exp(-estimate["k"] * ticks)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(ticks, lambda_, marker="o", label="empirical lambda")
    plt.plot(ticks, fitted, label="A * exp(-k * delta)")
    plt.xlabel("distance from mid (ticks)")
    plt.ylabel("arrival intensity / second")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def print_summary(summary: dict[str, float | int | None]) -> None:
    print(f"rows: {summary['rows']}")
    print(f"depth events: {summary['depth_events']}")
    print(f"trade events: {summary['trade_events']}")
    print(f"clear events: {summary['clear_events']}")
    print(f"best bid: {summary['best_bid']}")
    print(f"best ask: {summary['best_ask']}")
    print(f"mid price: {summary['mid_price']}")
    print(f"last valid best bid: {summary['last_valid_best_bid']}")
    print(f"last valid best ask: {summary['last_valid_best_ask']}")
    print(f"last valid mid price: {summary['last_valid_mid_price']}")
    print(f"trade qty: {summary['trade_qty']}")
    print(f"last trade price: {summary['last_trade_price']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a quick summary of a Polymarket hftbacktest .npz file")
    parser.add_argument("data", nargs="?", type=Path, default=Path("data/polymarket.npz"))
    parser.add_argument("--estimate-as", action="store_true", help="estimate AS/GLFT arrival intensity")
    parser.add_argument("--tick-size", type=float, default=0.01, help="price tick size used to convert distance to ticks")
    parser.add_argument("--max-ticks", type=int, default=100, help="maximum quote distance in ticks")
    parser.add_argument("--plot", type=Path, help="optional output path for intensity fit plot")
    args = parser.parse_args()

    data = np.load(args.data)["data"]
    print_summary(summarize_events(data))
    if args.estimate_as:
        estimate = estimate_as_intensity(data, args.tick_size, args.max_ticks)
        print_as_estimate(estimate)
        if args.plot is not None:
            plot_intensity_fit(estimate, args.plot)
            print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
