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
    args = parser.parse_args()

    data = np.load(args.data)["data"]
    print_summary(summarize_events(data))


if __name__ == "__main__":
    main()
