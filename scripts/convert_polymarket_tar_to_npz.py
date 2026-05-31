#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEPTH_EVENT = 1
TRADE_EVENT = 2
DEPTH_CLEAR_EVENT = 3
DEPTH_SNAPSHOT_EVENT = 4

EXCH_EVENT = 1 << 31
LOCAL_EVENT = 1 << 30
BUY_EVENT = 1 << 29
SELL_EVENT = 1 << 28

EVENT_DTYPE = np.dtype(
    [
        ("ev", "u8"),
        ("exch_ts", "i8"),
        ("local_ts", "i8"),
        ("px", "f8"),
        ("qty", "f8"),
        ("order_id", "u8"),
        ("ival", "i8"),
        ("fval", "f8"),
    ],
    align=True,
)


def decode_decimal128(raw: bytes, scale: float) -> float:
    return int.from_bytes(raw, "little", signed=True) / scale


def read_parquet_group(archive: tarfile.TarFile, marker: str) -> pa.Table | None:
    tables = []
    for member in archive.getmembers():
        if not member.isfile() or not member.name.endswith(".parquet"):
            continue
        if marker not in member.name:
            continue
        data = archive.extractfile(member)
        if data is None:
            continue
        tables.append(pq.read_table(io.BytesIO(data.read())))
    if not tables:
        return None
    return pa.concat_tables(tables)


def side_flag(side: int) -> int:
    if side == 1:
        return BUY_EVENT
    if side == 2:
        return SELL_EVENT
    if side == 0:
        return 0
    raise ValueError(f"unsupported side value: {side}")


def convert_order_book(table: pa.Table) -> np.ndarray:
    rows = table.to_pydict()
    out = np.zeros(table.num_rows, dtype=EVENT_DTYPE)

    for i in range(table.num_rows):
        action = rows["action"][i]
        side = rows["side"][i]
        flags = rows["flags"][i]

        if action == 4:
            ev = DEPTH_CLEAR_EVENT
        elif action == 1 and flags == 32:
            ev = DEPTH_SNAPSHOT_EVENT
        elif action in (1, 2, 3):
            ev = DEPTH_EVENT
        else:
            raise ValueError(f"unsupported order book action: {action}")

        ev |= side_flag(side)

        out[i] = (
            ev,
            rows["ts_event"][i],
            rows["ts_init"][i],
            decode_decimal128(rows["price"][i], 1e16),
            decode_decimal128(rows["size"][i], 1e18),
            rows["order_id"][i],
            flags,
            0.0,
        )

    return out


def convert_trades(table: pa.Table) -> np.ndarray:
    rows = table.to_pydict()
    out = np.zeros(table.num_rows, dtype=EVENT_DTYPE)

    for i in range(table.num_rows):
        ev = TRADE_EVENT | side_flag(rows["aggressor_side"][i])
        trade_id = rows["trade_id"][i]
        order_id = int(trade_id[:16], 16) if isinstance(trade_id, str) else 0

        out[i] = (
            ev,
            rows["ts_event"][i],
            rows["ts_init"][i],
            decode_decimal128(rows["price"][i], 1e16),
            decode_decimal128(rows["size"][i], 1e18),
            order_id,
            0,
            0.0,
        )

    return out


def correct_event_order(data: np.ndarray) -> np.ndarray:
    sorted_exch_index = np.argsort(data["exch_ts"], kind="mergesort")
    sorted_local_index = np.argsort(data["local_ts"], kind="mergesort")
    out = np.zeros(data.shape[0] * 2, dtype=EVENT_DTYPE)

    out_rn = 0
    exch_rn = 0
    local_rn = 0

    while True:
        if exch_rn < len(data):
            sorted_exch = data[sorted_exch_index[exch_rn]]
        if local_rn < len(data):
            sorted_local = data[sorted_local_index[local_rn]]

        if (
            exch_rn < len(data)
            and local_rn < len(data)
            and sorted_exch["exch_ts"] == sorted_local["exch_ts"]
            and sorted_exch["local_ts"] == sorted_local["local_ts"]
            and sorted_exch["ev"] == sorted_local["ev"]
            and sorted_exch["px"] == sorted_local["px"]
            and sorted_exch["qty"] == sorted_local["qty"]
            and sorted_exch["order_id"] == sorted_local["order_id"]
        ):
            out[out_rn] = sorted_exch
            out[out_rn]["ev"] = int(out[out_rn]["ev"]) | EXCH_EVENT | LOCAL_EVENT
            out_rn += 1
            exch_rn += 1
            local_rn += 1
        elif (
            (
                exch_rn < len(data)
                and local_rn < len(data)
                and sorted_exch["exch_ts"] == sorted_local["exch_ts"]
                and sorted_exch["local_ts"] < sorted_local["local_ts"]
            )
            or (
                exch_rn < len(data)
                and (local_rn >= len(data) or sorted_exch["exch_ts"] < sorted_local["exch_ts"])
            )
        ):
            out[out_rn] = sorted_exch
            out[out_rn]["ev"] = int(out[out_rn]["ev"]) | EXCH_EVENT
            out_rn += 1
            exch_rn += 1
        elif local_rn < len(data):
            out[out_rn] = sorted_local
            out[out_rn]["ev"] = int(out[out_rn]["ev"]) | LOCAL_EVENT
            out_rn += 1
            local_rn += 1
        elif exch_rn < len(data):
            out[out_rn] = sorted_exch
            out[out_rn]["ev"] = int(out[out_rn]["ev"]) | EXCH_EVENT
            out_rn += 1
            exch_rn += 1
        else:
            break

    return out[:out_rn]


def validate_event_order(data: np.ndarray) -> None:
    exch_events = (data["ev"] & EXCH_EVENT) == EXCH_EVENT
    local_events = (data["ev"] & LOCAL_EVENT) == LOCAL_EVENT

    if np.any(np.diff(data["exch_ts"][exch_events]) < 0):
        raise ValueError("exchange events are out of order")
    if np.any(np.diff(data["local_ts"][local_events]) < 0):
        raise ValueError("local events are out of order")

    latency = data["local_ts"] - data["exch_ts"]
    if np.any(latency < 0):
        raise ValueError("negative feed latency found")


def convert(input_archive: Path, output_file: Path) -> np.ndarray:
    with tarfile.open(input_archive, "r:gz") as archive:
        order_book = read_parquet_group(archive, "/order_book_deltas/")
        trades = read_parquet_group(archive, "/trade_tick/")

    arrays = []
    if order_book is not None:
        arrays.append(convert_order_book(order_book))
    if trades is not None:
        arrays.append(convert_trades(trades))
    if not arrays:
        raise ValueError("no order_book_deltas or trade_tick parquet files found")

    data = np.concatenate(arrays)
    data = correct_event_order(data)
    validate_event_order(data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, data=data)
    return data


def event_kind(data: np.ndarray) -> np.ndarray:
    return data["ev"] & 0xff


def summarize(data: np.ndarray, output_file: Path) -> None:
    kind = event_kind(data)
    print(f"saved: {output_file}")
    print(f"rows: {len(data)}")
    print(f"exchange time: {data['exch_ts'].min()} -> {data['exch_ts'].max()}")
    print(f"local time: {data['local_ts'].min()} -> {data['local_ts'].max()}")
    print(f"depth rows: {int(np.sum(kind == DEPTH_EVENT))}")
    print(f"trade rows: {int(np.sum(kind == TRADE_EVENT))}")
    print(f"snapshot rows: {int(np.sum(kind == DEPTH_SNAPSHOT_EVENT))}")
    print(f"clear rows: {int(np.sum(kind == DEPTH_CLEAR_EVENT))}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Polymarket parquet tar archive to hftbacktest npz")
    parser.add_argument("input", type=Path, help="input data.tar.gz")
    parser.add_argument("output", type=Path, help="output .npz file")
    args = parser.parse_args()

    data = convert(args.input, args.output)
    summarize(data, args.output)


if __name__ == "__main__":
    main()
