import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "polymarket_smoke_test.py"

DEPTH_EVENT = 1
TRADE_EVENT = 2
DEPTH_CLEAR_EVENT = 3
DEPTH_SNAPSHOT_EVENT = 4
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


def load_example_module():
    spec = importlib.util.spec_from_file_location("polymarket_smoke_test", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPolymarketSmokeExample(unittest.TestCase):
    def test_summarizes_depth_and_trade_events(self):
        module = load_example_module()
        data = np.array(
            [
                (LOCAL_EVENT | BUY_EVENT | DEPTH_SNAPSHOT_EVENT, 100, 110, 0.49, 10.0, 0, 0, 0.0),
                (LOCAL_EVENT | SELL_EVENT | DEPTH_SNAPSHOT_EVENT, 101, 111, 0.51, 12.0, 0, 0, 0.0),
                (LOCAL_EVENT | BUY_EVENT | TRADE_EVENT, 102, 112, 0.50, 1.5, 1, 0, 0.0),
            ],
            dtype=EVENT_DTYPE,
        )

        summary = module.summarize_events(data)

        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["depth_events"], 2)
        self.assertEqual(summary["trade_events"], 1)
        self.assertEqual(summary["best_bid"], 0.49)
        self.assertEqual(summary["best_ask"], 0.51)
        self.assertEqual(summary["mid_price"], 0.50)
        self.assertEqual(summary["trade_qty"], 1.5)

    def test_keeps_last_valid_book_when_data_ends_with_clear(self):
        module = load_example_module()
        data = np.array(
            [
                (LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT, 100, 110, 0.40, 3.0, 0, 0, 0.0),
                (LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT, 101, 111, 0.60, 4.0, 0, 0, 0.0),
                (LOCAL_EVENT | DEPTH_CLEAR_EVENT, 102, 112, 0.0, 0.0, 0, 0, 0.0),
            ],
            dtype=EVENT_DTYPE,
        )

        summary = module.summarize_events(data)

        self.assertIsNone(summary["best_bid"])
        self.assertIsNone(summary["best_ask"])
        self.assertEqual(summary["last_valid_best_bid"], 0.40)
        self.assertEqual(summary["last_valid_best_ask"], 0.60)
        self.assertEqual(summary["last_valid_mid_price"], 0.50)

    def test_cli_prints_summary_for_npz_file(self):
        data = np.array(
            [
                (LOCAL_EVENT | BUY_EVENT | DEPTH_EVENT, 100, 110, 0.40, 3.0, 0, 0, 0.0),
                (LOCAL_EVENT | SELL_EVENT | DEPTH_EVENT, 101, 111, 0.60, 4.0, 0, 0, 0.0),
                (LOCAL_EVENT | SELL_EVENT | TRADE_EVENT, 102, 112, 0.60, 2.0, 2, 0, 0.0),
            ],
            dtype=EVENT_DTYPE,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.npz"
            np.savez_compressed(path, data=data)
            result = subprocess.run(
                [sys.executable, str(EXAMPLE), str(path)],
                check=True,
                capture_output=True,
                text=True,
            )

        self.assertIn("rows: 3", result.stdout)
        self.assertIn("trade events: 1", result.stdout)
        self.assertIn("best bid: 0.4", result.stdout)
        self.assertIn("best ask: 0.6", result.stdout)


if __name__ == "__main__":
    unittest.main()
