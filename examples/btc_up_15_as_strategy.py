#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


FEE_RATE = 0.07
MAKER_REBATE_MULTIPLIER = 0.2


class RiskScaleResult:
    def __init__(self, fair_probability: float, risk_scale: float, sigma_total: float) -> None:
        self.fair_probability = fair_probability
        self.risk_scale = risk_scale
        self.sigma_total = sigma_total


class ReservationPriceResult:
    def __init__(self, price: float, skew: float) -> None:
        self.price = price
        self.skew = skew


class QuoteResult:
    def __init__(self, reservation_price: float, half_spread: float, bid: float, ask: float) -> None:
        self.reservation_price = reservation_price
        self.half_spread = half_spread
        self.bid = bid
        self.ask = ask


class OrderPlan:
    def __init__(self, price: float, liquidity: str, taker_slippage_cost: float) -> None:
        self.price = price
        self.liquidity = liquidity
        self.taker_slippage_cost = taker_slippage_cost


class FeeEquivalentState:
    def __init__(self) -> None:
        self.cumulative_maker_rebate = 0.0
        self.cumulative_taker_fee = 0.0
        self._processed_exec_qty_by_order_id: dict[int, float] = {}

    @property
    def net_fee_equivalent(self) -> float:
        return self.cumulative_maker_rebate - self.cumulative_taker_fee

    def apply_fill(self, order_id: int, exec_qty: float, exec_price: float, maker: bool) -> None:
        previous_qty = self._processed_exec_qty_by_order_id.get(order_id, 0.0)
        delta_qty = exec_qty - previous_qty
        if delta_qty <= 0:
            return
        if maker:
            self.cumulative_maker_rebate += maker_rebate_equivalent(delta_qty, exec_price)
        else:
            self.cumulative_taker_fee += taker_fee_equivalent(delta_qty, exec_price)
        self._processed_exec_qty_by_order_id[order_id] = exec_qty


class SettlementResult:
    def __init__(self, settlement_price: float, inventory_value: float, final_pnl: float) -> None:
        self.settlement_price = settlement_price
        self.inventory_value = inventory_value
        self.final_pnl = final_pnl


class StrategyConfig:
    def __init__(
        self,
        fill_rate: float = 0.0,
        tau_liq_cap: float = 900.0,
        u_max: float = 5.0,
        t_exit_seconds: float = 30.0,
        gamma: float = 0.05,
    ) -> None:
        self.fill_rate = fill_rate
        self.tau_liq_cap = tau_liq_cap
        self.u_max = u_max
        self.t_exit_seconds = t_exit_seconds
        self.gamma = gamma


class VolatilityEstimator:
    """
    EWMA volatility estimator with proper time-unit normalization.

    Maintains a running per-second variance estimate using exponential
    weighting.  Each update accepts a (price, timestamp_ns) pair and
    computes the time-normalized log return:

        r_norm = log(S_t / S_{t-1}) / sqrt(dt_seconds)

    The variance is then decayed by actual elapsed time so the effective
    half-life stays consistent regardless of tick frequency:

        alpha(dt) = 1 - exp(-dt * ln2 / halflife_seconds)
        var  ←  (1 - alpha) * var  +  alpha * r_norm^2

    sigma_per_second = sqrt(var) is what should be passed to
    compute_risk_scale() instead of the old realized-vol-from-open value.

    The estimator is designed to persist across consecutive market windows
    so each new window starts from the previous window's final sigma rather
    than from zero.
    """

    def __init__(
        self,
        halflife_seconds: float = 90.0,
        initial_sigma_per_second: float = 0.0,
    ) -> None:
        self.halflife = halflife_seconds
        self._variance: float = initial_sigma_per_second ** 2
        self._last_ts_ns: int | None = None
        self._last_log_price: float | None = None

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def sigma_per_second(self) -> float:
        return math.sqrt(self._variance)

    def update(self, price: float, ts_ns: int) -> float:
        """Feed one (price, timestamp_ns) observation; returns current sigma."""
        if price <= 0 or not math.isfinite(price):
            return self.sigma_per_second

        log_price = math.log(price)

        if self._last_ts_ns is None or self._last_log_price is None:
            self._last_ts_ns = ts_ns
            self._last_log_price = log_price
            return self.sigma_per_second

        dt = (ts_ns - self._last_ts_ns) / 1e9
        if dt <= 0:
            return self.sigma_per_second

        # time-normalised return: log-return per sqrt(second)
        r_norm = (log_price - self._last_log_price) / math.sqrt(dt)

        # EWMA decay scaled by actual elapsed time
        alpha = 1.0 - math.exp(-dt * math.log(2.0) / self.halflife)
        self._variance = (1.0 - alpha) * self._variance + alpha * r_norm * r_norm

        self._last_ts_ns = ts_ns
        self._last_log_price = log_price
        return self.sigma_per_second

    def warm_up(self, spot_ts: np.ndarray, spot_px: np.ndarray,
                from_ts_ns: int, to_ts_ns: int) -> None:
        """Feed a slice of the spot series to advance the estimator without running a backtest."""
        mask = (spot_ts > from_ts_ns) & (spot_ts <= to_ts_ns)
        for ts, px in zip(spot_ts[mask], spot_px[mask]):
            self.update(float(px), int(ts))

    def clone(self) -> "VolatilityEstimator":
        est = VolatilityEstimator(self.halflife)
        est._variance = self._variance
        est._last_ts_ns = self._last_ts_ns
        est._last_log_price = self._last_log_price
        return est


def tau_liq(inventory: float, cfg: StrategyConfig) -> float:
    if cfg.fill_rate <= 0:
        return cfg.tau_liq_cap
    return min(abs(inventory) / cfg.fill_rate, cfg.tau_liq_cap)


def u_val(tau: float, inventory: float, cfg: StrategyConfig) -> float:
    u_max = cfg.u_max
    if tau <= cfg.t_exit_seconds:
        return u_max
    if inventory == 0:
        return 1.0
    liquidation_time = tau_liq(inventory, cfg)
    phi = max(0.0, 1.0 - tau / liquidation_time)
    value = 1.0 + (u_max - 1.0) * phi
    return min(value, u_max)


def can_submit_buy(position: float, max_position: float, tau_seconds: float, cfg: StrategyConfig) -> bool:
    if tau_seconds <= cfg.t_exit_seconds:
        return position < 0
    return position < max_position


def can_submit_sell(position: float, max_position: float, tau_seconds: float, cfg: StrategyConfig) -> bool:
    if tau_seconds <= cfg.t_exit_seconds:
        return position > 0
    return position > -max_position


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def half_spread(k: float, gamma: float, fair_probability: float, u_value: float) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    risk_scale = fair_probability * (1.0 - fair_probability)
    return 1.0 / k + 0.5 * gamma * risk_scale * u_value


def reservation_price(
    fair_price: float,
    inventory: float,
    gamma: float,
    fair_probability: float,
    u_value: float,
    lo: float,
    hi: float,
) -> ReservationPriceResult:
    risk_scale = fair_probability * (1.0 - fair_probability)
    skew = inventory * gamma * risk_scale * u_value
    price = min(max(fair_price - skew, lo), hi)
    return ReservationPriceResult(price=price, skew=skew)


def compute_quotes(
    fair_price: float,
    inventory: float,
    gamma: float,
    fair_probability: float,
    u_value: float,
    k: float,
    lo: float,
    hi: float,
) -> QuoteResult:
    reservation = reservation_price(fair_price, inventory, gamma, fair_probability, u_value, lo, hi)
    spread = half_spread(k, gamma, fair_probability, u_value)
    bid = max(lo, reservation.price - spread)
    ask = min(hi, reservation.price + spread)
    return QuoteResult(reservation.price, spread, bid, ask)


def plan_buy_order(bid_px: float, best_ask: float, tick_size: float, order_qty: float) -> OrderPlan:
    if bid_px >= best_ask:
        return OrderPlan(
            price=best_ask + tick_size,
            liquidity="taker",
            taker_slippage_cost=tick_size * order_qty,
        )
    return OrderPlan(price=bid_px, liquidity="maker", taker_slippage_cost=0.0)


def plan_sell_order(ask_px: float, best_bid: float, tick_size: float, order_qty: float) -> OrderPlan:
    if ask_px <= best_bid:
        return OrderPlan(
            price=max(tick_size, best_bid - tick_size),
            liquidity="taker",
            taker_slippage_cost=tick_size * order_qty,
        )
    return OrderPlan(price=ask_px, liquidity="maker", taker_slippage_cost=0.0)


def settle_remaining_inventory(strike: float, final_spot: float, final_position: float, balance: float) -> SettlementResult:
    settlement_price = 1.0 if final_spot > strike else 0.0
    inventory_value = final_position * settlement_price
    final_pnl = balance + inventory_value
    return SettlementResult(settlement_price, inventory_value, final_pnl)


def adjust_final_pnl(
    final_pnl: float,
    cumulative_taker_slippage_cost: float,
    cumulative_taker_fee: float,
    cumulative_maker_rebate: float,
) -> float:
    return final_pnl - cumulative_taker_slippage_cost - cumulative_taker_fee + cumulative_maker_rebate


def fee_equivalent(qty: float, price: float, fee_rate: float = FEE_RATE) -> float:
    return qty * fee_rate * price * (1.0 - price)


def maker_rebate_equivalent(qty: float, price: float, fee_rate: float = FEE_RATE) -> float:
    return fee_equivalent(qty, price, fee_rate) * MAKER_REBATE_MULTIPLIER


def taker_fee_equivalent(qty: float, price: float, fee_rate: float = FEE_RATE) -> float:
    return fee_equivalent(qty, price, fee_rate)


def realized_volatility_from_open(spot_prices: np.ndarray) -> float:
    prices = np.asarray(spot_prices, dtype=np.float64)
    prices = prices[np.isfinite(prices) & (prices > 0)]
    if len(prices) < 2:
        return 0.0
    returns = np.diff(np.log(prices))
    if len(returns) == 0:
        return 0.0
    return float(np.std(returns))


def realized_volatility_ewma(
    spot_ts: np.ndarray,
    spot_px: np.ndarray,
    current_ts_ns: int,
    halflife_seconds: float = 90.0,
    window_seconds: float = 900.0,
) -> float:
    """
    EWMA realized volatility with time-unit normalization.

    Looks back `window_seconds` from `current_ts_ns`, computes
    time-normalised log returns (per sqrt-second), then weights
    each squared return by exp(-age / halflife).  Because the window
    always spans the last 15 min it naturally carries over volatility
    from the previous market window with no extra state.

    Returns sigma in units of per-sqrt-second (pass directly to
    compute_risk_scale in place of the old sigma_spot).
    """
    window_ns = int(window_seconds * 1_000_000_000)
    lo = current_ts_ns - window_ns
    mask = (spot_ts > lo) & (spot_ts <= current_ts_ns)
    ts = spot_ts[mask]
    px = spot_px[mask]

    if len(px) < 2:
        return 0.0

    dt = np.diff(ts.astype(np.float64)) / 1e9          # seconds between ticks
    log_ret = np.diff(np.log(px.astype(np.float64)))

    valid = dt > 0
    if not valid.any():
        return 0.0

    dt = dt[valid]
    log_ret = log_ret[valid]
    r_norm = log_ret / np.sqrt(dt)                      # per-sqrt-second

    # age of each return (seconds ago from current time)
    ret_ts = ts[1:][valid]
    ages = (current_ts_ns - ret_ts) / 1e9
    weights = np.exp(-ages * math.log(2.0) / halflife_seconds)
    weights /= weights.sum()

    return math.sqrt(float(np.dot(weights, r_norm ** 2)))


def compute_risk_scale(spot_price: float, strike: float, sigma_spot: float, tau_seconds: float) -> RiskScaleResult:
    if spot_price <= 0 or strike <= 0:
        raise ValueError("spot_price and strike must be positive")

    sigma_total = float(sigma_spot) * math.sqrt(max(float(tau_seconds), 0.0))
    if sigma_total <= 0 or not math.isfinite(sigma_total):
        if spot_price == strike:
            fair_probability = 0.5
        else:
            fair_probability = 1.0 - 1e-4 if spot_price > strike else 1e-4
    else:
        fair_probability = normal_cdf(math.log(spot_price / strike) / sigma_total)
        fair_probability = min(max(fair_probability, 1e-4), 1.0 - 1e-4)

    return RiskScaleResult(
        fair_probability=fair_probability,
        risk_scale=fair_probability * (1.0 - fair_probability),
        sigma_total=sigma_total,
    )


def normalize_market_name(market: str) -> str:
    return market.removeprefix("./")


def market_member_prefix(market: str) -> str:
    return normalize_market_name(market) + "/catalog/data/"


def convert_market_archive_to_npz(input_archive: Path, market: str, output_file: Path) -> np.ndarray:
    import io
    import tarfile

    import pyarrow as pa
    import pyarrow.parquet as pq

    from scripts.convert_polymarket_tar_to_npz import (
        convert_order_book,
        convert_trades,
        correct_event_order,
        validate_event_order,
    )

    prefix = market_member_prefix(market)
    order_book_tables = []
    trade_tables = []
    with tarfile.open(input_archive, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".parquet"):
                continue
            if prefix + "order_book_deltas/" in member.name:
                f = archive.extractfile(member)
                if f is not None:
                    order_book_tables.append(pq.read_table(io.BytesIO(f.read())))
            elif prefix + "trade_tick/" in member.name:
                f = archive.extractfile(member)
                if f is not None:
                    trade_tables.append(pq.read_table(io.BytesIO(f.read())))
    arrays = []
    if order_book_tables:
        arrays.append(convert_order_book(pa.concat_tables(order_book_tables)))
    if trade_tables:
        arrays.append(convert_trades(pa.concat_tables(trade_tables)))
    if not arrays:
        raise ValueError(f"no data found for market {market}")
    data = correct_event_order(np.concatenate(arrays))
    validate_event_order(data)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, data=data)
    return data


def load_btcusd_archive(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import io
    import tarfile

    import pyarrow.parquet as pq

    points = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".parquet"):
                continue
            f = archive.extractfile(member)
            if f is None:
                continue
            table = pq.read_table(io.BytesIO(f.read()), columns=["price_ts_ms", "price"])
            d = table.to_pydict()
            for ts, px in zip(d["price_ts_ms"], d["price"]):
                if ts is not None and px is not None:
                    points.append((int(ts) * 1_000_000, float(px)))
    points = sorted(set(points))
    if not points:
        raise ValueError(f"no BTCUSD prices found in {path}")
    return np.array([p[0] for p in points], dtype=np.int64), np.array([p[1] for p in points], dtype=np.float64)


def load_spot_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)["data"]
    return data["exch_ts"].astype(np.int64), data["px"].astype(np.float64)


def nearest_spot_index(spot_ts: np.ndarray, timestamp_ns: int) -> int:
    idx = int(np.searchsorted(spot_ts, timestamp_ns, side="right") - 1)
    if idx < 0:
        return 0
    if idx >= len(spot_ts):
        return len(spot_ts) - 1
    return idx


def collect_risk_scale_path(
    spot_ts: np.ndarray,
    spot_px: np.ndarray,
    start_ns: int,
    end_ns: int,
    interval_ns: int,
) -> list[dict[str, float | int]]:
    start_idx = nearest_spot_index(spot_ts, start_ns)
    strike = float(spot_px[start_idx])
    rows: list[dict[str, float | int]] = []

    for timestamp_ns in range(start_ns, end_ns + 1, interval_ns):
        idx = nearest_spot_index(spot_ts, timestamp_ns)
        current_prices = spot_px[start_idx : idx + 1]
        sigma_spot = realized_volatility_from_open(current_prices)
        tau_seconds = max((end_ns - timestamp_ns) / 1_000_000_000, 0.0)
        result = compute_risk_scale(float(spot_px[idx]), strike, sigma_spot, tau_seconds)
        rows.append(
            {
                "timestamp_ns": timestamp_ns,
                "spot_price": float(spot_px[idx]),
                "strike": strike,
                "tau_seconds": tau_seconds,
                "sigma_spot_realized": sigma_spot,
                "sigma_total": result.sigma_total,
                "fair_probability": result.fair_probability,
                "risk_scale": result.risk_scale,
            }
        )

    return rows


def write_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_hftbacktest_market(
    market_npz: Path,
    btcusd_archive: Path,
    market: str,
    output_csv: Path,
    k: float = 150.0,
    gamma: float = 0.05,
    order_qty: float = 1.0,
    max_position: float = 10.0,
    refresh_interval_ns: int = 1_000_000_000,
) -> list[dict[str, float | int]]:
    from hftbacktest import BacktestAsset, FILLED, GTC, GTX, LIMIT, HashMapMarketDepthBacktest
    from hftbacktest.order import PARTIALLY_FILLED

    spot_ts, spot_px = load_btcusd_archive(btcusd_archive)
    end_ns = market_end_ns(market)
    start_ns = end_ns - 900 * 1_000_000_000
    cfg = StrategyConfig(fill_rate=0.35, tau_liq_cap=900.0, u_max=5.0, t_exit_seconds=120.0, gamma=gamma)

    tick_size = 0.01
    asset = (
        BacktestAsset()
        .data([str(market_npz)])
        .linear_asset(1.0)
        .no_partial_fill_exchange()
        .trading_value_fee_model(0.0, 0.0)
        .tick_size(tick_size)
        .lot_size(0.001)
    )
    hbt = HashMapMarketDepthBacktest([asset])
    asset_no = 0
    rows: list[dict[str, float | int]] = []
    order_id_seq = 1
    start_idx = nearest_spot_index(spot_ts, start_ns)
    final_idx = nearest_spot_index(spot_ts, end_ns)
    strike = float(spot_px[start_idx])
    final_spot = float(spot_px[final_idx])
    settlement_price = 1.0 if final_spot > strike else 0.0
    cumulative_taker_slippage_cost = 0.0
    fee_state = FeeEquivalentState()

    try:
        while hbt.elapse(refresh_interval_ns) == 0:
            orders = hbt.orders(asset_no)
            order_values = orders.values()
            while order_values.has_next():
                order = order_values.get()
                if order.status in (FILLED, PARTIALLY_FILLED):
                    fee_state.apply_fill(
                        order_id=int(order.order_id),
                        exec_qty=float(order.exec_qty),
                        exec_price=float(order.exec_price),
                        maker=bool(order.arr[0]["maker"]),
                    )
                if order.cancellable:
                    hbt.cancel(asset_no, order.order_id, False)
            hbt.clear_inactive_orders(asset_no)

            depth = hbt.depth(asset_no)
            if (
                not np.isfinite(depth.best_bid)
                or not np.isfinite(depth.best_ask)
                or depth.best_bid >= depth.best_ask
            ):
                continue
            timestamp = int(hbt.current_timestamp)
            if timestamp < start_ns or timestamp > end_ns:
                continue
            state = hbt.state_values(asset_no)
            position = float(state.position)
            balance = float(state.balance)
            idx = nearest_spot_index(spot_ts, timestamp)
            sigma_spot = realized_volatility_ewma(spot_ts, spot_px, timestamp)
            tau_seconds = max((end_ns - timestamp) / 1_000_000_000, 0.0)
            risk = compute_risk_scale(float(spot_px[idx]), strike, sigma_spot, tau_seconds)
            urgency = u_val(tau_seconds, position, cfg)
            quotes = compute_quotes(risk.fair_probability, position, gamma, risk.fair_probability, urgency, k, 0.01, 0.99)
            bid_tick = math.floor(quotes.bid / 0.01)
            ask_tick = math.ceil(quotes.ask / 0.01)
            bid_px = bid_tick * 0.01
            ask_px = ask_tick * 0.01

            buy_liquidity = ""
            sell_liquidity = ""
            step_taker_slippage_cost = 0.0
            buy_order_px = np.nan
            sell_order_px = np.nan

            if can_submit_buy(position, max_position, tau_seconds, cfg):
                buy_plan = plan_buy_order(bid_px, float(depth.best_ask), tick_size, order_qty)
                buy_tif = GTC if buy_plan.liquidity == "taker" else GTX
                hbt.submit_buy_order(asset_no, order_id_seq, buy_plan.price, order_qty, buy_tif, LIMIT, False)
                order_id_seq += 1
                buy_liquidity = buy_plan.liquidity
                buy_order_px = buy_plan.price
                step_taker_slippage_cost += buy_plan.taker_slippage_cost

            if can_submit_sell(position, max_position, tau_seconds, cfg):
                sell_plan = plan_sell_order(ask_px, float(depth.best_bid), tick_size, order_qty)
                sell_tif = GTC if sell_plan.liquidity == "taker" else GTX
                hbt.submit_sell_order(asset_no, order_id_seq, sell_plan.price, order_qty, sell_tif, LIMIT, False)
                order_id_seq += 1
                sell_liquidity = sell_plan.liquidity
                sell_order_px = sell_plan.price
                step_taker_slippage_cost += sell_plan.taker_slippage_cost

            cumulative_taker_slippage_cost += step_taker_slippage_cost
            current_settlement = settle_remaining_inventory(strike, final_spot, position, balance)
            adjusted_final_pnl = adjust_final_pnl(
                current_settlement.final_pnl,
                cumulative_taker_slippage_cost,
                fee_state.cumulative_taker_fee,
                fee_state.cumulative_maker_rebate,
            )
            rows.append(
                {
                    "row_type": "quote",
                    "timestamp": timestamp,
                    "pm_mid": (float(depth.best_bid) + float(depth.best_ask)) / 2.0,
                    "btcusd_price": float(spot_px[idx]),
                    "strike": strike,
                    "final_spot": final_spot,
                    "tau_seconds": tau_seconds,
                    "sigma_spot": sigma_spot,
                    "fair_probability": risk.fair_probability,
                    "risk_scale": risk.risk_scale,
                    "u_value": urgency,
                    "reservation_price": quotes.reservation_price,
                    "half_spread": quotes.half_spread,
                    "bid_quote": bid_px,
                    "ask_quote": ask_px,
                    "buy_liquidity": buy_liquidity,
                    "sell_liquidity": sell_liquidity,
                    "buy_order_px": buy_order_px,
                    "sell_order_px": sell_order_px,
                    "step_taker_slippage_cost": step_taker_slippage_cost,
                    "cumulative_taker_slippage_cost": cumulative_taker_slippage_cost,
                    "fee_rate": FEE_RATE,
                    "cumulative_maker_rebate": fee_state.cumulative_maker_rebate,
                    "cumulative_taker_fee": fee_state.cumulative_taker_fee,
                    "net_fee_equivalent": fee_state.net_fee_equivalent,
                    "position": position,
                    "balance": balance,
                    "fee": float(state.fee),
                    "num_trades": int(state.num_trades),
                    "trading_volume": float(state.trading_volume),
                    "trading_value": float(state.trading_value),
                    "settlement_price": current_settlement.settlement_price,
                    "inventory_value": current_settlement.inventory_value,
                    "final_pnl": current_settlement.final_pnl,
                    "adjusted_final_pnl": adjusted_final_pnl,
                }
            )
        final_state = hbt.state_values(asset_no)
        final_position = float(final_state.position)
        final_balance = float(final_state.balance)
        final_settlement = settle_remaining_inventory(strike, final_spot, final_position, final_balance)
        adjusted_final_pnl = adjust_final_pnl(
            final_settlement.final_pnl,
            cumulative_taker_slippage_cost,
            fee_state.cumulative_taker_fee,
            fee_state.cumulative_maker_rebate,
        )
        rows.append(
            {
                "row_type": "settlement",
                "timestamp": end_ns,
                "pm_mid": np.nan,
                "btcusd_price": final_spot,
                "strike": strike,
                "final_spot": final_spot,
                "tau_seconds": 0.0,
                "sigma_spot": np.nan,
                "fair_probability": settlement_price,
                "risk_scale": settlement_price * (1.0 - settlement_price),
                "u_value": np.nan,
                "reservation_price": np.nan,
                "half_spread": np.nan,
                "bid_quote": np.nan,
                "ask_quote": np.nan,
                "buy_liquidity": "",
                "sell_liquidity": "",
                "buy_order_px": np.nan,
                "sell_order_px": np.nan,
                "step_taker_slippage_cost": 0.0,
                "cumulative_taker_slippage_cost": cumulative_taker_slippage_cost,
                "fee_rate": FEE_RATE,
                "cumulative_maker_rebate": fee_state.cumulative_maker_rebate,
                "cumulative_taker_fee": fee_state.cumulative_taker_fee,
                "net_fee_equivalent": fee_state.net_fee_equivalent,
                "position": final_position,
                "balance": final_balance,
                "fee": float(final_state.fee),
                "num_trades": int(final_state.num_trades),
                "trading_volume": float(final_state.trading_volume),
                "trading_value": float(final_state.trading_value),
                "settlement_price": final_settlement.settlement_price,
                "inventory_value": final_settlement.inventory_value,
                "final_pnl": final_settlement.final_pnl,
                "adjusted_final_pnl": adjusted_final_pnl,
            }
        )
    finally:
        hbt.close()

    write_rows(output_csv, rows)
    return rows


def plot_rows(path: Path, rows: list[dict[str, float | int]]) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.array([float(row["timestamp_ns"]) for row in rows], dtype=np.float64)
    x = (x - x[0]) / 1_000_000_000
    probability = np.array([float(row["fair_probability"]) for row in rows], dtype=np.float64)
    risk_scale = np.array([float(row["risk_scale"]) for row in rows], dtype=np.float64)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(x, probability)
    axes[0].set_ylabel("fair probability")
    axes[1].plot(x, risk_scale)
    axes[1].set_ylabel("risk_scale")
    axes[1].set_xlabel("seconds from market start")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def market_start_ns(market: str) -> int:
    return int(market.removeprefix("btc_up_15m_")) * 1_000_000_000


def market_end_ns(market: str) -> int:
    return market_start_ns(market) + 900 * 1_000_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC UP 15m AS strategy skeleton focused on risk_scale")
    parser.add_argument("--market", default="btc_up_15m_1780191900")
    parser.add_argument(
        "--spot",
        type=Path,
        default=Path("data/binance/spot/daily/aggTrades/BTCUSDT/converted/BTCUSDT-aggTrades-2026-04-01_2026-05-31.npz"),
    )
    parser.add_argument("--start-seconds-before-end", type=int, default=900)
    parser.add_argument("--interval-seconds", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("data/btc_up_15_analysis/as_strategy_risk_scale.csv"))
    parser.add_argument("--plot", type=Path, default=Path("data/btc_up_15_analysis/as_strategy_risk_scale.png"))
    parser.add_argument("--k", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--run-backtest", action="store_true")
    parser.add_argument("--market-archive", type=Path, default=Path("data/btc_up_15m.tar.gz"))
    parser.add_argument("--btcusd", type=Path, default=Path("data/btcusd.tar.gz"))
    parser.add_argument("--market-npz", type=Path)
    parser.add_argument("--order-qty", type=float, default=1.0)
    parser.add_argument("--max-position", type=float, default=10.0)
    args = parser.parse_args()

    if args.run_backtest:
        market_npz = args.market_npz or Path("data/btc_up_15m_analysis") / f"{normalize_market_name(args.market)}.npz"
        if not market_npz.exists():
            convert_market_archive_to_npz(args.market_archive, args.market, market_npz)
        rows = run_hftbacktest_market(
            market_npz,
            args.btcusd,
            args.market,
            args.output,
            k=args.k,
            gamma=args.gamma,
            order_qty=args.order_qty,
            max_position=args.max_position,
        )
        print(f"market: {args.market}")
        print(f"k: {args.k}")
        print(f"gamma: {args.gamma}")
        print(f"rows: {len(rows)}")
        print(f"output: {args.output}")
        return

    spot_ts, spot_px = load_spot_series(args.spot)
    end_ns = market_end_ns(args.market)
    start_ns = end_ns - args.start_seconds_before_end * 1_000_000_000
    interval_ns = args.interval_seconds * 1_000_000_000

    rows = collect_risk_scale_path(spot_ts, spot_px, start_ns, end_ns, interval_ns)
    write_rows(args.output, rows)
    plot_rows(args.plot, rows)
    print(f"market: {args.market}")
    print(f"k: {args.k}")
    print(f"gamma: {args.gamma}")
    print(f"rows: {len(rows)}")
    print(f"output: {args.output}")
    print(f"plot: {args.plot}")


if __name__ == "__main__":
    main()
