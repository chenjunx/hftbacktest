import numpy as np
from dataclasses import dataclass

from hftbacktest import BacktestAsset, HashMapMarketDepthBacktest, BUY, SELL, GTX, LIMIT, FILLED
from hftbacktest.order import PARTIALLY_FILLED

REFRESH_INTERVAL_NS = 100_000_000  # 100ms
ORDER_TIMEOUT_NS = 5_000_000_000


@dataclass
class QuoterConfig:
    sigma: float
    gamma: float
    kappa: float
    T: float
    eta: float = 0.0          # quadratic inventory penalty
    q_soft: float = 5.0       # soft inventory cap
    q_hard: float = 10.0      # hard inventory cap
    tick: float = 0.1         # min price increment
    min_size: float = 1.0     # min order size
    max_size: float = 10.0    # max order size per side


class ASQuoter:
    def __init__(self, cfg: QuoterConfig):
        self.cfg = cfg
        self.t = 0.0
        self.q = 0.0
        self.last_mid = None
        self.cash = 0.0

    def reservation(self, s):
        c = self.cfg
        tau = max(c.T - self.t, 1e-9)
        # AS reservation + quadratic inventory penalty
        r = s - self.q * (c.gamma * c.sigma**2 + 2.0 * c.eta) * tau
        return r, tau

    def half_spread(self, tau):
        c = self.cfg
        return 0.5 * (c.gamma * c.sigma**2 * tau
                      + (2.0 / c.gamma) * np.log(1.0 + c.gamma / c.kappa))

    def quote(self, mid: float):
        c = self.cfg
        r, tau = self.reservation(mid)
        hs = self.half_spread(tau)
        bid = r - hs
        ask = r + hs

        # tier 2: inventory bands -- widen same-side, tighten opposite-side
        aq = abs(self.q)
        if aq >= c.q_soft and aq < c.q_hard:
            band = (aq - c.q_soft) / (c.q_hard - c.q_soft)
            scale = np.exp(band)
            if self.q > 0:
                # long: discourage further buying, encourage selling
                bid -= hs * (scale - 1)
                ask -= hs * (scale - 1) * 0.5
            else:
                ask += hs * (scale - 1)
                bid += hs * (scale - 1) * 0.5

        # tier 3: hard cap -- one-sided only
        ban_bid = (self.q >= c.q_hard)
        ban_ask = (self.q <= -c.q_hard)

        # round to tick
        bid = np.floor(bid / c.tick) * c.tick
        ask = np.ceil(ask / c.tick) * c.tick

        # size scaling: shrink toward zero as inventory grows on the same side
        size_bid = c.max_size if not ban_bid else 0.0
        size_ask = c.max_size if not ban_ask else 0.0
        if self.q > 0:
            size_bid *= max(0.0, 1.0 - aq / c.q_hard)
        else:
            size_ask *= max(0.0, 1.0 - aq / c.q_hard)
        size_bid = max(size_bid, 0.0)
        size_ask = max(size_ask, 0.0)
        if size_bid > 0 and size_bid < c.min_size:
            size_bid = c.min_size
        if size_ask > 0 and size_ask < c.min_size:
            size_ask = c.min_size

        return {
            "bid_price": bid, "bid_size": size_bid,
            "ask_price": ask, "ask_size": size_ask,
            "reservation": r, "half_spread": hs,
        }

    def on_fill(self, side: str, price: float, size: float):
        if side == "bid":
            self.cash -= price * size
            self.q += size
        elif side == "ask":
            self.cash += price * size
            self.q -= size

    def step_time(self, dt: float):
        self.t += dt

    def mark_to_market(self, mid: float):
        return self.cash + self.q * mid


# Note: this loop is plain Python (no @numba.njit). ASQuoter relies on a
# Python dataclass, which numba's nopython mode does not support. njit is
# strongly recommended by hftbacktest for performance, so if this becomes a
# bottleneck, port ASQuoter to a numba jitclass and re-add @njit.
def as_market_making_strategy(hbt, cfg: QuoterConfig):
    asset_no = 0
    tick_size = hbt.depth(asset_no).tick_size

    quoter = ASQuoter(cfg)
    dt = REFRESH_INTERVAL_NS / 1_000_000_000.0
    # tracks exec_qty already applied to the quoter per order_id, so repeated
    # partial fills on the same order are only applied incrementally
    processed_exec_qty = {}

    while hbt.elapse(REFRESH_INTERVAL_NS) == 0:
        hbt.clear_inactive_orders(asset_no)
        quoter.step_time(dt)

        depth = hbt.depth(asset_no)
        mid_price = (depth.best_bid + depth.best_ask) / 2.0

        # ---------------------------------------------------------------
        # apply fills that happened since the last tick to the quoter's
        # inventory/cash state
        # ---------------------------------------------------------------
        order_values = hbt.orders(asset_no).values()
        while order_values.has_next():
            order = order_values.get()
            if order.status == FILLED or order.status == PARTIALLY_FILLED:
                prev_qty = processed_exec_qty.get(order.order_id, 0.0)
                delta_qty = order.exec_qty - prev_qty
                if delta_qty > 0:
                    side = 'bid' if order.side == BUY else 'ask'
                    quoter.on_fill(side, order.exec_price, delta_qty)
                    processed_exec_qty[order.order_id] = order.exec_qty

        # ---------------------------------------------------------------
        # requote from the Avellaneda-Stoikov quoter
        # ---------------------------------------------------------------
        q = quoter.quote(mid_price)

        # never cross the book with a GTX (post-only) order
        new_bid = min(q['bid_price'], depth.best_bid)
        new_ask = max(q['ask_price'], depth.best_ask)
        new_bid_tick = round(new_bid / tick_size)
        new_ask_tick = round(new_ask / tick_size)

        # ---------------------------------------------------------------
        # order management: cancel stale orders, then place new ones
        # ---------------------------------------------------------------
        update_bid = q['bid_size'] > 0
        update_ask = q['ask_size'] > 0
        last_order_id = -1

        order_values = hbt.orders(asset_no).values()
        while order_values.has_next():
            order = order_values.get()
            if order.side == BUY:
                if order.price_tick == new_bid_tick and update_bid:
                    update_bid = False
                elif order.cancellable:
                    hbt.cancel(asset_no, order.order_id, False)
                    last_order_id = order.order_id
            elif order.side == SELL:
                if order.price_tick == new_ask_tick and update_ask:
                    update_ask = False
                elif order.cancellable:
                    hbt.cancel(asset_no, order.order_id, False)
                    last_order_id = order.order_id

        if update_bid:
            order_id = new_bid_tick
            hbt.submit_buy_order(asset_no, order_id, new_bid_tick * tick_size, q['bid_size'], GTX, LIMIT, False)
            last_order_id = order_id
        if update_ask:
            order_id = new_ask_tick
            hbt.submit_sell_order(asset_no, order_id, new_ask_tick * tick_size, q['ask_size'], GTX, LIMIT, False)
            last_order_id = order_id

        if last_order_id >= 0:
            if not hbt.wait_order_response(asset_no, last_order_id, ORDER_TIMEOUT_NS):
                return False

    return True


if __name__ == '__main__':
    asset = (
        BacktestAsset()
            .data([
                'data/btcusdt_20220831.npz',
            ])
            .initial_snapshot('data/btcusdt_20220830_eod.npz')
            .linear_asset(1.0)
            .power_prob_queue_model(2.0)
            .no_partial_fill_exchange()
            .trading_value_fee_model(-0.00005, 0.0007)
            .tick_size(0.1)
            .lot_size(0.001)
    )
    hbt = HashMapMarketDepthBacktest([asset])

    cfg = QuoterConfig(
        sigma=0.0,  # TODO: plug in a volatility estimate
        gamma=0.05,
        kappa=1.5,  # TODO: calibrate order book liquidity/intensity
        T=1.0,      # TODO: trading horizon in seconds
        tick=0.1,
    )
    as_market_making_strategy(hbt, cfg)
