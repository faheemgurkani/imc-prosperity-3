import json
from collections import deque
from utils.datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

# -----------------------------
# Base Market Making Strategy
# -----------------------------
class MarketMakingStrategy:
    def __init__(self, symbol: str, limit: int):
        self.symbol = symbol          # Trading symbol (e.g. "RAINFOREST_RESIN" or "KELP")
        self.limit = limit            # Position limit for the product (50 in this round)
        # A sliding window to record if we were at the position limit recently.
        self.position_window = deque()
        self.window_size = 10         # Window size to decide liquidation

    def get_true_value(self, state: TradingState) -> int:
        """
        Abstract method to compute the fair or true value of the asset.
        For a stable asset, this can be constant.
        For a volatile asset, this should be estimated dynamically from order book data.
        """
        raise NotImplementedError("Please implement the get_true_value method for your strategy.")

    def act(self, state: TradingState) -> List[Order]:
        """
        Core logic to generate orders.
        Reads the current order book for the symbol,
        computes fair value, and then decides how many orders to place on each side.
        """
        orders = []
        true_value = self.get_true_value(state)
        
        # Get the order book for this product
        order_depth: OrderDepth = state.order_depths[self.symbol]
        # Sort buy orders (descending by price) and sell orders (ascending by price)
        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        
        # Get current position for the product; if not available, assume zero.
        position = state.position.get(self.symbol, 0)
        # Calculate available capacity for buying or selling.
        # Note: a buy order will increase our position and a sell order will decrease.
        to_buy = self.limit - position       # maximum additional quantity we can buy without exceeding limit
        to_sell = self.limit + position      # maximum additional quantity we can sell (i.e. reduce long position or add to short)

        # Update sliding window: record if we are exactly at the limit (either side)
        self.position_window.append(abs(position) >= self.limit)
        if len(self.position_window) > self.window_size:
            self.position_window.popleft()
        # Liquidation conditions:
        soft_liquidate = len(self.position_window) == self.window_size and sum(self.position_window) >= self.window_size / 2 and self.position_window[-1]
        hard_liquidate = len(self.position_window) == self.window_size and all(self.position_window)

        # Define our aggressive prices based on our current true value and position
        # If we are nearing long capacity, we adjust our buy orders to be more aggressive (lower price)
        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        # Likewise, if nearing a short position limit, we adjust our sell orders upward
        min_sell_price = true_value + 1 if position < -self.limit * 0.5 else true_value

        # --- Place BUY orders (to increase position) ---
        # Try to match with existing sell orders first
        for price, volume in sell_orders:
            # Remember: sell order volumes are negative.
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                orders.append(Order(self.symbol, price, quantity))  # positive quantity -> buy order
                to_buy -= quantity

        # If we still have capacity to buy and our window indicates extreme conditions, liquidate part of a short position.
        if to_buy > 0:
            if hard_liquidate:
                quantity = to_buy // 2
                orders.append(Order(self.symbol, true_value, quantity))
                to_buy -= quantity
            elif soft_liquidate:
                quantity = to_buy // 2
                orders.append(Order(self.symbol, true_value - 2, quantity))
                to_buy -= quantity

        # If capacity remains, place an order based on popular buy price
        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            # Adjust our buy price: do not exceed our max_buy_price
            price = min(max_buy_price, popular_buy_price + 1)
            orders.append(Order(self.symbol, price, to_buy))
            to_buy = 0

        # --- Place SELL orders (to decrease position) ---
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                orders.append(Order(self.symbol, price, -quantity))  # negative quantity -> sell order
                to_sell -= quantity

        if to_sell > 0:
            if hard_liquidate:
                quantity = to_sell // 2
                orders.append(Order(self.symbol, true_value, -quantity))
                to_sell -= quantity
            elif soft_liquidate:
                quantity = to_sell // 2
                orders.append(Order(self.symbol, true_value + 2, -quantity))
                to_sell -= quantity

        if to_sell > 0 and sell_orders:
            popular_sell_price = min(sell_orders, key=lambda tup: tup[1])[0]
            price = max(min_sell_price, popular_sell_price - 1)
            orders.append(Order(self.symbol, price, -to_sell))
            to_sell = 0

        return orders

    def save_state(self) -> list:
        """
        Save our sliding window state for persistence between iterations.
        """
        return list(self.position_window)

    def load_state(self, data: list) -> None:
        self.position_window = deque(data, maxlen=self.window_size)


# -----------------------------
# Specific Strategy Implementations
# -----------------------------

class RainforestResinStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        """
        For Rainforest Resin, the asset is stable.
        We assume a constant fair value. Here we choose 100 (this can be calibrated).
        """
        return 100

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        """
        For Kelp, the asset is volatile so we compute a dynamic fair value.
        We use the order book data: take the highest-volume buy order and lowest-volume sell order,
        then compute the mid-price as an approximation.
        """
        order_depth: OrderDepth = state.order_depths[self.symbol]
        # If no orders are available, default to a reasonable value.
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 50

        # Sort orders on each side by volume
        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda tup: tup[1], reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda tup: tup[1])
        # Popular prices are taken as the price at which the maximum volume is offered.
        popular_buy_price = buy_orders[0][0]
        popular_sell_price = sell_orders[0][0]
        # The mid-price is our fair value estimate.
        fair_value = round((popular_buy_price + popular_sell_price) / 2)
        return fair_value


# -----------------------------
# Trader Class
# -----------------------------
class Trader:
    def __init__(self) -> None:
        # Define the position limits for the tutorial round.
        limits: Dict[str, int] = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
        }
        # Associate each product with its specific strategy.
        self.strategies: Dict[str, MarketMakingStrategy] = {
            "RAINFOREST_RESIN": RainforestResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", limits["KELP"]),
        }

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        conversions = 0  # No conversion logic implemented in this round.
        # Load persisted state (if any) from traderData
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders: Dict[str, List[Order]] = {}
        # Process each product that we have a strategy for.
        for symbol, strategy in self.strategies.items():
            # Restore state if available.
            if symbol in old_trader_data:
                strategy.load_state(old_trader_data[symbol])
            # If there is market data for this symbol, generate orders.
            if symbol in state.order_depths:
                orders[symbol] = strategy.act(state)
            # Save current state for next iteration.
            new_trader_data[symbol] = strategy.save_state()

        # Persist the state as a JSON string.
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        return orders, conversions, trader_data
