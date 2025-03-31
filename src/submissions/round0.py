import json
from collections import deque
from datamodel import OrderDepth, TradingState, Order   # The import is in accordance with the submission criteria
from typing import List, Dict

# -----------------------------
# Base Market Making Strategy
# -----------------------------
class MarketMakingStrategy:
    def __init__(self, symbol: str, limit: int):
        self.symbol = symbol          # Trading symbol (e.g. "RAINFOREST_RESIN" or "KELP")
        self.limit = limit            # Position limit for the product (50 in this round)
        # A sliding window to record if our absolute position has reached the limit in recent iterations.
        self.position_window = deque()
        self.window_size = 10         # Window size to decide liquidation conditions

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
        Reads the current order book for the symbol, computes fair value, and then decides how many orders to place on each side.
        """
        orders: List[Order] = []
        true_value = self.get_true_value(state)
        
        # Retrieve the order book for this product
        order_depth: OrderDepth = state.order_depths[self.symbol]
        # Sort buy orders descending by price and sell orders ascending by price.
        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        
        # Retrieve current position; if not set, assume zero.
        position = state.position.get(self.symbol, 0)
        # Compute how many units we can buy or sell without breaching the position limit.
        to_buy = self.limit - position       # additional units we can buy (increasing our long position)
        to_sell = self.limit + position      # additional units we can sell (increasing our short position)

        # Update sliding window state to record if we've reached our limit.
        self.position_window.append(abs(position) >= self.limit)
        if len(self.position_window) > self.window_size:
            self.position_window.popleft()
        # Determine liquidation conditions:
        soft_liquidate = (len(self.position_window) == self.window_size and 
                          sum(self.position_window) >= self.window_size / 2 and 
                          self.position_window[-1])
        hard_liquidate = (len(self.position_window) == self.window_size and 
                          all(self.position_window))

        # Adjust prices based on current true value and position.
        # If nearing a long limit, we lower our buy orders (more aggressive).
        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        # If nearing a short limit, we raise our sell orders.
        min_sell_price = true_value + 1 if position < -self.limit * 0.5 else true_value

        # --- Place BUY orders (to increase our position) ---
        # Try matching existing sell orders first.
        for price, volume in sell_orders:
            # Sell order volumes are negative.
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
                orders.append(Order(self.symbol, price, quantity))  # positive quantity indicates a buy order
                to_buy -= quantity

        # If capacity remains and liquidation conditions are met, unwind part of a short position.
        if to_buy > 0:
            if hard_liquidate:
                quantity = to_buy // 2
                orders.append(Order(self.symbol, true_value, quantity))
                to_buy -= quantity
            elif soft_liquidate:
                quantity = to_buy // 2
                orders.append(Order(self.symbol, true_value - 2, quantity))
                to_buy -= quantity

        # If buying capacity still remains, use popular buy prices.
        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            orders.append(Order(self.symbol, price, to_buy))
            to_buy = 0

        # --- Place SELL orders (to decrease our position) ---
        for price, volume in buy_orders:
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                orders.append(Order(self.symbol, price, -quantity))  # negative quantity indicates a sell order
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
        Persist our sliding window state for future iterations.
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
        For Rainforest Resin, we assume a stable asset with a constant fair value.
        Here we set the value at 100 (this can be tuned based on historical data).
        """
        return 100

class KelpStrategy(MarketMakingStrategy):
    def get_true_value(self, state: TradingState) -> int:
        """
        For Kelp, a volatile asset, we compute a dynamic fair value.
        We use the order book data:
         - Identify the price with the highest buy volume.
         - Identify the price with the highest sell volume.
         - Estimate the fair value as the mid-price between these two.
        If order data is missing, default to 50.
        """
        order_depth: OrderDepth = state.order_depths[self.symbol]
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 50

        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda tup: tup[1], reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda tup: tup[1])
        popular_buy_price = buy_orders[0][0]
        popular_sell_price = sell_orders[0][0]
        fair_value = round((popular_buy_price + popular_sell_price) / 2)
        return fair_value


# -----------------------------
# Trader Class
# -----------------------------
class Trader:
    def __init__(self) -> None:
        # Define position limits per product for the tutorial round.
        limits: Dict[str, int] = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
        }
        # Instantiate specific strategies for each product.
        self.strategies: Dict[str, MarketMakingStrategy] = {
            "RAINFOREST_RESIN": RainforestResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", limits["KELP"]),
        }

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        conversions = 0  # No conversion logic is implemented in round 0.
        # Load any persisted state from previous iterations.
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders: Dict[str, List[Order]] = {}
        # Process each product for which we have a strategy.
        for symbol, strategy in self.strategies.items():
            if symbol in old_trader_data:
                strategy.load_state(old_trader_data[symbol])
            if symbol in state.order_depths:
                orders[symbol] = strategy.act(state)
            new_trader_data[symbol] = strategy.save_state()

        # Persist the state for the next iteration.
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        return orders, conversions, trader_data
