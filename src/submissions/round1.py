import json
from collections import deque
from datamodel import OrderDepth, TradingState, Order   # The import is in accordance with the submission criteria
from typing import List, Dict



# -----------------------------
# Base Market Making Strategy
# -----------------------------
class MarketMakingStrategy:

    def _init_(self, symbol: str, limit: int):
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
        
        # Retrieving the order book for this product
        order_depth: OrderDepth = state.order_depths[self.symbol]
        
        # Sortting buy orders descending by price and sell orders ascending by price.
        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        
        # Retrieving current position; if not set, assume zero.
        position = state.position.get(self.symbol, 0)
        
        # Computing how many units we can buy or sell without breaching the position limit.
        to_buy = self.limit - position       # additional units we can buy (increasing our long position)
        to_sell = self.limit + position      # additional units we can sell (increasing our short position)

        # Updating sliding window state to record if we've reached our limit.
        self.position_window.append(abs(position) >= self.limit)
        
        if len(self.position_window) > self.window_size:
            self.position_window.popleft()
        
        # Determining liquidation conditions:
        soft_liquidate = (len(self.position_window) == self.window_size and 
                          sum(self.position_window) >= self.window_size / 2 and 
                          self.position_window[-1])
        
        hard_liquidate = (len(self.position_window) == self.window_size and 
                          all(self.position_window))

        # Adjusting prices based on current true value and position.
        # If nearing a long limit, we lower our buy orders (more aggressive).
        max_buy_price = true_value - 1 if position > self.limit * 0.5 else true_value
        
        # If nearing a short limit, we raise our sell orders.
        min_sell_price = true_value + 1 if position < -self.limit * 0.5 else true_value

        # Placing BUY orders (to increase our position)
        # Trying matching existing sell orders first.
        for price, volume in sell_orders:
            
            # Selling order volumes are negative.
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

        # Placing SELL orders (to decrease our position) 
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

class SquidInkStrategy(MarketMakingStrategy):

    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.mid_price_window = deque(maxlen=10)

    def get_true_value(self, state: TradingState) -> int:
        """
        For Squid Ink, we compute a dynamic fair value based on short-term mean reversion.
        - Calculate the current mid price from best bid and ask.
        - Maintain a sliding window of recent mid prices.
        - Return the average of the window as the fair value.
        """
        order_depth: OrderDepth = state.order_depths[self.symbol]
        
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        
        if best_bid is None or best_ask is None:
            return 100  # fallback value

        mid_price = (best_bid + best_ask) / 2
        self.mid_price_window.append(mid_price)

        if len(self.mid_price_window) < self.mid_price_window.maxlen:
            return round(mid_price)

        average_mid_price = sum(self.mid_price_window) / len(self.mid_price_window)
        
        return round(average_mid_price)

    def save_state(self) -> list:
        """
        Persist both position window and mid-price window.
        """
        return {
            "position_window": list(self.position_window),
            "mid_price_window": list(self.mid_price_window),
        }

    def load_state(self, data: dict) -> None:
        self.position_window = deque(data.get("position_window", []), maxlen=self.window_size)
        self.mid_price_window = deque(data.get("mid_price_window", []), maxlen=10)

# -----------------------------
# Trader Class
# -----------------------------
class Trader:
    
    def __init__(self) -> None:
        # Defining position limits per product for the tutorial round.
        limits: Dict[str, int] = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
        }
        
        # Instantiating specific strategies for each product.
        self.strategies: Dict[str, MarketMakingStrategy] = {
            "RAINFOREST_RESIN": RainforestResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", limits["KELP"]),
            "SQUID_INK": SquidInkStrategy("SQUID_INK", limits["SQUID_INK"]),
        }

    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        conversions = 0  # No conversion logic is implemented in round 0.
       
        # Loading any persisted state from previous iterations.
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}

        orders: Dict[str, List[Order]] = {}
        
        # Processing each product for which we have a strategy.
        for symbol, strategy in self.strategies.items():
        
            if symbol in old_trader_data:
                strategy.load_state(old_trader_data[symbol])
        
            if symbol in state.order_depths:
                orders[symbol] = strategy.act(state)
        
            new_trader_data[symbol] = strategy.save_state()

        # Persisting the state for the next iteration.
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        
        return orders, conversions, trader_data
