import json
from collections import deque
from datamodel import OrderDepth, TradingState, Order   # In accordance with the submission criteria
from typing import List, Dict
import numpy as np



# -----------------------------
# Base Market Making Strategy
# -----------------------------
class MarketMakingStrategy:
    
    def __init__(self, symbol: str, limit: int):
        self.symbol = symbol
        self.limit = limit
        
        # Adjusted window_size from 10 -> 5 for faster liquidation triggers
        self.window_size = 5
        self.position_window = deque()

    def get_true_value(self, state: TradingState) -> int:
        raise NotImplementedError("Please implement the get_true_value method for your strategy.")

    def act(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        true_value = self.get_true_value(state)
        
        order_depth: OrderDepth = state.order_depths[self.symbol]

        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda x: x[0], reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])

        position = state.position.get(self.symbol, 0)
        to_buy = self.limit - position
        to_sell = self.limit + position

        # Liquidation window: track if we hit the limit
        self.position_window.append(abs(position) >= self.limit)
        
        if len(self.position_window) > self.window_size:
            self.position_window.popleft()

        # Soft/hard liquidation conditions
        # With smaller window_size, these triggers happen more quickly.
        soft_liquidate = (
            len(self.position_window) == self.window_size 
            and sum(self.position_window) >= self.window_size / 2
            and self.position_window[-1]
        )
        hard_liquidate = (
            len(self.position_window) == self.window_size 
            and all(self.position_window)
        )

        # Adjusting prices based on current true value and position.
        # If nearing a long limit, we lower our buy orders (more aggressive).
        if position > self.limit * 0.5:
            max_buy_price = true_value - 1
        else:
            max_buy_price = true_value

        # If nearing a short limit, we raise our sell orders.
        if position < -self.limit * 0.5:
            min_sell_price = true_value + 1
        else:
            min_sell_price = true_value

        # 1) Process SELL orders (i.e., buying from the market)
        for price, volume in sell_orders:
            
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)  # volume is negative for sell side
                orders.append(Order(self.symbol, price, quantity))
                to_buy -= quantity

        # 2) Adjust buy volume for liquidation
        if to_buy > 0:
            
            if hard_liquidate:
                quantity = to_buy // 2
                orders.append(Order(self.symbol, true_value, quantity))
                to_buy -= quantity
            elif soft_liquidate:
                quantity = to_buy // 2
                orders.append(Order(self.symbol, true_value - 2, quantity))
                to_buy -= quantity

        # If buy orders remain, use a "popular" market buy price
        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            orders.append(Order(self.symbol, price, to_buy))
            to_buy = 0

        # 3) Process BUY orders (i.e., selling to the market)
        for price, volume in buy_orders:
            
            if to_sell > 0 and price >= min_sell_price:
                quantity = min(to_sell, volume)
                orders.append(Order(self.symbol, price, -quantity))
                to_sell -= quantity

        # 4) Adjust sell volume for liquidation
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

        # --- Begin Added Voucher Multiplier in act() ---
        peak_hour = 12
        if state.timestamp // 3600 == peak_hour:
            for order in orders:
                order.quantity = int(order.quantity * 1.2)
        # --- End Added Voucher Multiplier in act() ---

        return orders

    def save_state(self) -> list:
        return list(self.position_window)

    def load_state(self, data: list) -> None:
        self.position_window = deque(data, maxlen=self.window_size)

# -----------------------------
# Generic Strategy Implementations
# -----------------------------
class MeanReversionStrategy(MarketMakingStrategy):
    """
    Shortened the memory window from 20 -> 10
    so it doesn't accumulate too much stale data
    and adjusts more quickly to trends.
    """
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.history = []

    def get_true_value(self, state: TradingState) -> int:
        market = state.order_depths.get(self.symbol)
   
        if market:
            best_ask = min(market.sell_orders.keys(), default=10500)
            best_bid = max(market.buy_orders.keys(), default=10400)
            mid = (best_ask + best_bid) / 2
        else:
            mid = 10450

        self.history.append(mid)
   
        if len(self.history) > 10:
            self.history.pop(0)

        return int(np.mean(self.history))

class MomentumStrategy(MarketMakingStrategy):
    """
    Increase the momentum push from ±5 -> ±10
    so that we follow the trend more strongly.
    """
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.prev_price = None
        self.momentum_factor = 10  # changed from 5 to 10

    def get_true_value(self, state: TradingState) -> int:
        market = state.order_depths.get(self.symbol)
   
        if market:
            best_ask = min(market.sell_orders.keys(), default=10500)
            best_bid = max(market.buy_orders.keys(), default=10400)
            current = (best_ask + best_bid) / 2
        else:
            current = 10450

        value = current
   
        if self.prev_price is not None:
            delta = current - self.prev_price
            value += self.momentum_factor * np.sign(delta)

        self.prev_price = current
   
        return int(value)

class VolatilitySpreadStrategy(MarketMakingStrategy):
   
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.history = []

    def get_true_value(self, state: TradingState) -> int:
        market = state.order_depths.get(self.symbol)
   
        if market:
            best_ask = min(market.sell_orders.keys(), default=10500)
            best_bid = max(market.buy_orders.keys(), default=10450)
            mid = (best_ask + best_bid) / 2
        else:
            mid = 10450

        self.history.append(mid)
   
        if len(self.history) > 30:
            self.history.pop(0)

        volatility = np.std(self.history) if len(self.history) > 5 else 10
   
        return int(mid + np.random.normal(0, volatility // 2))

class TimeDecayVoucherStrategy(MarketMakingStrategy):
   
    def __init__(self, symbol: str, limit: int, strike_price: int):
        super().__init__(symbol, limit)
        self.strike_price = strike_price

    def get_true_value(self, state: TradingState) -> int:
        rock = state.order_depths.get("VOLCANIC_ROCK")
   
        if rock:
            ask = min(rock.sell_orders.keys(), default=10500)
            bid = max(rock.buy_orders.keys(), default=10450)
            price = (ask + bid) / 2
        else:
            price = 10450

        intrinsic = max(0, price - self.strike_price)
        day = state.timestamp // 1000000
        decay = max(0.1, (7 - day) / 7)
        time_value = decay * 80
        true_val = intrinsic + time_value
   
        # --- Begin Added Timestamp Multiplier for Vouchers ---
        multiplier = 1.0
        peak_hour = 12
        if state.timestamp // 3600 == peak_hour:
            multiplier = 1.2
        # --- End Added Timestamp Multiplier for Vouchers ---
   
        return int(true_val * multiplier)

class ArbitrageVoucherStrategy(MarketMakingStrategy):
   
    def __init__(self, symbol: str, limit: int, strike_price: int):
        super().__init__(symbol, limit)
        self.strike_price = strike_price

    def get_rock_price(self, state: TradingState):
        rock = state.order_depths.get("VOLCANIC_ROCK")
   
        if rock:
            ask = min(rock.sell_orders.keys(), default=10500)
            bid = max(rock.buy_orders.keys(), default=10450)
   
            return (ask + bid) / 2
   
        return 10450

    def get_true_value(self, state: TradingState) -> int:
        rock_price = self.get_rock_price(state)
        intrinsic = max(0, rock_price - self.strike_price)
        hedge_buffer = 10
        true_val = intrinsic + hedge_buffer
   
        # --- Begin Added Timestamp Multiplier for Vouchers ---
        multiplier = 1.0
        peak_hour = 12
        if state.timestamp // 3600 == peak_hour:
            multiplier = 1.2
        # --- End Added Timestamp Multiplier for Vouchers ---
   
        return int(true_val * multiplier)

class ReversalStrategy(MarketMakingStrategy):
   
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.history = []

    def get_true_value(self, state: TradingState) -> int:
        market = state.order_depths.get(self.symbol)
   
        if market:
            best_ask = min(market.sell_orders.keys(), default=10500)
            best_bid = max(market.buy_orders.keys(), default=10450)
            mid = (best_ask + best_bid) / 2
        else:
            mid = 10450

        self.history.append(mid)
   
        if len(self.history) > 10:
            self.history.pop(0)

        if len(self.history) >= 5 and (self.history[-1] < self.history[-5] - 50):
            # Price crashed: expect rebound
            return int(mid + 30)
        
        return int(mid)

    def save_state(self) -> list:
        return {
            "position_window": list(self.position_window),
            "history": self.history
        }

    def load_state(self, data: dict) -> None:
        self.position_window = deque(data.get("position_window", []), maxlen=self.window_size)
        self.history = data.get("history", [])

# -----------------------------
# Product Based Strategies
# -----------------------------
class RainforestResinStrategy(MarketMakingStrategy):
   
    def get_true_value(self, state: TradingState) -> int:
        # Assuming stable asset with fair value close to 100
        return 100

    def act(self, state: TradingState) -> List[Order]:
        # Base logic
        orders = super().act(state)
        
        # Adjusting orders based on the peak trading hour and price impact
        peak_hour = 12
        
        if state.timestamp // 3600 == peak_hour:  # If current hour matches peak hour
            # Increasing trade sizes slightly during the peak hour (or adjust order size dynamically)
            for order in orders:
                order.quantity = int(order.quantity * 1.2)  # Increase size by 20%
        
        # If price impact is negative, we place smaller orders
        price_impact = -0.2013437715581671  # Example static value
    
        if price_impact < 0:
    
            for order in orders:
                order.quantity = max(1, order.quantity // 2)  # Reduce size to limit price impact
        
        return orders

class KelpStrategy(MarketMakingStrategy):
   
    def get_true_value(self, state: TradingState) -> int:
        # Dynamic fair value based on order book imbalance
        order_depth: OrderDepth = state.order_depths[self.symbol]
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 50

        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda tup: tup[1], reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda tup: tup[1])
        popular_buy_price = buy_orders[0][0]
        popular_sell_price = sell_orders[0][0]
        fair_value = round((popular_buy_price + popular_sell_price) / 2)

        return fair_value
    
    def act(self, state: TradingState) -> List[Order]:
        orders = super().act(state)

        # Using average intraday volatility to adjust position sizes
        volatility = 12539.86  # Example static value
        
        if volatility > 10000:
          
            for order in orders:
                order.quantity = max(1, order.quantity // 2)  # Reduce position size if volatility is high
     
        # Using mean-reversion factor to tweak pricing strategy if necessary
        mean_reversion_factor = 14485.321156250004
        
        if mean_reversion_factor > 10000:
            # Adjusting the strategy to push the price toward the mean
            for order in orders:
                order.price = int(order.price * 0.95)  # Push price closer to the mean
        
        return orders
    
class SquidInkStrategy(MarketMakingStrategy):
   
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.mid_price_window = deque(maxlen=5)
    
    def get_true_value(self, state: TradingState) -> int:
        order_depth: OrderDepth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
    
        if best_bid is None or best_ask is None:
            return 100
    
        mid_price = (best_bid + best_ask) / 2
        self.mid_price_window.append(mid_price)
    
        if len(self.mid_price_window) < self.mid_price_window.maxlen:
            return round(mid_price)
    
        average_mid_price = sum(self.mid_price_window) / len(self.mid_price_window)
   
        return round(average_mid_price)
    
    def save_state(self) -> dict:
        return {
            "position_window": list(self.position_window),
            "mid_price_window": list(self.mid_price_window),
        }
    
    def load_state(self, data: dict) -> None:
        self.position_window = deque(data.get("position_window", []), maxlen=self.window_size)
        self.mid_price_window = deque(data.get("mid_price_window", []), maxlen=10)

class SimpleMidPriceStrategy(MarketMakingStrategy):
   
    def get_true_value(self, state: TradingState) -> int:
        order_depth: OrderDepth = state.order_depths[self.symbol]
   
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 100
   
        return round((max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2)
    
class CroissantsStrategy(MarketMakingStrategy):
   
    def get_true_value(self, state: TradingState) -> int:
        # Simple fair value calculation
        order_depth: OrderDepth = state.order_depths[self.symbol]
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 100
        
        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda tup: tup[1], reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda tup: tup[1])
        
        popular_buy_price = buy_orders[0][0]
        popular_sell_price = sell_orders[0][0]
        
        return (popular_buy_price + popular_sell_price) // 2
    
    def act(self, state: TradingState) -> List[Order]:
        orders = super().act(state)

        # Using the croissant ratio to adjust aggressiveness
        croissant_ratio = 13.773725171332327  # Example static value
    
        if croissant_ratio > 10:
            # If the ratio is high, place more aggressive orders
            for order in orders:
                order.price = int(order.price * 1.05)  
        
        return orders

class JamsStrategy(MarketMakingStrategy):
    
    def get_true_value(self, state: TradingState) -> int:
        """
        For Jams, we adjust the true value using external market factors.
        We use the sugarPrice from the observations and adjust the fair value.
        """
        observation = state.observations.conversionObservations.get(self.symbol)
        
        if observation:
            # Adjusting true value based on sugar price
            return max(50, int(observation.sugarPrice)) 
        
        return 50

class DjembesStrategy(MarketMakingStrategy):
   
    def get_true_value(self, state: TradingState) -> int:
        observation = state.observations.conversionObservations.get(self.symbol)
   
        if observation:
            return int(observation.transportFees + observation.exportTariff + observation.importTariff)
   
        return 100
    
    def act(self, state: TradingState) -> List[Order]:
        orders = super().act(state)

        # Adjusting orders if transport fees or tariffs are high
        if orders:
            
            for order in orders:
              
                if order.price > 150:
                    order.quantity = int(order.quantity * 0.8)  # Reduce size if price is high
        
        return orders

class PicnicBasket1Strategy(MarketMakingStrategy):
   
    def get_true_value(self, state: TradingState) -> int:
        """
        For Picnic Basket 1, assume a base value adjusted by external factors like sunlight index.
        """
        observation = state.observations.conversionObservations.get(self.symbol)
        
        if observation:
            return int(observation.sunlightIndex * 1.5)  # Example adjustment with sunlight index
        
        return 75  # Default value

class PicnicBasket2Strategy(MarketMakingStrategy):
    
    def get_true_value(self, state: TradingState) -> int:
        """
        For Picnic Basket 2, assume a base value adjusted similarly to Picnic Basket 1.
        """
        observation = state.observations.conversionObservations.get(self.symbol)
        
        if observation:
            return int(observation.sunlightIndex * 2)  # Different adjustment with sunlight index
        
        return 80  # Default value

class VolcanicRockStrategy(MarketMakingStrategy):
    
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.price_history = []

    def get_true_value(self, state: TradingState) -> int:
        market = state.order_depths.get(self.symbol)
        
        if market:
            best_ask = min(market.sell_orders.keys(), default=400)
            best_bid = max(market.buy_orders.keys(), default=400)
            mid_price = (best_ask + best_bid) / 2
        else:
            mid_price = 400  # fallback

        # Storing price for volatility tracking
        self.price_history.append(mid_price)
        
        if len(self.price_history) > 20:
            self.price_history.pop(0)

        return int(mid_price)

    def calculate_volatility(self) -> float:
        
        if len(self.price_history) < 5:
            return 10  # arbitrary default
        
        return np.std(self.price_history)


class VoucherStrategy(MarketMakingStrategy):
    
    def __init__(self, symbol: str, limit: int, strike_price: int):
        super().__init__(symbol, limit)
        # super().__init__(symbol, limit)
        self.price_history = []
        self.strike_price = strike_price
        self.fallback_price = 400

    def get_true_value(self, state: TradingState) -> int:
        market = state.order_depths.get(self.symbol)
       
        if market:
            best_ask = min(market.sell_orders.keys(), default=100)
            best_bid = max(market.buy_orders.keys(), default=100)
            mid_price = (best_ask + best_bid) / 2
        else:
            mid_price = 400  # fallback

        # Storing price for volatility tracking
        self.price_history.append(mid_price)
        
        if len(self.price_history) > 20:
            self.price_history.pop(0)

        return int(mid_price)

# -----------------------------
# Trader Class
# -----------------------------
class Trader:
    
    def __init__(self) -> None:
        limits: Dict[str, int] = {
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50,
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "VOLCANIC_ROCK": 400,
            "VOLCANIC_ROCK_VOUCHER_9500": 200,
            "VOLCANIC_ROCK_VOUCHER_9750": 200,
            "VOLCANIC_ROCK_VOUCHER_10000": 200,
            "VOLCANIC_ROCK_VOUCHER_10250": 200,
            "VOLCANIC_ROCK_VOUCHER_10500": 200
        }
        # self.strategies: Dict[str, MarketMakingStrategy] = {
        #     # Non-voucher products
        #     "CROISSANTS": MeanReversionStrategy("CROISSANTS", limits["CROISSANTS"]),
        #     "DJEMBES": MomentumStrategy("DJEMBES", limits["DJEMBES"]),
        #     "JAMS": MeanReversionStrategy("JAMS", limits["JAMS"]),
        #     "KELP": MeanReversionStrategy("KELP", limits["KELP"]),
        #     "PICNIC_BASKET1": MomentumStrategy("PICNIC_BASKET1", limits["PICNIC_BASKET1"]),
        #     "PICNIC_BASKET2": MomentumStrategy("PICNIC_BASKET2", limits["PICNIC_BASKET2"]),
        #     "RAINFOREST_RESIN": MeanReversionStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
        #     "SQUID_INK": MeanReversionStrategy("SQUID_INK", limits["SQUID_INK"]),
        #     "VOLCANIC_ROCK": MeanReversionStrategy("VOLCANIC_ROCK", limits["VOLCANIC_ROCK"]),

        #     # Voucher products
        #     "VOLCANIC_ROCK_VOUCHER_10000": TimeDecayVoucherStrategy(
        #         "VOLCANIC_ROCK_VOUCHER_10000", 
        #         limits["VOLCANIC_ROCK_VOUCHER_10000"], 
        #         10000
        #     ),
        #     "VOLCANIC_ROCK_VOUCHER_10250": TimeDecayVoucherStrategy(
        #         "VOLCANIC_ROCK_VOUCHER_10250", 
        #         limits["VOLCANIC_ROCK_VOUCHER_10250"], 
        #         10250
        #     ),
        #     "VOLCANIC_ROCK_VOUCHER_10500": TimeDecayVoucherStrategy(
        #         "VOLCANIC_ROCK_VOUCHER_10500", 
        #         limits["VOLCANIC_ROCK_VOUCHER_10500"], 
        #         10500
        #     ),
        #     "VOLCANIC_ROCK_VOUCHER_9500": ArbitrageVoucherStrategy(
        #         "VOLCANIC_ROCK_VOUCHER_9500", 
        #         limits["VOLCANIC_ROCK_VOUCHER_9500"], 
        #         9500
        #     ),
        #     "VOLCANIC_ROCK_VOUCHER_9750": ArbitrageVoucherStrategy(
        #         "VOLCANIC_ROCK_VOUCHER_9750", 
        #         limits["VOLCANIC_ROCK_VOUCHER_9750"], 
        #         9750
        #     )
        # }
        self.strategies: Dict[str, MarketMakingStrategy] = {
            "RAINFOREST_RESIN": RainforestResinStrategy("RAINFOREST_RESIN", limits["RAINFOREST_RESIN"]),
            "KELP": KelpStrategy("KELP", limits["KELP"]),
            "SQUID_INK": SquidInkStrategy("SQUID_INK", limits["SQUID_INK"]),
            "CROISSANTS": CroissantsStrategy("CROISSANTS", limits["CROISSANTS"]),
            "JAMS": JamsStrategy("JAMS", limits["JAMS"]),
            "DJEMBES": DjembesStrategy("DJEMBES", limits["DJEMBES"]),
            "PICNIC_BASKET1": PicnicBasket1Strategy("PICNIC_BASKET1", limits["PICNIC_BASKET1"]),
            "PICNIC_BASKET2": PicnicBasket2Strategy("PICNIC_BASKET2", limits["PICNIC_BASKET2"]),
            "VOLCANIC_ROCK": VolcanicRockStrategy("VOLCANIC_ROCK", limits["VOLCANIC_ROCK"]),
            "VOLCANIC_ROCK_VOUCHER_9500": VoucherStrategy("VOLCANIC_ROCK_VOUCHER_9500", limits["VOLCANIC_ROCK_VOUCHER_9500"], 9500),
            "VOLCANIC_ROCK_VOUCHER_9750": VoucherStrategy("VOLCANIC_ROCK_VOUCHER_9750", limits["VOLCANIC_ROCK_VOUCHER_9750"], 9750),
            "VOLCANIC_ROCK_VOUCHER_10000": VoucherStrategy("VOLCANIC_ROCK_VOUCHER_10000", limits["VOLCANIC_ROCK_VOUCHER_10000"], 10000),
            "VOLCANIC_ROCK_VOUCHER_10250": VoucherStrategy("VOLCANIC_ROCK_VOUCHER_10250", limits["VOLCANIC_ROCK_VOUCHER_10250"], 10250),
            "VOLCANIC_ROCK_VOUCHER_10500": VoucherStrategy("VOLCANIC_ROCK_VOUCHER_10500", limits["VOLCANIC_ROCK_VOUCHER_10500"], 10500),
        }
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:  
        # Retrieving stored trader data from previous rounds (if any)
        old_trader_data = json.loads(state.traderData) if state.traderData != "" else {}
        new_trader_data = {}
        orders: Dict[str, List[Order]] = {}
        
        # Executing each strategy
        for symbol, strategy in self.strategies.items():
    
            if symbol in old_trader_data:
                strategy.load_state(old_trader_data[symbol])
            if symbol in state.order_depths:
                orders[symbol] = strategy.act(state)
            new_trader_data[symbol] = strategy.save_state()
        
        # Serializing new state data for the next run
        trader_data = json.dumps(new_trader_data, separators=(",", ":"))
        conversions = 0
        
        return orders, conversions, trader_data
