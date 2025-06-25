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
        self.position_window = deque(maxlen=self.window_size)

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

        # Soft/hard liquidation conditions
        soft_liquidate = (
            len(self.position_window) == self.window_size 
            and sum(self.position_window) >= self.window_size / 2
            and self.position_window[-1]
        )
        hard_liquidate = (
            len(self.position_window) == self.window_size 
            and all(self.position_window)
        )

        # Adjust prices based on current true value and position.
        if position > self.limit * 0.5:
            max_buy_price = true_value - 1
        else:
            max_buy_price = true_value

        if position < -self.limit * 0.5:
            min_sell_price = true_value + 1
        else:
            min_sell_price = true_value

        # 1) Process SELL orders (buying from the market)
        for price, volume in sell_orders:
            if to_buy > 0 and price <= max_buy_price:
                quantity = min(to_buy, -volume)
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

        # Use a "popular" market buy price if orders remain
        if to_buy > 0 and buy_orders:
            popular_buy_price = max(buy_orders, key=lambda tup: tup[1])[0]
            price = min(max_buy_price, popular_buy_price + 1)
            orders.append(Order(self.symbol, price, to_buy))
            to_buy = 0

        # 3) Process BUY orders (selling to the market)
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
    
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.prev_price = None
        self.momentum_factor = 10  # increased from 5 to 10

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
        multiplier = 1.0
        peak_hour = 12

        if state.timestamp // 3600 == peak_hour:
            multiplier = 1.2

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
        multiplier = 1.0
        peak_hour = 12

        if state.timestamp // 3600 == peak_hour:
            multiplier = 1.2

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
        # Stable asset: adjust default to historical mean 10000.
        return 10000

    def act(self, state: TradingState) -> List[Order]:
        orders = super().act(state)
        peak_hour = 12
   
        if state.timestamp // 3600 == peak_hour:
   
            for order in orders:
                order.quantity = int(order.quantity * 1.25)
   
        price_impact = -0.2013437715581671
   
        if price_impact < 0:
   
            for order in orders:
                order.quantity = max(1, order.quantity // 2)
   
        return orders

class KelpStrategy(MarketMakingStrategy):
   
    def get_true_value(self, state: TradingState) -> int:
        order_depth: OrderDepth = state.order_depths[self.symbol]
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 2044
       
        buy_orders = sorted(order_depth.buy_orders.items(), key=lambda tup: tup[1], reverse=True)
        sell_orders = sorted(order_depth.sell_orders.items(), key=lambda tup: tup[1])
        popular_buy_price = buy_orders[0][0]
        popular_sell_price = sell_orders[0][0]
        fair_value = round((popular_buy_price + popular_sell_price) / 2)
        
        return fair_value
    
    def act(self, state: TradingState) -> List[Order]:
        orders = super().act(state)
        
        # Using a volatility threshold in line with KELP's low variability.
        volatility = 4  # adjusted from 12539.86
        
        if volatility > 0:
        
            for order in orders:
                order.quantity = max(1, order.quantity // 2)
        
        mean_reversion_factor = 15  # adjusted from 14485.32
        
        if mean_reversion_factor > 0:
        
            for order in orders:
                order.price = int(order.price * 0.95)
        
        return orders

class SquidInkStrategy(MarketMakingStrategy):
    
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.mid_price_window = deque(maxlen=4)  # kept at 5 for faster adaptation
    
    def get_true_value(self, state: TradingState) -> int:
        order_depth: OrderDepth = state.order_depths[self.symbol]
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
      
        if best_bid is None or best_ask is None:
            return 1827  # adjusted fallback for SQUID_INK
      
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
        self.mid_price_window = deque(data.get("mid_price_window", []), maxlen=5)

class SimpleMidPriceStrategy(MarketMakingStrategy):
    
    def get_true_value(self, state: TradingState) -> int:
        order_depth: OrderDepth = state.order_depths[self.symbol]
        
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return 100
      
        return round((max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2)

class CroissantsStrategy(MarketMakingStrategy):
  
    def get_true_value(self, state: TradingState) -> int:
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
        croissant_ratio = 13.773725171332327  # static value remains for now
        
        if croissant_ratio > 10:
        
            # Adjusting multiplier from 1.05 to 1.01 for tighter order prices
            for order in orders:
                order.price = int(order.price * 1.01)
        
        return orders

class JamsStrategy(MarketMakingStrategy):

    def get_true_value(self, state: TradingState) -> int:
        observation = state.observations.conversionObservations.get(self.symbol)

        # If missing observation, use historical average ~6600.
        if observation:
            return max(50, int(observation.sugarPrice))
    
        return 6480

class DjembesStrategy(MarketMakingStrategy):
    
    def get_true_value(self, state: TradingState) -> int:
        observation = state.observations.conversionObservations.get(self.symbol)
       
        # Returning sum of fees if available, otherwise use historical mean ~13462.
        if observation:
            return int(observation.transportFees + observation.exportTariff + observation.importTariff)
       
        return 13400
    
    def act(self, state: TradingState) -> List[Order]:
        orders = super().act(state)
        
        # Adjusting threshold to match price scale (~13500)
        if orders:
    
            for order in orders:
    
                if order.price > 13500:
                    order.quantity = int(order.quantity * 0.7)
    
        return orders

class PicnicBasket1Strategy(MarketMakingStrategy):

    def get_true_value(self, state: TradingState) -> int:
        obs = state.observations.conversionObservations.get(self.symbol)

        if obs:
            return int(obs.sunlightIndex * 1.6)

        return 58000

class PicnicBasket2Strategy(MarketMakingStrategy):

    def get_true_value(self, state: TradingState) -> int:
        obs = state.observations.conversionObservations.get(self.symbol)

        if obs:
            return int(obs.sunlightIndex * 2.0)

        return 30000

class VolcanicRockStrategy(MarketMakingStrategy):
    
    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.price_history = []

    def get_true_value(self, state: TradingState) -> int:
        market = state.order_depths.get(self.symbol)
    
        if market:
            best_ask = min(market.sell_orders.keys(), default=10091)
            best_bid = max(market.buy_orders.keys(), default=10091)
            mid_price = (best_ask + best_bid) / 2
        else:
            mid_price = 10091  # adjusted fallback
    
        self.price_history.append(mid_price)
    
        if len(self.price_history) > 20:
            self.price_history.pop(0)
    
        return int(mid_price)
    
    def calculate_volatility(self) -> float:

        if len(self.price_history) < 5:
            return 10

        return np.std(self.price_history)

class VoucherStrategy(MarketMakingStrategy):

    def __init__(self, symbol: str, limit: int, strike_price: int):
        super().__init__(symbol, limit)
        self.price_history = []
        self.strike_price = strike_price
        
        # Updated fallback prices based on mean mids over 3 days
        fallback_prices = {
            "VOLCANIC_ROCK_VOUCHER_10000": 172,    # Day 3’s value
            "VOLCANIC_ROCK_VOUCHER_10250": 41,
            "VOLCANIC_ROCK_VOUCHER_10500": 2.5,
            "VOLCANIC_ROCK_VOUCHER_9500": 655,
            "VOLCANIC_ROCK_VOUCHER_9750": 420,
        }
        self.fallback_price = fallback_prices.get(self.symbol, 400)

    def get_true_value(self, state: TradingState) -> int:
        market = state.order_depths.get(self.symbol)

        if market:
            best_ask = min(market.sell_orders.keys(), default=self.fallback_price)
            best_bid = max(market.buy_orders.keys(), default=self.fallback_price)
            mid_price = (best_ask + best_bid) / 2
        else:
            mid_price = self.fallback_price
        self.price_history.append(mid_price)

        if len(self.price_history) > 20:
            self.price_history.pop(0)

        return int(mid_price)

class MagnificentMacaronsStrategy(MarketMakingStrategy):
    CSI = 65  # Critical Sunlight Index threshold

    def __init__(self, symbol: str, limit: int):
        super().__init__(symbol, limit)
        self.sunlight_history = deque(maxlen=10)

    def get_true_value(self, state: TradingState) -> int:
        obs = state.observations.conversionObservations.get(self.symbol)
        # Fallback defaults
        sunlight = obs.sunlightIndex if obs and hasattr(obs, 'sunlightIndex') else MagnificentMacaronsStrategy.CSI
        sugar = obs.sugarPrice if obs and hasattr(obs, 'sugarPrice') else 100
        shipping = obs.transportFees if obs and hasattr(obs, 'transportFees') else 50
        tariffs = 0
        if obs and hasattr(obs, 'exportTariff') and hasattr(obs, 'importTariff'):
            tariffs = obs.exportTariff + obs.importTariff

        # Track recent sunlight levels
        self.sunlight_history.append(sunlight)
        sustained_low = (
            len(self.sunlight_history) == self.sunlight_history.maxlen
            and all(s < MagnificentMacaronsStrategy.CSI for s in self.sunlight_history)
        )

        # Base value calculation
        base_value = sugar + shipping + tariffs

        if sustained_low:
            # Panic premium when sunlight remains low
            premium = (MagnificentMacaronsStrategy.CSI - np.mean(self.sunlight_history)) / MagnificentMacaronsStrategy.CSI
            true_val = base_value * (1 + premium * 1.3)
        else:
            # Normal fair trading around base value
            true_val = base_value

        return int(true_val)
    
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
            "VOLCANIC_ROCK_VOUCHER_10500": 200,
            "MAGNIFICENT_MACARONS": 75
        }
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
            "MAGNIFICENT_MACARONS": MagnificentMacaronsStrategy("MAGNIFICENT_MACARONS", limits["MAGNIFICENT_MACARONS"]),
        }
        self.conversion_limits: Dict[str, int] = {
            "MAGNIFICENT_MACARONS": 10
        }
    
    def run(self, state: TradingState) -> tuple[Dict[str, List[Order]], int, str]:
        # Loading previous state, including counterparty counts
        old_data = json.loads(state.traderData) if state.traderData else {}
        cp_counts = old_data.get('cp_counts', {})
        new_data = {}
        orders: Dict[str, List[Order]] = {}

        # Updating counterparty counts from own trades
        for symbol, trades in state.own_trades.items():
            
            for trade in trades:
                cp = getattr(trade, 'counter_party', None)
            
                if cp:
                    cp_counts.setdefault(symbol, {})
                    cp_counts[symbol][cp] = cp_counts[symbol].get(cp, 0) + 1

        # Generating orders per strategy and scale by counterparty diversity
        for symbol, strategy in self.strategies.items():
            
            if symbol in old_data:
                strategy.load_state(old_data[symbol])
            
            if symbol in state.order_depths:
                base_orders = strategy.act(state)
                # Scale quantities by unique counterparty factor
                unique_cps = len(cp_counts.get(symbol, {}))
                factor = 1 + min(unique_cps / 10, 0.2)
                
                for order in base_orders:
                    order.quantity = int(order.quantity * factor)
                
                orders[symbol] = base_orders
            
            new_data[symbol] = strategy.save_state()

        # Persisting counterparty counts
        new_data['cp_counts'] = cp_counts
        trader_data = json.dumps(new_data, separators=(',', ':'))
        conversions = 0
        
        return orders, conversions, trader_data