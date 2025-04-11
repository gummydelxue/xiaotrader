import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import datetime
import time
import logging
import os

# Demo account credentials (replace with your actual details)
DEMO_ACCOUNT = os.getenv("MT5_ACCOUNT_ID")  # Your demo account number
DEMO_PASSWORD = os.getenv("MT5_PASSWORD")  # Your demo account password
DEMO_SERVER = os.getenv("MT5_SERVER")  # Your demo server name (e.g., "ICMarkets-Demo")
# Set up logging
log_dir = "trading_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"trade_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Initialize MT5 connection
def initialize_mt5():
    if not mt5.initialize():
        logging.error("Failed to initialize MT5")
        quit()
    
    symbol_info = mt5.symbol_info(SYMBOL)
    if not symbol_info:
        logging.error(f"Symbol {SYMBOL} not available on server {DEMO_SERVER}")
        quit()
    logging.info(f"Symbol {SYMBOL}: Spread={symbol_info.spread}, Point={symbol_info.point}")

    current_account = mt5.account_info()
    if current_account is None or str(current_account.login) != DEMO_ACCOUNT:
        logging.info(f"Switching to demo account: {DEMO_ACCOUNT}")
        if not mt5.login(int(DEMO_ACCOUNT), password=DEMO_PASSWORD, server=DEMO_SERVER):
            logging.error(f"Failed to login to demo account {DEMO_ACCOUNT}: {mt5.last_error()}")
            quit()
    account_info = mt5.account_info()
    logging.info(f"Connected to account: {account_info.login}, Balance: {account_info.balance}, "
                 f"Equity: {account_info.equity}, Server: {account_info.server}")

# Strategy parameters
SYMBOL = "XAUUSD"
LOT_SIZE = 0.01
EMA_FAST = 50
EMA_SLOW = 200
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RISK_PIPS = 20
REWARD_PIPS = 40

# Fetch 1 year of data
def get_data(timeframe, days=365):
    try:
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(days=days)
        rates = mt5.copy_rates_range(SYMBOL, timeframe, start_time, end_time)
        if rates is None or len(rates) == 0:
            logging.warning(f"No data retrieved for {SYMBOL} on timeframe {timeframe} for {days} days")
            start_time = end_time - datetime.timedelta(days=90)
            rates = mt5.copy_rates_range(SYMBOL, timeframe, start_time, end_time)
            if rates is None or len(rates) == 0:
                logging.error(f"Fallback failed: No data for {SYMBOL} on timeframe {timeframe}")
                return pd.DataFrame()
            logging.info(f"Fallback: Fetched {len(rates)} bars for {SYMBOL} on timeframe {timeframe} for 90 days")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        logging.info(f"Fetched {len(df)} bars for {SYMBOL} on timeframe {timeframe} for {days} days")
        return df[['open', 'high', 'low', 'close']]
    except Exception as e:
        logging.error(f"Error fetching data for {SYMBOL} on timeframe {timeframe}: {str(e)}")
        return pd.DataFrame()

# Calculate indicators
def calculate_indicators(df, timeframe):
    min_bars = max(EMA_SLOW, RSI_PERIOD, MACD_SLOW + MACD_SIGNAL)
    if df.empty or len(df) < min_bars:
        logging.warning(f"Insufficient data for indicators: {len(df)} bars, need {min_bars}")
        return df
    try:
        df = df.copy()
        df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
        if timeframe == mt5.TIMEFRAME_H1:
            ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
            ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
            df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
        return df
    except Exception as e:
        logging.error(f"Error calculating indicators: {str(e)}")
        return df

def calculate_rsi(prices, period):
    try:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        rs = rs.where(loss != 0, 0).fillna(0)
        return 100 - (100 / (1 + rs))
    except Exception as e:
        logging.error(f"Error calculating RSI: {str(e)}")
        return pd.Series(index=prices.index)

# Calculate support and resistance
def calculate_support_resistance(df_daily):
    if df_daily.empty or len(df_daily) < 20:
        logging.warning(f"Insufficient daily data for support/resistance: {len(df_daily)} bars")
        return None, None
    try:
        support = df_daily['low'].rolling(window=20).min().iloc[-1]
        resistance = df_daily['high'].rolling(window=20).max().iloc[-1]
        if pd.isna(support) or pd.isna(resistance):
            logging.warning("NaN in support/resistance")
            return None, None
        return support, resistance
    except Exception as e:
        logging.error(f"Error calculating support/resistance: {str(e)}")
        return None, None

# Trading logic for live trading
def trading_logic():
    try:
        # Check account equity
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to retrieve account info")
            return []
        equity = account_info.equity
        logging.debug(f"Current equity: ${equity:.2f}")
        if equity <= 0:
            logging.warning("Equity is zero or negative. No trades allowed.")
            return []

        # Fetch 1 year of data
        df_h1 = get_data(mt5.TIMEFRAME_H1)
        df_4h = get_data(mt5.TIMEFRAME_H4)
        df_daily = get_data(mt5.TIMEFRAME_D1)

        if df_h1.empty or df_4h.empty or df_daily.empty:
            logging.error("Failed to retrieve data for one or more timeframes")
            return []

        # Calculate indicators
        df_h1 = calculate_indicators(df_h1, mt5.TIMEFRAME_H1)
        df_4h = calculate_indicators(df_4h, mt5.TIMEFRAME_H4)
        df_daily = calculate_indicators(df_daily, mt5.TIMEFRAME_D1)

        # Validate indicator columns
        required_columns = ['ema_fast', 'ema_slow']
        h1_columns = required_columns + ['macd', 'macd_signal', 'rsi']
        if not all(col in df_h1.columns for col in h1_columns) or \
           not all(col in df_4h.columns for col in required_columns) or \
           not all(col in df_daily.columns for col in required_columns):
            logging.error("Missing indicator columns")
            return []

        # Check for NaN in latest indicators
        if df_h1[h1_columns].iloc[-1].isna().any() or \
           df_4h[required_columns].iloc[-1].isna().any() or \
           df_daily[required_columns].iloc[-1].isna().any():
            logging.error("NaN values in latest indicators")
            return []

        # Get real-time price
        tick = mt5.symbol_info_tick(SYMBOL)
        current_price = tick.ask if tick else df_h1['close'].iloc[-1]

        # Get indicator values
        ema50_4h = df_4h['ema_fast'].iloc[-1]
        ema200_4h = df_4h['ema_slow'].iloc[-1]
        ema50_daily = df_daily['ema_fast'].iloc[-1]
        ema200_daily = df_daily['ema_slow'].iloc[-1]
        macd = df_h1['macd'].iloc[-1]
        macd_signal = df_h1['macd_signal'].iloc[-1]
        rsi = df_h1['rsi'].iloc[-1]

        support, resistance = calculate_support_resistance(df_daily)
        if support is None or resistance is None:
            logging.error("Invalid support/resistance")
            return []

        # Check positions
        positions = mt5.positions_get(symbol=SYMBOL)
        if positions is None:
            logging.error("Failed to retrieve positions")
            return []
        position_open = len(positions) > 0

        # Trading conditions
        trend_up = ema50_4h > ema200_4h and ema50_daily > ema200_daily
        trend_down = ema50_4h < ema200_4h and ema50_daily < ema200_daily
        macd_buy = macd > macd_signal
        macd_sell = macd < macd_signal
        rsi_ok_buy = rsi < 65
        rsi_ok_sell = rsi > 35

        # Log signal checks
        logging.debug(f"Signal check: Price={current_price:.5f}, Trend_Up={trend_up}, MACD_Buy={macd_buy}, "
                      f"RSI_Buy={rsi_ok_buy}, Support={support:.5f}, Trend_Down={trend_down}, "
                      f"MACD_Sell={macd_sell}, RSI_Sell={rsi_ok_sell}, Resistance={resistance:.5f}")

        # Execute trades only if equity > 0 (redundant here due to initial check, but kept for clarity)
        if not position_open and equity > 0:
            if trend_up and macd_buy and rsi_ok_buy and current_price > support:
                send_order("buy", current_price)
            elif trend_down and macd_sell and rsi_ok_sell and current_price < resistance:
                send_order("sell", current_price)
            else:
                logging.debug("No trade: Conditions not met.")

        return [{'ticket': pos.ticket} for pos in positions]

    except Exception as e:
        logging.error(f"Error in trading_logic: {str(e)}")
        return []

# Send order to MT5
def send_order(order_type, price):
    try:
        sl = price - (RISK_PIPS * 0.0001) if order_type == "buy" else price + (RISK_PIPS * 0.0001)
        tp = price + (REWARD_PIPS * 0.0001) if order_type == "buy" else price - (REWARD_PIPS * 0.0001)
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": LOT_SIZE,
            "type": mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456,
            "comment": "Python script order",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"{order_type.upper()} order executed - Price: {price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}, "
                         f"Ticket: {result.deal}")
            return result.deal
        else:
            logging.error(f"Order failed: {result.comment}")
            return None
    except Exception as e:
        logging.error(f"Error sending order: {str(e)}")
        return None

# Check closed positions
def check_closed_positions(prev_positions):
    try:
        current_positions = mt5.positions_get(symbol=SYMBOL)
        current_tickets = {pos.ticket for pos in current_positions} if current_positions else set()
        prev_tickets = {pos['ticket'] for pos in prev_positions}

        closed_tickets = prev_tickets - current_tickets
        for ticket in closed_tickets:
            history = mt5.history_deals_get(position=ticket)
            if history:
                entry_price = next(deal.price for deal in history if deal.entry == mt5.DEAL_ENTRY_IN)
                exit_price = next(deal.price for deal in history if deal.entry == mt5.DEAL_ENTRY_OUT)
                profit = sum(deal.profit for deal in history)
                order_type = "BUY" if history[0].type == mt5.DEAL_TYPE_BUY else "SELL"
                logging.info(f"Position closed - Ticket: {ticket}, Type: {order_type}, "
                             f"Entry: {entry_price:.5f}, Exit: {exit_price:.5f}, Profit/Loss: {profit:.2f}")
    except Exception as e:
        logging.error(f"Error checking closed positions: {str(e)}")

# Close position (optional)
def close_position(ticket):
    try:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logging.error(f"Position {ticket} not found")
            return
        position = position[0]
        price = mt5.symbol_info_tick(SYMBOL).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": SYMBOL,
            "volume": position.volume,
            "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logging.info(f"Position {ticket} closed at {price:.5f}")
        else:
            logging.error(f"Close failed: {result.comment}")
    except Exception as e:
        logging.error(f"Error closing position {ticket}: {str(e)}")

# Backtesting function
def backtest(start_date, end_date):
    try:
        # Fetch data from 1 year prior to end_date
        adjusted_start = start_date - datetime.timedelta(days=365)
        df_h1 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, adjusted_start, end_date)
        df_4h = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H4, adjusted_start, end_date)
        df_daily = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_D1, adjusted_start, end_date)

        if df_h1 is None or len(df_h1) == 0:
            logging.error("No H1 data retrieved")
            return []
        if df_4h is None or len(df_4h) == 0:
            logging.error("No H4 data retrieved")
            return []
        if df_daily is None or len(df_daily) == 0:
            logging.error("No daily data retrieved")
            return []

        df_h1 = pd.DataFrame(df_h1).set_index(pd.to_datetime([t['time'] for t in df_h1], unit='s'))
        df_4h = pd.DataFrame(df_4h).set_index(pd.to_datetime([t['time'] for t in df_4h], unit='s'))
        df_daily = pd.DataFrame(df_daily).set_index(pd.to_datetime([t['time'] for t in df_daily], unit='s'))

        logging.info(f"Backtest data: H1={len(df_h1)} bars, H4={len(df_4h)} bars, Daily={len(df_daily)} bars")

        # Calculate indicators
        df_h1 = calculate_indicators(df_h1, mt5.TIMEFRAME_H1)
        df_4h = calculate_indicators(df_4h, mt5.TIMEFRAME_H4)
        df_daily = calculate_indicators(df_daily, mt5.TIMEFRAME_D1)

        # Validate indicator columns
        required_columns = ['ema_fast', 'ema_slow']
        h1_columns = required_columns + ['macd', 'macd_signal', 'rsi']
        if not all(col in df_h1.columns for col in h1_columns) or \
           not all(col in df_4h.columns for col in required_columns) or \
           not all(col in df_daily.columns for col in required_columns):
            logging.error("Missing indicator columns in backtest")
            return []

        start_equity = 10000
        equity = start_equity
        position = 0  # 1 for long, -1 for short, 0 for no position
        trades = []
        entry_price = sl = tp = 0

        for i in range(EMA_SLOW, len(df_h1)):
            current_time = df_h1.index[i]
            if current_time < start_date:  # Skip pre-start_date bars
                continue
            current_price = df_h1['close'].iloc[i]
            df_4h_current = df_4h[df_4h.index <= current_time]
            df_daily_current = df_daily[df_daily.index <= current_time]

            if df_4h_current.empty or df_daily_current.empty:
                logging.debug(f"Skipping {current_time}: No H4/Daily data")
                continue
            df_4h_current = df_4h_current.iloc[-1]
            df_daily_current = df_daily_current.iloc[-1]

            support, resistance = calculate_support_resistance(df_daily[df_daily.index <= current_time])
            if support is None or resistance is None:
                logging.debug(f"Skipping {current_time}: Invalid support/resistance")
                continue

            # Validate indicator values
            if df_h1[h1_columns].iloc[i].isna().any() or \
               df_4h_current[required_columns].isna().any() or \
               df_daily_current[required_columns].isna().any():
                logging.debug(f"Skipping {current_time}: NaN indicators")
                continue

            trend_up = df_4h_current['ema_fast'] > df_4h_current['ema_slow'] and \
                      df_daily_current['ema_fast'] > df_daily_current['ema_slow']
            trend_down = df_4h_current['ema_fast'] < df_4h_current['ema_slow'] and \
                        df_daily_current['ema_fast'] < df_daily_current['ema_slow']
            macd_buy = df_h1['macd'].iloc[i] > df_h1['macd_signal'].iloc[i]
            macd_sell = df_h1['macd'].iloc[i] < df_h1['macd_signal'].iloc[i]
            rsi_ok_buy = df_h1['rsi'].iloc[i] < 65
            rsi_ok_sell = df_h1['rsi'].iloc[i] > 35

            logging.debug(f"Backtest check at {current_time}: Price={current_price:.5f}, Equity=${equity:.2f}, "
                          f"Trend_Up={trend_up}, MACD_Buy={macd_buy}, RSI_Buy={rsi_ok_buy}, Support={support:.5f}")

            if position == 0:
                if equity <= 0:
                    logging.warning(f"Equity reached zero or negative (${equity:.2f}) at {current_time}. Stopping backtest.")
                    break
                if trend_up and macd_buy and rsi_ok_buy and current_price > support:
                    position = 1
                    entry_price = current_price
                    sl = entry_price - (RISK_PIPS * 0.0001)
                    tp = entry_price + (REWARD_PIPS * 0.0001)
                    trades.append(('BUY', current_time, entry_price, sl, tp))
                    logging.info(f"Backtest - BUY executed - Time: {current_time}, Price: {entry_price:.5f}, "
                                 f"SL: {sl:.5f}, TP: {tp:.5f}, Equity: ${equity:.2f}")
                elif trend_down and macd_sell and rsi_ok_sell and current_price < resistance:
                    position = -1
                    entry_price = current_price
                    sl = entry_price + (RISK_PIPS * 0.0001)
                    tp = entry_price - (REWARD_PIPS * 0.0001)
                    trades.append(('SELL', current_time, entry_price, sl, tp))
                    logging.info(f"Backtest - SELL executed - Time: {current_time}, Price: {entry_price:.5f}, "
                                 f"SL: {sl:.5f}, TP: {tp:.5f}, Equity: ${equity:.2f}")
            elif position == 1:
                if current_price <= sl or current_price >= tp or current_price >= resistance or \
                   df_h1['macd'].iloc[i] < df_h1['macd_signal'].iloc[i]:
                    profit_loss = (current_price - entry_price) * LOT_SIZE * 100000
                    equity += profit_loss
                    position = 0
                    trades.append(('CLOSE', current_time, current_price, profit_loss))
                    logging.info(f"Backtest - Position closed - Time: {current_time}, Exit Price: {current_price:.5f}, "
                                 f"Profit/Loss: ${profit_loss:.2f}, Equity: ${equity:.2f}")
                    if equity <= 0:
                        logging.warning(f"Equity reached zero or negative (${equity:.2f}) at {current_time}. Stopping backtest.")
                        break
            elif position == -1:
                if current_price >= sl or current_price <= tp or current_price <= support or \
                   df_h1['macd'].iloc[i] > df_h1['macd_signal'].iloc[i]:
                    profit_loss = (entry_price - current_price) * LOT_SIZE * 100000
                    equity += profit_loss
                    position = 0
                    trades.append(('CLOSE', current_time, current_price, profit_loss))
                    logging.info(f"Backtest - Position closed - Time: {current_time}, Exit Price: {current_price:.5f}, "
                                 f"Profit/Loss: ${profit_loss:.2f}, Equity: ${equity:.2f}")
                    if equity <= 0:
                        logging.warning(f"Equity reached zero or negative (${equity:.2f}) at {current_time}. Stopping backtest.")
                        break

        end_equity = equity
        percent_diff = ((end_equity - start_equity) / start_equity) * 100

        logging.info(f"Backtest Results - Start Equity: ${start_equity:.2f}, End Equity: ${end_equity:.2f}, "
                     f"Return: {percent_diff:.2f}%, Trades: {len([t for t in trades if t[0] in ['BUY', 'SELL']])}")
        
        return trades

    except Exception as e:
        logging.error(f"Error in backtest: {str(e)}")
        return []

# Sync with hourly execution
def sync_with_hour():
    now = datetime.datetime.now()
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1))
    delay = (next_hour - now).total_seconds() + 10
    logging.info(f"Sleeping {delay:.1f} seconds until {next_hour + datetime.timedelta(seconds=10)}")
    time.sleep(delay)

if __name__ == "__main__":
    initialize_mt5()

    # Backtesting
    start = datetime.datetime(2025, 4, 9)
    end = datetime.datetime(2025, 4, 10)
    logging.info("Running backtest...")
    logging.info(f"Backtest Start Date: {start}, End Date: {end}")
    trades = backtest(start, end)

    # Live trading (uncomment to run)
    """
    logging.info("Starting hourly live trading...")
    while True:
        sync_with_hour()
        prev_positions = trading_logic()
        check_closed_positions(prev_positions)
    """

    mt5.shutdown()
