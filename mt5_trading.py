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
        logging.FileHandler(log_file),  # Log to file
        logging.StreamHandler()         # Log to console
    ]
)

# Initialize MT5 connection and ensure correct account
def initialize_mt5():
    if not mt5.initialize():
        logging.error("Failed to initialize MT5")
        quit()

    current_account = mt5.account_info()
    if current_account is None:
        logging.info("No account currently connected. Logging into demo account...")
    else:
        logging.info(f"Current account: {current_account.login} (Server: {current_account.server})")

    if current_account is None or str(current_account.login) != DEMO_ACCOUNT:
        logging.info(f"Switching to demo account: {DEMO_ACCOUNT}")
        if not mt5.login(int(DEMO_ACCOUNT), password=DEMO_PASSWORD, server=DEMO_SERVER):
            logging.error(f"Failed to login to demo account {DEMO_ACCOUNT}: {mt5.last_error()}")
            quit()
        else:
            logging.info(f"Successfully logged into demo account: {DEMO_ACCOUNT} (Server: {DEMO_SERVER})")
    else:
        logging.info(f"Already connected to correct demo account: {DEMO_ACCOUNT}")

    account_info = mt5.account_info()
    logging.info(f"Confirmed Account: {account_info.login}, Balance: {account_info.balance}, "
                 f"Equity: {account_info.equity}, Server: {account_info.server}")

# Define strategy parameters
SYMBOL = "XAUUSD"
LOT_SIZE = 0.01
EMA_FAST = 50
EMA_SLOW = 200
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14
RISK_PIPS = 20  # Risk 20 pips
REWARD_PIPS = 40  # Reward 40 pips (1:2 ratio)

# Function to get OHLC data from MT5
def get_data(timeframe, count):
    rates = mt5.copy_rates_from_pos(SYMBOL, timeframe, 0, count)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df[['open', 'high', 'low', 'close']]

# Calculate indicators
def calculate_indicators(df, timeframe):
    df['ema_fast'] = df['close'].ewm(span=EMA_FAST, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    if timeframe == mt5.TIMEFRAME_H1:
        ema_fast = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
        df['rsi'] = calculate_rsi(df['close'], RSI_PERIOD)
    return df

def calculate_rsi(prices, period):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Calculate support and resistance
def calculate_support_resistance(df_daily):
    support = df_daily['low'].rolling(window=20).min().iloc[-1]
    resistance = df_daily['high'].rolling(window=20).max().iloc[-1]
    return support, resistance

# Trading logic for live trading
def trading_logic():
    df_h1 = get_data(mt5.TIMEFRAME_H1, 300)
    df_4h = get_data(mt5.TIMEFRAME_H4, 300)
    df_daily = get_data(mt5.TIMEFRAME_D1, 100)

    df_h1 = calculate_indicators(df_h1, mt5.TIMEFRAME_H1)
    df_4h = calculate_indicators(df_4h, mt5.TIMEFRAME_H4)
    df_daily = calculate_indicators(df_daily, mt5.TIMEFRAME_D1)

    current_price = df_h1['close'].iloc[-1]
    ema50_4h = df_4h['ema_fast'].iloc[-1]
    ema200_4h = df_4h['ema_slow'].iloc[-1]
    ema50_daily = df_daily['ema_fast'].iloc[-1]
    ema200_daily = df_daily['ema_slow'].iloc[-1]
    macd = df_h1['macd'].iloc[-1]
    macd_signal = df_h1['macd_signal'].iloc[-1]
    rsi = df_h1['rsi'].iloc[-1]

    support, resistance = calculate_support_resistance(df_daily)

    positions = mt5.positions_get(symbol=SYMBOL)
    position_open = len(positions) > 0

    trend_up = ema50_4h > ema200_4h and ema50_daily > ema200_daily
    trend_down = ema50_4h < ema200_4h and ema50_daily < ema200_daily
    macd_buy = macd > macd_signal
    macd_sell = macd < macd_signal
    rsi_ok_buy = rsi < 65
    rsi_ok_sell = rsi > 35

    if not position_open:
        if trend_up and macd_buy and rsi_ok_buy and current_price > support:
            send_order("buy", current_price)
        elif trend_down and macd_sell and rsi_ok_sell and current_price < resistance:
            send_order("sell", current_price)

    return [{'ticket': pos.ticket} for pos in positions] if positions else []

# Send order to MT5 with SL/TP and log trade execution
def send_order(order_type, price):
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
        return result.deal  # Return ticket number for tracking
    else:
        logging.error(f"Order failed: {result.comment}")
        return None

# Check and log closed positions for live trading
def check_closed_positions(prev_positions):
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

# Close position (optional, for manual closure)
def close_position(ticket):
    position = mt5.positions_get(ticket=ticket)[0]
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

# Backtesting function with SL/TP (unchanged, with added logging)
def backtest(start_date, end_date):
    df_h1 = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H1, start_date, end_date)
    df_4h = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_H4, start_date, end_date)
    df_daily = mt5.copy_rates_range(SYMBOL, mt5.TIMEFRAME_D1, start_date, end_date)

    df_h1 = pd.DataFrame(df_h1).set_index(pd.to_datetime([t['time'] for t in df_h1], unit='s'))
    df_4h = pd.DataFrame(df_4h).set_index(pd.to_datetime([t['time'] for t in df_4h], unit='s'))
    df_daily = pd.DataFrame(df_daily).set_index(pd.to_datetime([t['time'] for t in df_daily], unit='s'))

    df_h1 = calculate_indicators(df_h1, mt5.TIMEFRAME_H1)
    df_4h = calculate_indicators(df_4h, mt5.TIMEFRAME_H4)
    df_daily = calculate_indicators(df_daily, mt5.TIMEFRAME_D1)

    start_equity = 10000
    equity = start_equity
    position = 0  # 1 for long, -1 for short, 0 for no position
    trades = []
    entry_price = sl = tp = 0

    for i in range(EMA_SLOW, len(df_h1)):
        current_time = df_h1.index[i]
        current_price = df_h1['close'].iloc[i]
        df_4h_current = df_4h[df_4h.index <= current_time].iloc[-1]
        df_daily_current = df_daily[df_daily.index <= current_time].iloc[-1]
        support, resistance = calculate_support_resistance(df_daily[df_daily.index <= current_time])

        trend_up = df_4h_current['ema_fast'] > df_4h_current['ema_slow'] and df_daily_current['ema_fast'] > df_daily_current['ema_slow']
        trend_down = df_4h_current['ema_fast'] < df_4h_current['ema_slow'] and df_daily_current['ema_fast'] < df_daily_current['ema_slow']
        macd_buy = df_h1['macd'].iloc[i] > df_h1['macd_signal'].iloc[i]
        macd_sell = df_h1['macd'].iloc[i] < df_h1['macd_signal'].iloc[i]
        rsi_ok_buy = df_h1['rsi'].iloc[i] < 65
        rsi_ok_sell = df_h1['rsi'].iloc[i] > 35

        if position == 0:
            if trend_up and macd_buy and rsi_ok_buy and current_price > support:
                position = 1
                entry_price = current_price
                sl = entry_price - (RISK_PIPS * 0.0001)
                tp = entry_price + (REWARD_PIPS * 0.0001)
                trades.append(('BUY', current_time, entry_price, sl, tp))
                logging.info(f"Backtest - BUY executed - Time: {current_time}, Price: {entry_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
            elif trend_down and macd_sell and rsi_ok_sell and current_price < resistance:
                position = -1
                entry_price = current_price
                sl = entry_price + (RISK_PIPS * 0.0001)
                tp = entry_price - (REWARD_PIPS * 0.0001)
                trades.append(('SELL', current_time, entry_price, sl, tp))
                logging.info(f"Backtest - SELL executed - Time: {current_time}, Price: {entry_price:.5f}, SL: {sl:.5f}, TP: {tp:.5f}")
        elif position == 1:
            if current_price <= sl or current_price >= tp or current_price >= resistance or df_h1['macd'].iloc[i] < df_h1['macd_signal'].iloc[i]:
                profit_loss = (current_price - entry_price) * LOT_SIZE * 100000
                equity += profit_loss
                position = 0
                trades.append(('CLOSE', current_time, current_price, profit_loss))
                logging.info(f"Backtest - Position closed - Time: {current_time}, Exit Price: {current_price:.5f}, Profit/Loss: ${profit_loss:.2f}")
        elif position == -1:
            if current_price >= sl or current_price <= tp or current_price <= support or df_h1['macd'].iloc[i] > df_h1['macd_signal'].iloc[i]:
                profit_loss = (entry_price - current_price) * LOT_SIZE * 100000
                equity += profit_loss
                position = 0
                trades.append(('CLOSE', current_time, current_price, profit_loss))
                logging.info(f"Backtest - Position closed - Time: {current_time}, Exit Price: {current_price:.5f}, Profit/Loss: ${profit_loss:.2f}")

    end_equity = equity
    percent_diff = ((end_equity - start_equity) / start_equity) * 100

    logging.info(f"Backtest Results - Starting Equity: ${start_equity:.2f}, Ending Equity: ${end_equity:.2f}, "
                 f"Percentage Difference: {percent_diff:.2f}%, Number of Trades: {len([t for t in trades if t[0] in ['BUY', 'SELL']])}")
    logging.info("Backtest Trade History:")
    for i in range(0, len(trades), 2):  # Step by 2 to pair BUY/SELL with CLOSE
        entry = trades[i]
        exit = trades[i + 1] if i + 1 < len(trades) else None
        if entry[0] in ['BUY', 'SELL'] and exit and exit[0] == 'CLOSE':
            profit_loss = exit[3]  # Get profit/loss from CLOSE tuple
            logging.info(f"{entry[0]} - Entry: {entry[2]:.5f} on {entry[1]}, Exit: {exit[2]:.5f} on {exit[1]}, Profit/Loss: ${profit_loss:.2f}")

    return trades

import time
from datetime import datetime, timedelta

def sync_with_hour():
    now = datetime.now()
    next_hour = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
    delay = (next_hour - now).total_seconds()
    print(f"Sleeping {delay:.1f} seconds until {next_hour}")
    time.sleep(delay)

# Main execution
if __name__ == "__main__":
    # Initialize MT5 with demo account check
    initialize_mt5()

    # # Backtesting (uncomment to run)
    # start = datetime.datetime(2025, 1, 1)
    # end = datetime.datetime(2025, 4, 10)
    # logging.info("Running backtest...")
    # logging.info(f"Backtest Start Date: {start}")
    # logging.info(f"Backtest End Date: {end}")
    # trades = backtest(start, end)

    # Live trading loop (uncomment to run live)
    logging.info("Starting live trading...")
    # Run trading logic once immediately before starting the loop
    prev_positions = trading_logic()

    while True:
        try:
            current_positions = trading_logic()
            sync_with_hour()
            check_closed_positions(prev_positions)
            prev_positions = current_positions
            time.sleep(3600)  # Check every hour
        except Exception as e:
            logging.error(f"Error in trading loop: {str(e)}")
            time.sleep(60)  # Wait a minute before retrying
    

# Shutdown MT5 connection
mt5.shutdown()