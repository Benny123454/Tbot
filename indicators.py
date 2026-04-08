"""
Technische Indikatoren: RSI, MACD, Bollinger Bands
"""
import pandas as pd
import numpy as np


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    return upper, sma, lower


def calculate_ema(prices: pd.Series, period: int = 50) -> pd.Series:
    return prices.ewm(span=period, adjust=False).mean()


def generate_signal(df: pd.DataFrame) -> dict:
    """
    Kombinierte Strategie: RSI + MACD + Bollinger Bands
    Returns: dict mit 'signal' ('BUY'/'SELL'/'HOLD'), 'strength' (0-100), 'reasons'
    """
    close = df['close']

    # Indikatoren berechnen
    rsi = calculate_rsi(close)
    macd_line, signal_line, hist = calculate_macd(close)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close)
    ema50 = calculate_ema(close, 50)

    last = -1  # letzter Wert
    prev = -2  # vorletzter Wert

    rsi_val = rsi.iloc[last]
    price = close.iloc[last]
    macd_cross_up = (macd_line.iloc[prev] < signal_line.iloc[prev]) and (macd_line.iloc[last] > signal_line.iloc[last])
    macd_cross_down = (macd_line.iloc[prev] > signal_line.iloc[prev]) and (macd_line.iloc[last] < signal_line.iloc[last])
    bb_low = bb_lower.iloc[last]
    bb_high = bb_upper.iloc[last]
    ema_trend_up = price > ema50.iloc[last]

    buy_score = 0
    sell_score = 0
    reasons = []

    # RSI Signale
    if rsi_val < 30:
        buy_score += 35
        reasons.append(f"RSI stark überverkauft ({rsi_val:.1f})")
    elif rsi_val < 40:
        buy_score += 20
        reasons.append(f"RSI überverkauft ({rsi_val:.1f})")
    elif rsi_val > 70:
        sell_score += 35
        reasons.append(f"RSI stark überkauft ({rsi_val:.1f})")
    elif rsi_val > 60:
        sell_score += 20
        reasons.append(f"RSI überkauft ({rsi_val:.1f})")

    # MACD Signale
    if macd_cross_up:
        buy_score += 30
        reasons.append("MACD bullisches Kreuz")
    elif macd_line.iloc[last] > signal_line.iloc[last]:
        buy_score += 10

    if macd_cross_down:
        sell_score += 30
        reasons.append("MACD bärisches Kreuz")
    elif macd_line.iloc[last] < signal_line.iloc[last]:
        sell_score += 10

    # Bollinger Bands
    if price <= bb_low:
        buy_score += 25
        reasons.append("Preis an unterem Bollinger Band")
    elif price <= bb_low * 1.01:
        buy_score += 15

    if price >= bb_high:
        sell_score += 25
        reasons.append("Preis an oberem Bollinger Band")
    elif price >= bb_high * 0.99:
        sell_score += 15

    # Trend Filter (EMA50)
    if ema_trend_up:
        buy_score += 10
    else:
        sell_score += 10

    # Signal bestimmen
    if buy_score >= 55:
        signal = "BUY"
        strength = min(buy_score, 100)
    elif sell_score >= 55:
        signal = "SELL"
        strength = min(sell_score, 100)
    else:
        signal = "HOLD"
        strength = 0

    return {
        "signal": signal,
        "strength": strength,
        "reasons": reasons,
        "rsi": round(rsi_val, 2),
        "macd": round(macd_line.iloc[last], 4),
        "macd_signal": round(signal_line.iloc[last], 4),
        "bb_upper": round(bb_high, 4),
        "bb_lower": round(bb_low, 4),
        "ema50": round(ema50.iloc[last], 4),
        "price": round(price, 4),
    }


def get_chart_data(df: pd.DataFrame) -> dict:
    """Gibt OHLCV + Indikatoren für Frontend zurück"""
    close = df['close']
    rsi = calculate_rsi(close)
    macd_line, signal_line, histogram = calculate_macd(close)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close)

    timestamps = df['timestamp'].tolist() if 'timestamp' in df.columns else df.index.tolist()

    return {
        "timestamps": [str(t) for t in timestamps[-100:]],
        "ohlcv": df[['open', 'high', 'low', 'close', 'volume']].tail(100).to_dict('records'),
        "rsi": rsi.tail(100).round(2).tolist(),
        "macd": macd_line.tail(100).round(4).tolist(),
        "macd_signal": signal_line.tail(100).round(4).tolist(),
        "macd_hist": histogram.tail(100).round(4).tolist(),
        "bb_upper": bb_upper.tail(100).round(4).tolist(),
        "bb_mid": bb_mid.tail(100).round(4).tolist(),
        "bb_lower": bb_lower.tail(100).round(4).tolist(),
    }
