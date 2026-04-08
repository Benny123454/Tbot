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
    close = df['close']
    rsi = calculate_rsi(close)
    macd_line, signal_line, hist = calculate_macd(close)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close)
    ema50 = calculate_ema(close, 50)

    rsi_val    = rsi.iloc[-1]
    price      = close.iloc[-1]
    macd_cur   = macd_line.iloc[-1]
    macd_prev  = macd_line.iloc[-2]
    sig_cur    = signal_line.iloc[-1]
    sig_prev   = signal_line.iloc[-2]

    macd_cross_up   = macd_prev < sig_prev and macd_cur > sig_cur
    macd_cross_down = macd_prev > sig_prev and macd_cur < sig_cur

    bb_low  = bb_lower.iloc[-1]
    bb_high = bb_upper.iloc[-1]
    ema_val = ema50.iloc[-1]

    buy_score = 0
    sell_score = 0
    reasons = []

    # RSI
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

    # MACD
    if macd_cross_up:
        buy_score += 30
        reasons.append("MACD bullisches Kreuz")
    elif macd_cur > sig_cur:
        buy_score += 10
    if macd_cross_down:
        sell_score += 30
        reasons.append("MACD bärisches Kreuz")
    elif macd_cur < sig_cur:
        sell_score += 10

    # Bollinger Bands
    if pd.notna(bb_low) and price <= bb_low:
        buy_score += 25
        reasons.append("Preis am unteren Bollinger Band")
    elif pd.notna(bb_low) and price <= bb_low * 1.01:
        buy_score += 15
    if pd.notna(bb_high) and price >= bb_high:
        sell_score += 25
        reasons.append("Preis am oberen Bollinger Band")
    elif pd.notna(bb_high) and price >= bb_high * 0.99:
        sell_score += 15

    # EMA Trend
    if pd.notna(ema_val):
        if price > ema_val:
            buy_score += 10
        else:
            sell_score += 10

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
        "rsi": round(float(rsi_val), 2),
        "macd": round(float(macd_cur), 6),
        "macd_signal": round(float(sig_cur), 6),
        "bb_upper": round(float(bb_high), 4) if pd.notna(bb_high) else None,
        "bb_lower": round(float(bb_low), 4) if pd.notna(bb_low) else None,
        "ema50": round(float(ema_val), 4) if pd.notna(ema_val) else None,
        "price": round(float(price), 4),
    }


def get_chart_data(df: pd.DataFrame) -> dict:
    """Gibt OHLCV + Indikatoren für das Frontend zurück"""
    df = df.tail(120).copy().reset_index(drop=True)
    close = df['close']

    rsi       = calculate_rsi(close)
    macd_line, signal_line, histogram = calculate_macd(close)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(close)

    def ts(t):
        """Timestamp → Unix-Sekunden (int) für TradingView"""
        try:
            if isinstance(t, (int, float)):
                return int(t)
            ts_val = pd.Timestamp(t)
            return int(ts_val.timestamp())
        except Exception:
            return None

    timestamps = [ts(t) for t in df['timestamp']]

    def safe_list(series):
        return [round(float(v), 6) if pd.notna(v) else None for v in series]

    ohlcv = []
    for i, row in df.iterrows():
        ohlcv.append({
            "open":   round(float(row['open']), 6),
            "high":   round(float(row['high']), 6),
            "low":    round(float(row['low']), 6),
            "close":  round(float(row['close']), 6),
            "volume": round(float(row['volume']), 2),
        })

    return {
        "timestamps":   timestamps,
        "ohlcv":        ohlcv,
        "rsi":          safe_list(rsi),
        "macd":         safe_list(macd_line),
        "macd_signal":  safe_list(signal_line),
        "macd_hist":    safe_list(histogram),
        "bb_upper":     safe_list(bb_upper),
        "bb_mid":       safe_list(bb_mid),
        "bb_lower":     safe_list(bb_lower),
    }
