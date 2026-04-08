"""
Trading Engine – Paper + Live Trading
Unterstützt: Krypto (Binance via ccxt) + Aktien (yfinance)
"""
import json
import os
import time
import threading
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf
import ccxt

from indicators import generate_signal, get_chart_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'portfolio.json')


def load_portfolio() -> dict:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    return {
        "balance": 10000.0,
        "initial_balance": 10000.0,
        "positions": {},
        "trades": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def save_portfolio(portfolio: dict):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)


class TradingBot:
    CRYPTO_SYMBOLS = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"]
    STOCK_SYMBOLS  = ["AAPL", "NVDA", "MSFT", "AMZN"]

    # Risiko-Parameter
    MAX_POSITION_PCT  = 0.10   # max 10% des Portfolios pro Trade
    STOP_LOSS_PCT     = 0.03   # 3% Stop Loss
    TAKE_PROFIT_PCT   = 0.06   # 6% Take Profit
    MAX_POSITIONS     = 6      # max offene Positionen

    def __init__(self, mode: str = "paper", binance_key: str = "", binance_secret: str = ""):
        self.mode = mode
        self.running = False
        self.portfolio = load_portfolio()
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self.last_scan = None
        self.log_entries = []

        # Binance (public für Daten, auth für Live)
        exchange_config = {"enableRateLimit": True}
        if binance_key and binance_secret:
            exchange_config.update({"apiKey": binance_key, "secret": binance_secret})
        self.exchange = ccxt.binance(exchange_config)

    # ──────────────────────────────────────────
    # Daten abrufen
    # ──────────────────────────────────────────

    def _fetch_crypto_df(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> Optional[pd.DataFrame]:
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logger.warning(f"Krypto-Daten Fehler {symbol}: {e}")
            return None

    def _fetch_stock_df(self, symbol: str, interval: str = "1h", period: str = "60d") -> Optional[pd.DataFrame]:
        try:
            ticker = yf.Ticker(symbol)
            raw = ticker.history(interval=interval, period=period)
            if raw.empty:
                return None
            df = raw.reset_index()
            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={'datetime': 'timestamp', 'date': 'timestamp'})
            if 'timestamp' not in df.columns:
                df['timestamp'] = df.index
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            logger.warning(f"Aktien-Daten Fehler {symbol}: {e}")
            return None

    def get_df(self, symbol: str) -> Optional[pd.DataFrame]:
        if '/' in symbol:
            return self._fetch_crypto_df(symbol)
        return self._fetch_stock_df(symbol)

    def get_current_price(self, symbol: str) -> Optional[float]:
        try:
            if '/' in symbol:
                ticker = self.exchange.fetch_ticker(symbol)
                return ticker['last']
            else:
                t = yf.Ticker(symbol)
                info = t.fast_info
                return float(info.last_price)
        except Exception as e:
            logger.warning(f"Preis Fehler {symbol}: {e}")
            return None

    # ──────────────────────────────────────────
    # Trade Ausführung (Paper)
    # ──────────────────────────────────────────

    def _buy(self, symbol: str, price: float, signal_info: dict):
        with self._lock:
            if len(self.portfolio['positions']) >= self.MAX_POSITIONS:
                return
            if symbol in self.portfolio['positions']:
                return

            invest = self.portfolio['balance'] * self.MAX_POSITION_PCT
            if invest < 10:
                return

            qty = invest / price
            self.portfolio['balance'] -= invest

            self.portfolio['positions'][symbol] = {
                "symbol": symbol,
                "qty": qty,
                "entry_price": price,
                "invested": invest,
                "stop_loss": price * (1 - self.STOP_LOSS_PCT),
                "take_profit": price * (1 + self.TAKE_PROFIT_PCT),
                "opened_at": datetime.now(timezone.utc).isoformat(),
                "signal_strength": signal_info.get('strength', 0),
                "reasons": signal_info.get('reasons', []),
            }

            trade = {
                "id": len(self.portfolio['trades']) + 1,
                "symbol": symbol,
                "type": "BUY",
                "price": price,
                "qty": qty,
                "value": invest,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "pnl": 0,
            }
            self.portfolio['trades'].insert(0, trade)
            save_portfolio(self.portfolio)
            self._log(f"🟢 KAUF {symbol} @ {price:.4f} | Investiert: {invest:.2f}€")

    def _sell(self, symbol: str, price: float, reason: str = "Signal"):
        with self._lock:
            if symbol not in self.portfolio['positions']:
                return

            pos = self.portfolio['positions'].pop(symbol)
            proceeds = pos['qty'] * price
            pnl = proceeds - pos['invested']
            pnl_pct = (pnl / pos['invested']) * 100
            self.portfolio['balance'] += proceeds

            trade = {
                "id": len(self.portfolio['trades']) + 1,
                "symbol": symbol,
                "type": "SELL",
                "price": price,
                "qty": pos['qty'],
                "value": proceeds,
                "pnl": round(pnl, 2),
                "pnl_pct": round(pnl_pct, 2),
                "reason": reason,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self.portfolio['trades'].insert(0, trade)
            save_portfolio(self.portfolio)
            icon = "🟢" if pnl >= 0 else "🔴"
            self._log(f"{icon} VERKAUF {symbol} @ {price:.4f} | PnL: {pnl:+.2f}€ ({pnl_pct:+.1f}%) | Grund: {reason}")

    # ──────────────────────────────────────────
    # Hauptschleife
    # ──────────────────────────────────────────

    def _process_symbol(self, symbol: str):
        df = self.get_df(symbol)
        if df is None or len(df) < 50:
            return

        signal_info = generate_signal(df)
        price = signal_info['price']

        # Prüfe offene Positionen auf SL/TP
        with self._lock:
            pos = self.portfolio['positions'].get(symbol)

        if pos:
            if price <= pos['stop_loss']:
                self._sell(symbol, price, "Stop Loss")
                return
            if price >= pos['take_profit']:
                self._sell(symbol, price, "Take Profit")
                return
            if signal_info['signal'] == 'SELL':
                self._sell(symbol, price, "Verkaufssignal")
                return
        else:
            if signal_info['signal'] == 'BUY':
                self._buy(symbol, price, signal_info)

    def _run(self):
        all_symbols = self.CRYPTO_SYMBOLS + self.STOCK_SYMBOLS
        while self.running:
            self._log("🔍 Scanning alle Märkte...")
            for symbol in all_symbols:
                if not self.running:
                    break
                try:
                    self._process_symbol(symbol)
                except Exception as e:
                    logger.error(f"Fehler bei {symbol}: {e}")
                time.sleep(0.5)

            self.last_scan = datetime.now(timezone.utc).isoformat()
            self._update_pnl()

            # Warte 5 Minuten bis zum nächsten Scan
            for _ in range(300):
                if not self.running:
                    break
                time.sleep(1)

    def _update_pnl(self):
        """Aktualisiert unrealisierte PnL aller Positionen"""
        with self._lock:
            for symbol, pos in self.portfolio['positions'].items():
                try:
                    price = self.get_current_price(symbol)
                    if price:
                        pos['current_price'] = price
                        pos['unrealized_pnl'] = round((price - pos['entry_price']) * pos['qty'], 2)
                        pos['unrealized_pnl_pct'] = round(((price - pos['entry_price']) / pos['entry_price']) * 100, 2)
                except Exception:
                    pass
        save_portfolio(self.portfolio)

    def _log(self, msg: str):
        entry = {"time": datetime.now(timezone.utc).isoformat(), "msg": msg}
        self.log_entries.insert(0, entry)
        self.log_entries = self.log_entries[:100]
        logger.info(msg)

    # ──────────────────────────────────────────
    # Steuerung
    # ──────────────────────────────────────────

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._log("✅ Bot gestartet")

    def stop(self):
        self.running = False
        self._log("⛔ Bot gestoppt")

    def reset_portfolio(self):
        self.portfolio = {
            "balance": 10000.0,
            "initial_balance": 10000.0,
            "positions": {},
            "trades": [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        save_portfolio(self.portfolio)
        self._log("🔄 Portfolio zurückgesetzt")

    # ──────────────────────────────────────────
    # Status für API
    # ──────────────────────────────────────────

    def get_status(self) -> dict:
        with self._lock:
            portfolio = self.portfolio.copy()

        balance = portfolio['balance']
        positions = portfolio['positions']
        trades = portfolio['trades']

        # Portfolio Wert berechnen
        positions_value = sum(
            p.get('current_price', p['entry_price']) * p['qty']
            for p in positions.values()
        )
        total_value = balance + positions_value
        initial = portfolio['initial_balance']
        total_pnl = total_value - initial
        total_pnl_pct = (total_pnl / initial) * 100

        # Win Rate berechnen
        closed = [t for t in trades if t['type'] == 'SELL']
        win_rate = 0
        if closed:
            wins = sum(1 for t in closed if t.get('pnl', 0) >= 0)
            win_rate = round((wins / len(closed)) * 100, 1)

        return {
            "running": self.running,
            "mode": self.mode,
            "last_scan": self.last_scan,
            "balance": round(balance, 2),
            "positions_value": round(positions_value, 2),
            "total_value": round(total_value, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "initial_balance": initial,
            "open_positions": len(positions),
            "total_trades": len(closed),
            "win_rate": win_rate,
            "positions": list(positions.values()),
            "trades": trades[:50],
            "logs": self.log_entries[:30],
        }

    def get_chart(self, symbol: str) -> dict:
        df = self.get_df(symbol)
        if df is None:
            return {}
        return get_chart_data(df)

    def get_signal(self, symbol: str) -> dict:
        df = self.get_df(symbol)
        if df is None:
            return {"signal": "HOLD", "error": "Keine Daten"}
        return generate_signal(df)
