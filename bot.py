"""
Trading Engine – Paper Trading
Krypto:  CoinGecko API  (kostenlos, keine Auth, funktioniert überall)
Aktien:  yfinance mit User-Agent Fix
"""
import json
import os
import time
import threading
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import numpy as np
import requests
import yfinance as yf

from indicators import generate_signal, get_chart_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'portfolio.json')

# ── Gemeinsame HTTP-Session mit Browser User-Agent ───────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': 'application/json',
})

# ── Symbol-Definitionen ──────────────────────────────────────────────────────
CRYPTO_SYMBOLS = {
    'BTC-USD':  'bitcoin',
    'ETH-USD':  'ethereum',
    'BNB-USD':  'binancecoin',
    'SOL-USD':  'solana',
    'XRP-USD':  'ripple',
    'ADA-USD':  'cardano',
    'DOGE-USD': 'dogecoin',
    'AVAX-USD': 'avalanche-2',
    'LINK-USD': 'chainlink',
    'DOT-USD':  'polkadot',
}

STOCK_SYMBOLS = {
    'AAPL':  'Apple',
    'NVDA':  'Nvidia',
    'MSFT':  'Microsoft',
    'AMZN':  'Amazon',
    'GOOGL': 'Alphabet',
    'META':  'Meta',
    'TSLA':  'Tesla',
    'AMD':   'AMD',
    'NFLX':  'Netflix',
    'JPM':   'JPMorgan',
}

DISPLAY_NAMES = {
    'BTC-USD':'Bitcoin','ETH-USD':'Ethereum','BNB-USD':'BNB',
    'SOL-USD':'Solana','XRP-USD':'XRP','ADA-USD':'Cardano',
    'DOGE-USD':'Dogecoin','AVAX-USD':'Avalanche','LINK-USD':'Chainlink',
    'DOT-USD':'Polkadot',
    **STOCK_SYMBOLS,
}

ALL_SYMBOLS = {**{k: k for k in CRYPTO_SYMBOLS}, **{k: k for k in STOCK_SYMBOLS}}


# ── Portfolio I/O ────────────────────────────────────────────────────────────

def load_portfolio() -> dict:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'balance': 10000.0,
        'initial_balance': 10000.0,
        'positions': {},
        'trades': [],
        'created_at': datetime.now(timezone.utc).isoformat(),
    }


def save_portfolio(portfolio: dict):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)


# ── Datenabruf ───────────────────────────────────────────────────────────────

def fetch_crypto_df(symbol: str) -> Optional[pd.DataFrame]:
    """CoinGecko OHLC – kostenlos, keine Auth, 4h-Kerzen über 30 Tage"""
    cg_id = CRYPTO_SYMBOLS.get(symbol)
    if not cg_id:
        return None
    try:
        url = f'https://api.coingecko.com/api/v3/coins/{cg_id}/ohlc'
        resp = _SESSION.get(url, params={'vs_currency': 'usd', 'days': '30'}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['volume'] = 0.0
        return df
    except Exception as e:
        logger.warning(f'CoinGecko Fehler {symbol}: {e}')
        return None


def fetch_stock_df(symbol: str) -> Optional[pd.DataFrame]:
    """yfinance – curl_cffi Session wird intern von yfinance verwaltet"""
    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.history(interval='1h', period='60d')
        if raw is None or raw.empty:
            # Fallback: kürzerer Zeitraum
            ticker2 = yf.Ticker(symbol)
            raw = ticker2.history(interval='1d', period='180d')
        if raw is None or raw.empty:
            logger.warning(f'Keine Daten für {symbol}')
            return None
        df = raw.reset_index()
        df.columns = [c.lower() for c in df.columns]
        for col in ['datetime', 'date']:
            if col in df.columns:
                df = df.rename(columns={col: 'timestamp'})
                break
        if 'timestamp' not in df.columns:
            df['timestamp'] = df.index
        # Timezone entfernen
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)
        except Exception:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(None)
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    except Exception as e:
        logger.warning(f'yfinance Fehler {symbol}: {e}')
        return None


def get_df(symbol: str) -> Optional[pd.DataFrame]:
    if symbol in CRYPTO_SYMBOLS:
        return fetch_crypto_df(symbol)
    return fetch_stock_df(symbol)


def get_current_price(symbol: str) -> Optional[float]:
    try:
        if symbol in CRYPTO_SYMBOLS:
            cg_id = CRYPTO_SYMBOLS[symbol]
            r = _SESSION.get(
                'https://api.coingecko.com/api/v3/simple/price',
                params={'ids': cg_id, 'vs_currencies': 'usd'},
                timeout=10,
            )
            r.raise_for_status()
            return float(r.json()[cg_id]['usd'])
        else:
            t = yf.Ticker(symbol)
            return float(t.fast_info.last_price)
    except Exception as e:
        logger.warning(f'Preisfehler {symbol}: {e}')
        return None


# ── TradingBot ───────────────────────────────────────────────────────────────

class TradingBot:
    MAX_POSITION_PCT = 0.10
    STOP_LOSS_PCT    = 0.03
    TAKE_PROFIT_PCT  = 0.06
    MAX_POSITIONS    = 6

    def __init__(self, mode: str = 'paper'):
        self.mode = mode
        self.running = False
        self.portfolio = load_portfolio()
        self._lock = threading.Lock()
        self._thread = None
        self.last_scan = None
        self.log_entries = []

    # ── Trade-Ausführung ────────────────────────────────────────────────────

    def _buy(self, symbol: str, price: float, signal_info: dict):
        with self._lock:
            if len(self.portfolio['positions']) >= self.MAX_POSITIONS:
                return
            if symbol in self.portfolio['positions']:
                return
            invest = self.portfolio['balance'] * self.MAX_POSITION_PCT
            if invest < 5:
                return
            qty = invest / price
            self.portfolio['balance'] -= invest
            self.portfolio['positions'][symbol] = {
                'symbol':            symbol,
                'name':              DISPLAY_NAMES.get(symbol, symbol),
                'qty':               qty,
                'entry_price':       price,
                'current_price':     price,
                'invested':          invest,
                'stop_loss':         price * (1 - self.STOP_LOSS_PCT),
                'take_profit':       price * (1 + self.TAKE_PROFIT_PCT),
                'unrealized_pnl':    0,
                'unrealized_pnl_pct': 0,
                'opened_at':         datetime.now(timezone.utc).isoformat(),
                'signal_strength':   signal_info.get('strength', 0),
                'reasons':           signal_info.get('reasons', []),
            }
            self.portfolio['trades'].insert(0, {
                'id':        len(self.portfolio['trades']) + 1,
                'symbol':    symbol,
                'name':      DISPLAY_NAMES.get(symbol, symbol),
                'type':      'BUY',
                'price':     price,
                'qty':       qty,
                'value':     invest,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'pnl':       0,
            })
            save_portfolio(self.portfolio)
            self._log(f'🟢 KAUF {symbol} @ {price:.4f} | {invest:.2f}€ investiert')

    def _sell(self, symbol: str, price: float, reason: str = 'Signal'):
        with self._lock:
            if symbol not in self.portfolio['positions']:
                return
            pos = self.portfolio['positions'].pop(symbol)
            proceeds  = pos['qty'] * price
            pnl       = proceeds - pos['invested']
            pnl_pct   = (pnl / pos['invested']) * 100
            self.portfolio['balance'] += proceeds
            self.portfolio['trades'].insert(0, {
                'id':        len(self.portfolio['trades']) + 1,
                'symbol':    symbol,
                'name':      DISPLAY_NAMES.get(symbol, symbol),
                'type':      'SELL',
                'price':     price,
                'qty':       pos['qty'],
                'value':     proceeds,
                'pnl':       round(pnl, 2),
                'pnl_pct':   round(pnl_pct, 2),
                'reason':    reason,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })
            save_portfolio(self.portfolio)
            icon = '🟢' if pnl >= 0 else '🔴'
            self._log(f'{icon} VERKAUF {symbol} @ {price:.4f} | PnL: {pnl:+.2f}€ ({pnl_pct:+.1f}%) | {reason}')

    # ── Hauptschleife ────────────────────────────────────────────────────────

    def _process_symbol(self, symbol: str):
        df = get_df(symbol)
        if df is None or len(df) < 50:
            return
        signal_info = generate_signal(df)
        price = signal_info.get('price', 0)
        if not price or price <= 0:
            return

        with self._lock:
            pos = self.portfolio['positions'].get(symbol)

        if pos:
            if price <= pos['stop_loss']:
                self._sell(symbol, price, 'Stop Loss')
            elif price >= pos['take_profit']:
                self._sell(symbol, price, 'Take Profit')
            elif signal_info['signal'] == 'SELL':
                self._sell(symbol, price, 'Verkaufssignal')
        else:
            if signal_info['signal'] == 'BUY':
                self._buy(symbol, price, signal_info)

    def _run(self):
        symbols = list(ALL_SYMBOLS.keys())
        while self.running:
            self._log(f'🔍 Scanning {len(symbols)} Assets...')
            for symbol in symbols:
                if not self.running:
                    break
                try:
                    self._process_symbol(symbol)
                except Exception as e:
                    logger.error(f'Fehler bei {symbol}: {e}')
                time.sleep(2)   # kurze Pause zwischen Symbolen

            self.last_scan = datetime.now(timezone.utc).isoformat()
            self._update_pnl()

            # 5 Minuten warten
            for _ in range(300):
                if not self.running:
                    break
                time.sleep(1)

    def _update_pnl(self):
        with self._lock:
            symbols = list(self.portfolio['positions'].keys())
        for symbol in symbols:
            try:
                price = get_current_price(symbol)
                if price:
                    with self._lock:
                        pos = self.portfolio['positions'].get(symbol)
                        if pos:
                            pos['current_price']      = price
                            pos['unrealized_pnl']     = round((price - pos['entry_price']) * pos['qty'], 2)
                            pos['unrealized_pnl_pct'] = round(((price / pos['entry_price']) - 1) * 100, 2)
            except Exception:
                pass
        save_portfolio(self.portfolio)

    def _log(self, msg: str):
        entry = {'time': datetime.now(timezone.utc).isoformat(), 'msg': msg}
        self.log_entries.insert(0, entry)
        self.log_entries = self.log_entries[:100]
        logger.info(msg)

    # ── Steuerung ────────────────────────────────────────────────────────────

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._log('✅ Bot gestartet')

    def stop(self):
        self.running = False
        self._log('⛔ Bot gestoppt')

    def reset_portfolio(self):
        self.portfolio = {
            'balance': 10000.0,
            'initial_balance': 10000.0,
            'positions': {},
            'trades': [],
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        save_portfolio(self.portfolio)
        self._log('🔄 Portfolio zurückgesetzt')

    # ── API ──────────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        with self._lock:
            portfolio = json.loads(json.dumps(self.portfolio, default=str))

        balance         = portfolio['balance']
        positions       = portfolio.get('positions', {})
        trades          = portfolio.get('trades', [])
        positions_value = sum(
            p.get('current_price', p['entry_price']) * p['qty']
            for p in positions.values()
        )
        total_value  = balance + positions_value
        initial      = portfolio['initial_balance']
        total_pnl    = total_value - initial
        total_pnl_pct = (total_pnl / initial) * 100

        closed   = [t for t in trades if t['type'] == 'SELL']
        win_rate = 0
        if closed:
            wins     = sum(1 for t in closed if t.get('pnl', 0) >= 0)
            win_rate = round((wins / len(closed)) * 100, 1)

        return {
            'running':         self.running,
            'mode':            self.mode,
            'last_scan':       self.last_scan,
            'balance':         round(balance, 2),
            'positions_value': round(positions_value, 2),
            'total_value':     round(total_value, 2),
            'total_pnl':       round(total_pnl, 2),
            'total_pnl_pct':   round(total_pnl_pct, 2),
            'initial_balance': initial,
            'open_positions':  len(positions),
            'total_trades':    len(closed),
            'win_rate':        win_rate,
            'positions':       list(positions.values()),
            'trades':          trades[:50],
            'logs':            self.log_entries[:30],
        }

    def get_chart(self, symbol: str) -> dict:
        df = get_df(symbol)
        if df is None or df.empty:
            return {}
        return get_chart_data(df)

    def get_signal(self, symbol: str) -> dict:
        df = get_df(symbol)
        if df is None or df.empty:
            return {'signal': 'HOLD', 'error': 'Keine Daten'}
        return generate_signal(df)
