"""
Trading Engine – Paper Trading
Datenquelle: Yahoo Finance v8 (Krypto + Aktien, funktioniert von überall)
"""
import json
import os
import time
import threading
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import requests

from indicators import generate_signal, get_chart_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'portfolio.json')

# ── HTTP-Session ─────────────────────────────────────────────────────────────
_SESSION = requests.Session()
_SESSION.headers.update({
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept': '*/*',
    'Referer': 'https://finance.yahoo.com',
})

def _warm_session():
    """Yahoo Finance Cookie holen (einmalig beim Start)"""
    try:
        _SESSION.get('https://finance.yahoo.com', timeout=10)
        logger.info('Yahoo Finance Session initialisiert')
    except Exception as e:
        logger.warning(f'Session-Init Fehler: {e}')

# ── Symbol-Definitionen ──────────────────────────────────────────────────────
CRYPTO_SYMBOLS = {
    'BTC-USD':  'Bitcoin',
    'ETH-USD':  'Ethereum',
    'SOL-USD':  'Solana',
    'XRP-USD':  'XRP',
    'ADA-USD':  'Cardano',
    'DOGE-USD': 'Dogecoin',
    'AVAX-USD': 'Avalanche',
    'LINK-USD': 'Chainlink',
    'DOT-USD':  'Polkadot',
    'LTC-USD':  'Litecoin',
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

DISPLAY_NAMES = {**CRYPTO_SYMBOLS, **STOCK_SYMBOLS}
ALL_SYMBOLS   = list(CRYPTO_SYMBOLS.keys()) + list(STOCK_SYMBOLS.keys())

# ── Datenabruf ───────────────────────────────────────────────────────────────

def fetch_df(symbol: str, days: int = 90) -> Optional[pd.DataFrame]:
    """Yahoo Finance v8 – funktioniert für Krypto (BTC-USD) + Aktien (AAPL)"""
    end_ts   = int(time.time())
    start_ts = end_ts - days * 86400
    try:
        r = _SESSION.get(
            f'https://query2.finance.yahoo.com/v8/finance/chart/{symbol}',
            params={
                'period1':       start_ts,
                'period2':       end_ts,
                'interval':      '1d',
                'includePrePost': 'False',
            },
            timeout=15,
        )
        r.raise_for_status()
        data   = r.json()
        result = data['chart']['result'][0]
        ts     = result['timestamp']       # singular – Yahoo's key
        q      = result['indicators']['quote'][0]

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(ts, unit='s'),
            'open':   [float(x) if x else None for x in q.get('open',   [])],
            'high':   [float(x) if x else None for x in q.get('high',   [])],
            'low':    [float(x) if x else None for x in q.get('low',    [])],
            'close':  [float(x) if x else None for x in q.get('close',  [])],
            'volume': [float(x) if x else 0    for x in q.get('volume', [])],
        })
        df = df.dropna(subset=['close']).reset_index(drop=True)
        return df if not df.empty else None
    except Exception as e:
        logger.warning(f'Yahoo Fehler {symbol}: {e}')
        return None


def get_df(symbol: str) -> Optional[pd.DataFrame]:
    return fetch_df(symbol)


def get_current_price(symbol: str) -> Optional[float]:
    try:
        df = fetch_df(symbol, days=3)
        if df is not None and not df.empty:
            return float(df['close'].iloc[-1])
    except Exception as e:
        logger.warning(f'Preisfehler {symbol}: {e}')
    return None


# ── Portfolio I/O ────────────────────────────────────────────────────────────

def load_portfolio() -> dict:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        'balance':         10000.0,
        'initial_balance': 10000.0,
        'positions':       {},
        'trades':          [],
        'created_at':      datetime.now(timezone.utc).isoformat(),
    }


def save_portfolio(portfolio: dict):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2, default=str)


# ── TradingBot ───────────────────────────────────────────────────────────────

class TradingBot:
    MAX_POSITION_PCT = 0.10
    STOP_LOSS_PCT    = 0.03
    TAKE_PROFIT_PCT  = 0.06
    MAX_POSITIONS    = 6

    def __init__(self, mode: str = 'paper'):
        self.mode      = mode
        self.running   = False
        self.portfolio = load_portfolio()
        self._lock     = threading.Lock()
        self._thread   = None
        self.last_scan = None
        self.log_entries = []
        # Session beim Start aufwärmen
        threading.Thread(target=_warm_session, daemon=True).start()

    # ── Trades ───────────────────────────────────────────────────────────────

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
                'symbol':             symbol,
                'name':               DISPLAY_NAMES.get(symbol, symbol),
                'qty':                qty,
                'entry_price':        price,
                'current_price':      price,
                'invested':           invest,
                'stop_loss':          price * (1 - self.STOP_LOSS_PCT),
                'take_profit':        price * (1 + self.TAKE_PROFIT_PCT),
                'unrealized_pnl':     0.0,
                'unrealized_pnl_pct': 0.0,
                'opened_at':          datetime.now(timezone.utc).isoformat(),
                'signal_strength':    signal_info.get('strength', 0),
                'reasons':            signal_info.get('reasons', []),
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
            self._log(f'🟢 KAUF {symbol} @ {price:.4f} | {invest:.2f}€')

    def _sell(self, symbol: str, price: float, reason: str = 'Signal'):
        with self._lock:
            if symbol not in self.portfolio['positions']:
                return
            pos      = self.portfolio['positions'].pop(symbol)
            proceeds = pos['qty'] * price
            pnl      = proceeds - pos['invested']
            pnl_pct  = (pnl / pos['invested']) * 100
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

    # ── Hauptschleife ─────────────────────────────────────────────────────────

    def _process_symbol(self, symbol: str):
        df = get_df(symbol)
        if df is None or len(df) < 20:
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
        while self.running:
            self._log(f'🔍 Scanning {len(ALL_SYMBOLS)} Assets...')
            for symbol in ALL_SYMBOLS:
                if not self.running:
                    break
                try:
                    self._process_symbol(symbol)
                except Exception as e:
                    logger.error(f'Fehler bei {symbol}: {e}')
                time.sleep(1)
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

    # ── Steuerung ──────────────────────────────────────────────────────────

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
            'balance': 10000.0, 'initial_balance': 10000.0,
            'positions': {}, 'trades': [],
            'created_at': datetime.now(timezone.utc).isoformat(),
        }
        save_portfolio(self.portfolio)
        self._log('🔄 Portfolio zurückgesetzt')

    # ── API ────────────────────────────────────────────────────────────────

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
        total_value   = balance + positions_value
        initial       = portfolio['initial_balance']
        total_pnl     = total_value - initial
        total_pnl_pct = (total_pnl / initial) * 100
        closed        = [t for t in trades if t['type'] == 'SELL']
        win_rate      = 0
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
