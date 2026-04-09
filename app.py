"""
Flask Backend – Trading Bot API
"""
from flask import Flask, jsonify, request, render_template, send_from_directory
from flask_cors import CORS
import os

from bot import TradingBot

app = Flask(__name__)
CORS(app)

bot = TradingBot(mode="paper")

# ──────────────────────────────────────────
# Frontend
# ──────────────────────────────────────────

@app.route('/')
def index():
    from flask import make_response
    resp = make_response(render_template('index.html'))
    # Browser-Cache deaktivieren damit neue JS-Version geladen wird
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate'
    resp.headers['Pragma'] = 'no-cache'
    return resp

# ──────────────────────────────────────────
# API Endpunkte
# ──────────────────────────────────────────

@app.route('/api/status')
def api_status():
    return jsonify(bot.get_status())

@app.route('/api/start', methods=['POST'])
def api_start():
    bot.start()
    return jsonify({"ok": True, "running": bot.running})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    bot.stop()
    return jsonify({"ok": True, "running": bot.running})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    bot.stop()
    bot.reset_portfolio()
    return jsonify({"ok": True})

@app.route('/api/chart/<path:symbol>')
def api_chart(symbol):
    # Kein Umwandeln – Symbole wie BTC-USD behalten ihr Bindestrich
    data = bot.get_chart(symbol)
    return jsonify(data)

@app.route('/api/signal/<path:symbol>')
def api_signal(symbol):
    data = bot.get_signal(symbol)
    return jsonify(data)

@app.route('/api/symbols')
def api_symbols():
    return jsonify({
        "crypto": bot.CRYPTO_SYMBOLS,
        "stocks": bot.STOCK_SYMBOLS,
    })

@app.route('/api/config', methods=['POST'])
def api_config():
    data = request.json or {}
    if 'stop_loss' in data:
        bot.STOP_LOSS_PCT = float(data['stop_loss']) / 100
    if 'take_profit' in data:
        bot.TAKE_PROFIT_PCT = float(data['take_profit']) / 100
    if 'position_size' in data:
        bot.MAX_POSITION_PCT = float(data['position_size']) / 100

    # SL/TP bei allen offenen Positionen sofort aktualisieren
    updated = 0
    with bot._lock:
        for symbol, pos in bot.portfolio['positions'].items():
            entry = pos['entry_price']
            pos['stop_loss']   = round(entry * (1 - bot.STOP_LOSS_PCT), 6)
            pos['take_profit'] = round(entry * (1 + bot.TAKE_PROFIT_PCT), 6)
            updated += 1
    if updated:
        from bot import save_portfolio
        save_portfolio(bot.portfolio)

    return jsonify({"ok": True, "updated_positions": updated, "config": {
        "stop_loss":     round(bot.STOP_LOSS_PCT * 100, 1),
        "take_profit":   round(bot.TAKE_PROFIT_PCT * 100, 1),
        "position_size": round(bot.MAX_POSITION_PCT * 100, 1),
    }})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
