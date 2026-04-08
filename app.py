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
    return render_template('index.html')

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
    symbol = symbol.replace('-', '/')
    data = bot.get_chart(symbol)
    return jsonify(data)

@app.route('/api/signal/<path:symbol>')
def api_signal(symbol):
    symbol = symbol.replace('-', '/')
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
    return jsonify({"ok": True, "config": {
        "stop_loss": round(bot.STOP_LOSS_PCT * 100, 1),
        "take_profit": round(bot.TAKE_PROFIT_PCT * 100, 1),
        "position_size": round(bot.MAX_POSITION_PCT * 100, 1),
    }})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
