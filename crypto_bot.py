# crypto_bot.py – Math-Based Crypto Scanner with WebSocket (No Trade API Needed)

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import entropy
from binance.websockets import BinanceSocketManager
from binance.client import Client
import websocket
import json

# ===== CONFIG SECTION =====
# We're only using read-only WebSocket, so no API keys needed

# Create Binance client (only for symbol list)
client = Client("", "")  # No keys needed for public data

# ===== PATTERN DETECTION =====
def detect_fft_pattern(prices):
    fft_result = np.fft.fft(prices)
    freq = np.fft.fftfreq(len(prices))
    segment = fft_result[1:len(freq)//2]
    abs_fft = np.abs(segment)
    max_index = np.argmax(abs_fft)
    dominant_freq = freq[max_index]
    return dominant_freq

def calculate_entropy(prices):
    returns = np.diff(prices)
    hist, _ = np.histogram(returns, bins=10, density=True)
    return entropy(hist)

def detect_trend(prices):
    return "Up" if prices[-1] > prices[0] else "Down"

def get_pattern_signal(prices):
    ent = calculate_entropy(prices)
    trend = detect_trend(prices)
    dom_freq = detect_fft_pattern(prices)

    return {
        "dominant_frequency": dom_freq,
        "entropy": ent,
        "entropy_level": "Low" if ent < 1.5 else "High",
        "trend": trend
    }

# ===== LOW-CAP COIN SCANNER =====
def scan_lowcap_coins(min_market_cap=50_000_000):
    candidates = []
    symbols = [s['symbol'] for s in client.get_exchange_info()['symbols'] if 'USDT' in s['symbol']]
    
    for symbol in symbols:
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])

            stats = client.get_ticker(symbol=symbol)
            volume_change = float(stats['priceChangePercent'])
            market_cap = price * float(stats['weightedAvgPrice'])  # rough estimate

            if market_cap < min_market_cap and abs(volume_change) > 10:
                candidates.append({
                    'symbol': symbol,
                    'price': price,
                    'volume_change': volume_change,
                    'market_cap': market_cap
                })
        except Exception as e:
            continue

    return candidates

# ===== WEBSOCKET FOR BTC/USDT =====
BTC_PRICE_HISTORY = []

def on_message(ws, message):
    data = json.loads(message)
    price = float(data['k']['c'])  # Close price from kline update
    BTC_PRICE_HISTORY.append(price)
    if len(BTC_PRICE_HISTORY) > 100:
        BTC_PRICE_HISTORY.pop(0)

def start_websocket():
    bm = BinanceSocketManager(client)
    conn_key = bm.start_kline_socket('BTCUSDT', on_message, interval='1m')
    bm.start()

# Start WebSocket in background
import threading
threading.Thread(target=start_websocket, daemon=True).start()

# ===== STREAMLIT DASHBOARD =====
st.set_page_config(page_title="🧠 Crypto Pattern Detection Bot", layout="wide")
st.title("🧠 Quantitative Crypto Signal Detector")

st.markdown("Uses advanced math to detect early price movement and low-cap coins with potential.")

# BTC Signal
if len(BTC_PRICE_HISTORY) >= 30:
    btc_signal = get_pattern_signal(np.array(BTC_PRICE_HISTORY))
else:
    btc_signal = {"trend": "N/A", "entropy_level": "N/A", "dominant_frequency": 0}

st.subheader("📈 BTC/USDT Signal")
col1, col2, col3 = st.columns(3)
col1.metric("Trend", btc_signal['trend'])
col2.metric("Entropy Level", btc_signal['entropy_level'])
col3.metric("Dominant Frequency", f"{btc_signal.get('dominant_frequency', 0):.2f}")

# Low-cap Coin Scanner
st.subheader("🚀 Low-Cap Coin Candidates")
lowcap_coins = scan_lowcap_coins()
if lowcap_coins:
    st.table(pd.DataFrame(lowcap_coins))
else:
    st.info("No low-cap coins found with strong volume spikes yet.")

# Keep Dashboard Alive
st.write("📡 Waiting for live price updates...")
