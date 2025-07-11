# crypto_bot.py – All-in-One Crypto Trading Bot

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import entropy
from binance.client import Client

# ===== CONFIG SECTION =====
# Replace with your Binance US API keys or use Streamlit Secrets
try:
    from config import BINANCE_API_KEY, BINANCE_API_SECRET
except ImportError:
    import streamlit as st
    BINANCE_API_KEY = st.secrets.get(l0CunQm66vMeHwjpxM5RSKkyOC1lBAAftB7VGp0iPOwBwW6fl3lXXDhIy1THHGhu)
    BINANCE_API_SECRET = st.secrets.get(wNyiS1bI1OyuAFO97kDSJpjSpXB7jE30dOLzhECTxb5RohniGKng5buQNAEihSl6)

# ===== BINANCE CONNECTION =====
client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET, tld='us')

# ===== DATA FETCHING =====
def get_ohlc(symbol, interval='1m', limit=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].astype(float)
    return df

def get_all_symbols():
    exchange_info = client.get_exchange_info()
    symbols = [s['symbol'] for s in exchange_info['symbols'] if 'USDT' in s['symbol']]
    return symbols

# ===== PATTERN DETECTION =====
def detect_fft_pattern(prices):
    fft_result = np.fft.fft(prices)
    freq = np.fft.fftfreq(len(prices))
    dominant_freq = freq[np.argmax(np.abs(fft_result[1:len(freq)//2])]  # Fixed
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
    symbols = get_all_symbols()

    for symbol in symbols:
        try:
            ticker = client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])

            stats = client.get_ticker(symbol=symbol)
            volume_change = float(stats['priceChangePercent'])
            volume_24h = float(stats['quoteVolume'])

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

# ===== STREAMLIT DASHBOARD =====
st.set_page_config(page_title="🧠 Crypto Pattern Detection Bot", layout="wide")
st.title("🧠 Quantitative Crypto Trading Bot")
st.markdown("Uses advanced math to detect early price movements and low-cap coins with potential.")

# BTC Signal
btc_df = get_ohlc('BTCUSDT')
btc_signal = get_pattern_signal(btc_df['close'].values)

st.subheader("📈 BTC/USDT Signal")
col1, col2, col3 = st.columns(3)
col1.metric("Trend", btc_signal['trend'])
col2.metric("Entropy Level", btc_signal['entropy_level'])
col3.metric("Dominant Frequency", f"{btc_signal['dominant_frequency']:.2f}")

# Low-cap Coin Scanner
st.subheader("🚀 Low-Cap Coin Candidates")
lowcap_coins = scan_lowcap_coins()
if lowcap_coins:
    st.table(pd.DataFrame(lowcap_coins))
else:
    st.info("No low-cap coins found with strong volume spikes yet.")
