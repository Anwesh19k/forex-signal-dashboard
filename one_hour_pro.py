import pandas as pd
import numpy as np
import requests
import streamlit as st
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

API_KEY = 'f3fbe8d7f5a4462e97b279dc2557f559'
INTERVAL = '1h'
SYMBOLS = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP']


def time_until_next_hour():
    now = datetime.now()
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    return str(next_hour - now).split(".")[0]


@st.cache_data(ttl=3600)
def fetch_data(symbol):
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={INTERVAL}&outputsize=300&apikey={API_KEY}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df.sort_values('datetime')
    except:
        return pd.DataFrame()


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(period).mean()
    avg_loss = pd.Series(loss).rolling(period).mean()
    rs = avg_gain / (avg_loss + 1e-6)
    return 100 - (100 / (1 + rs))


def compute_macd(df):
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def compute_adx(df, period=14):
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = np.where((high.diff() > low.diff()) & (high.diff() > 0), high.diff(), 0)
    minus_dm = np.where((low.diff() > high.diff()) & (low.diff() > 0), low.diff(), 0)
    tr = np.maximum.reduce([high - low, abs(high - close.shift()), abs(low - close.shift())])
    atr = pd.Series(tr).rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / (atr + 1e-6)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / (atr + 1e-6)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-6)) * 100
    return pd.Series(dx).rolling(window=period).mean()


def add_features(df):
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ema10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['rsi14'] = compute_rsi(df['close'])
    df['momentum'] = df['close'] - df['close'].shift(4)
    df['macd'] = compute_macd(df)
    df['adx'] = compute_adx(df)
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    df['volatility'] = df['high'] - df['low']
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    return df.dropna()


@st.cache_resource
def train_cached_model(df):
    features = ['ma5', 'ma10', 'ema10', 'rsi14', 'momentum', 'macd', 'adx', 'bb_upper', 'bb_lower', 'volatility']
    X = df[features]
    y = df['target']

    if y.value_counts().min() < 10:
        return None, 0

    df_combined = pd.concat([X, y], axis=1)
    df_1 = df_combined[df_combined['target'] == 1]
    df_0 = df_combined[df_combined['target'] == 0]
    df_balanced = pd.concat([
        resample(df_1, replace=True, n_samples=min(len(df_1), len(df_0)), random_state=42),
        resample(df_0, replace=True, n_samples=min(len(df_1), len(df_0)), random_state=42)
    ])
    df_balanced = df_balanced.sample(frac=1, random_state=42)
    X = df_balanced[features]
    y = df_balanced['target']

    tscv = TimeSeriesSplit(n_splits=5)
    acc_scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                              use_label_encoder=False, eval_metric='logloss', verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, preds))

    final_model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                use_label_encoder=False, eval_metric='logloss', verbosity=0)
    final_model.fit(X, y)

    return final_model, np.mean(acc_scores)


def predict_signal(symbol, df, model):
    latest = df[['ma5', 'ma10', 'ema10', 'rsi14', 'momentum', 'macd', 'adx', 'bb_upper', 'bb_lower',
                 'volatility']].iloc[-1:]
    row = df.iloc[-1]
    pred = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0]
    signal = "BUY 📈" if pred == 1 else "SELL 📉"

    rsi = row['rsi14']
    ema_cross = row['ema10'] > row['ma10']
    momentum = row['momentum'] > 0
    macd = row['macd'] > 0
    adx_strong = row['adx'] > 20
    bb_signal = row['close'] < row['bb_lower'] if pred == 1 else row['close'] > row['bb_upper']
    confidence_score = sum([ema_cross, momentum, macd, adx_strong, bb_signal])
    label = "✅ Strong" if confidence_score >= 4 else "⚠️ Weak"

    multiplier = 100
    return [
        symbol,
        str(row['datetime']),
        signal,
        f"{proba[0]:.2f}",
        f"{proba[1]:.2f}",
        f"{rsi:.1f}",
        label,
        f"{row['close'] * multiplier:.2f}"
    ]


def run_signal_engine():
    headers = ["Symbol", "Timestamp", "Signal", "Prob SELL", "Prob BUY", "RSI", "Confidence", "Scaled Close"]

    def process_symbol(symbol):
        df = fetch_data(symbol)
        if df.empty or len(df) < 100:
            return [symbol, "-", "❌ Insufficient data", "-", "-", "-", "-", "-"]

        df = add_features(df)
        if len(df) < 100:
            return [symbol, "-", "⚠️ Not enough data", "-", "-", "-", "-", "-"]

        model, acc = train_cached_model(df)
        if model is None or acc < 0.7:
            return [symbol, "-", "⚠️ Model skipped/low acc", "-", "-", "-", "-", "-"]

        return predict_signal(symbol, df, model)

    with ThreadPoolExecutor() as executor:
        table = list(executor.map(process_symbol, SYMBOLS))

    return pd.DataFrame(table, columns=headers)

