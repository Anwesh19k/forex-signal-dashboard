# === one_hour_pro.py ===
import pandas as pd
import numpy as np
import requests
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

API_KEYS = [
    '54a7479bdf2040d3a35d6b3ae6457f9d',
    'd162b35754ca4c54a13ebe7abecab4e0',
    'a7266b2503fd497496d47527a7e63b5d'
]
INTERVAL = '1h'
SYMBOLS = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP']
MULTIPLIER = 100
api_usage_index = 0
_cached_data = {}

def get_next_api_key():
    global api_usage_index
    key = API_KEYS[api_usage_index % len(API_KEYS)]
    api_usage_index += 1
    return key

def fetch_data(symbol):
    if symbol in _cached_data:
        return _cached_data[symbol]
    try:
        api_key = get_next_api_key()
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={INTERVAL}&outputsize=300&apikey={api_key}"
        r = requests.get(url, timeout=10)
        data = r.json()
        if "values" not in data:
            return pd.DataFrame()
        df = pd.DataFrame(data["values"])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime')
        _cached_data[symbol] = df
        return df
    except Exception as e:
        print(f"[ERROR] {symbol} - {e}")
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

def train_model(df):
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
    features = ['ma5', 'ma10', 'ema10', 'rsi14', 'momentum', 'macd', 'adx', 'bb_upper', 'bb_lower', 'volatility']
    latest = df[features].iloc[-1:]
    row = df.iloc[-1]
    pred = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0]
    signal = "BUY 📈" if pred == 1 else "SELL 🖉"

    confidence_score = sum([
        row['ema10'] > row['ma10'],
        row['momentum'] > 0,
        row['macd'] > 0,
        row['adx'] > 20,
        (row['close'] < row['bb_lower']) if pred == 1 else (row['close'] > row['bb_upper'])
    ])
    label = "✅ Strong" if confidence_score >= 4 else "⚠️ Weak"

    return [
        symbol,
        str(row['datetime']),
        signal,
        f"{proba[0]:.2f}",
        f"{proba[1]:.2f}",
        f"{row['rsi14']:.1f}",
        label,
        f"{row['close'] * MULTIPLIER:.2f}"
    ]

def run_signal_engine():
    headers = ["Symbol", "Timestamp", "Signal", "Prob SELL", "Prob BUY", "RSI", "Confidence", f"Price x{MULTIPLIER}"]

    def process(symbol):
        df = fetch_data(symbol)
        if df.empty or len(df) < 100:
            return [symbol, "-", "❌ Insufficient data", "-", "-", "-", "-", "-"]
        df = add_features(df)
        if len(df) < 100:
            return [symbol, "-", "⚠️ Not enough data", "-", "-", "-", "-", "-"]
        model, acc = train_model(df)
        if model is None or acc < 0.7:
            return [symbol, "-", f"⚠️ Model skipped (acc={acc:.2f})", "-", "-", "-", "-", "-"]
        return predict_signal(symbol, df, model)

    with ThreadPoolExecutor() as executor:
        table = list(executor.map(process, SYMBOLS))

    return pd.DataFrame(table, columns=headers)
