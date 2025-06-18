import pandas as pd
import numpy as np
import requests
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils import resample
from datetime import datetime

# === Config ===
API_KEYS = [
    '54a7479bdf2040d3a35d6b3ae6457f9d',
    '09c09d58ed5e4cf4afd9a9cac8e09b5d',
    'df00920c02c54a59a426948a47095543'
]
api_usage_index = 0
INTERVAL = '1h'
SYMBOLS = ['EUR/USD', 'USD/JPY', 'GBP/USD', 'USD/CHF', 'AUD/USD', 'USD/CAD', 'NZD/USD', 'EUR/GBP']
MULTIPLIER = 100
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
        url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={INTERVAL}&outputsize=500&apikey={get_next_api_key()}"
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
    df['target'] = np.where((df['close'].shift(-1) - df['close']) / df['close'] > 0.001, 1, 0)  # 0.1% gain = BUY
    return df.dropna()

def train_model(df):
    features = ['ma5', 'ma10', 'ema10', 'rsi14', 'momentum', 'macd', 'adx', 'bb_upper', 'bb_lower', 'volatility']
    X = df[features]
    y = df['target']

    if y.value_counts().min() < 5:  # reduced to 5
        return None, 0

    df_1 = df[df['target'] == 1]
    df_0 = df[df['target'] == 0]
    df_bal = pd.concat([
        resample(df_1, replace=True, n_samples=min(len(df_1), len(df_0)), random_state=42),
        resample(df_0, replace=True, n_samples=min(len(df_1), len(df_0)), random_state=42)
    ]).sample(frac=1, random_state=42)

    X_bal = df_bal[features]
    y_bal = df_bal['target']

    acc_scores = []
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X_bal):
        X_train, X_test = X_bal.iloc[train_idx], X_bal.iloc[test_idx]
        y_train, y_test = y_bal.iloc[train_idx], y_bal.iloc[test_idx]
        model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                              use_label_encoder=False, eval_metric='logloss', verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc_scores.append(accuracy_score(y_test, preds))

    final_model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                use_label_encoder=False, eval_metric='logloss', verbosity=0)
    final_model.fit(X_bal, y_bal)
    return final_model, np.mean(acc_scores)

def predict_signal(symbol, df, model):
    latest = df[['ma5', 'ma10', 'ema10', 'rsi14', 'momentum', 'macd', 'adx', 'bb_upper', 'bb_lower', 'volatility']].iloc[-1:]
    row = df.iloc[-1]
    pred = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0]
    signal = "BUY 📈" if pred == 1 else "SELL 🖉"

    # Confidence assessment
    confidence_factors = [
        row['ema10'] > row['ma10'],
        row['momentum'] > 0,
        row['macd'] > 0,
        row['adx'] > 20,
        (row['close'] < row['bb_lower']) if pred == 1 else (row['close'] > row['bb_upper'])
    ]
    label = "✅ Strong" if sum(confidence_factors) >= 4 else "⚠️ Weak"

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
    table = []
    for symbol in SYMBOLS:
        df = fetch_data(symbol)
        if df.empty or len(df) < 100:
            table.append([symbol, "-", "❌ Insufficient data", "-", "-", "-", "-", "-"])
            continue

        df = add_features(df)
        if len(df) < 100:
            table.append([symbol, "-", "⚠️ Not enough features", "-", "-", "-", "-", "-"])
            continue

        model, acc = train_model(df)
        if model is None or acc < 0.65:
            table.append([symbol, "-", f"⚠️ Model skipped (acc={acc:.2f})", "-", "-", "-", "-", "-"])
            continue

        signal_row = predict_signal(symbol, df, model)
        table.append(signal_row)

    return pd.DataFrame(table, columns=headers)

