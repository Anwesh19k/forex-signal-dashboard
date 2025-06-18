import pandas as pd
import numpy as np
import requests
from xgboost import XGBClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit

# === CONFIG ===
API_KEYS = ['your_key_1', 'your_key_2', 'your_key_3']  # Add your API keys
INTERVAL = '1h'
SYMBOLS = ['EUR/USD']
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
    key = get_next_api_key()
    url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={INTERVAL}&outputsize=300&apikey={key}"
    try:
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
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
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
    df['ema10'] = df['close'].ewm(span=10).mean()
    df['rsi14'] = compute_rsi(df['close'])
    df['momentum'] = df['close'] - df['close'].shift(4)
    df['macd'] = compute_macd(df)
    df['adx'] = compute_adx(df)
    df['bb_upper'] = df['close'].rolling(20).mean() + 2 * df['close'].rolling(20).std()
    df['bb_lower'] = df['close'].rolling(20).mean() - 2 * df['close'].rolling(20).std()
    df['volatility'] = df['high'] - df['low']
    df['atr'] = df['volatility'].rolling(14).mean()
    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    return df.dropna()

def train_model(df):
    features = ['ma5', 'ma10', 'ema10', 'rsi14', 'momentum', 'macd', 'adx',
                'bb_upper', 'bb_lower', 'volatility']
    X, y = df[features], df['target']
    if y.value_counts().min() < 10:
        return None
    df_1 = df[df['target'] == 1]
    df_0 = df[df['target'] == 0]
    df_bal = pd.concat([
        resample(df_1, replace=True, n_samples=min(len(df_1), len(df_0))),
        resample(df_0, replace=True, n_samples=min(len(df_1), len(df_0)))
    ]).sample(frac=1)
    X, y = df_bal[features], df_bal['target']
    model = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                          use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def predict(df, model, symbol):
    features = ['ma5', 'ma10', 'ema10', 'rsi14', 'momentum', 'macd', 'adx',
                'bb_upper', 'bb_lower', 'volatility']
    last = df.iloc[-1]
    X_pred = df[features].iloc[[-1]]
    proba = model.predict_proba(X_pred)[0]
    signal = "BUY 📈" if proba[1] > 0.5 else "SELL 🖉"

    # Advanced confidence
    rsi_buy = last['rsi14'] < 30 and proba[1] > 0.6
    rsi_sell = last['rsi14'] > 70 and proba[0] > 0.6
    bb_signal = last['close'] < last['bb_lower'] if proba[1] > 0.5 else last['close'] > last['bb_upper']

    confidence = sum([
        last['ema10'] > last['ma10'],
        last['momentum'] > 0,
        last['macd'] > 0,
        last['adx'] > 25,
        rsi_buy if proba[1] > 0.5 else rsi_sell,
        bb_signal
    ])
    conf_label = "✅ Strong" if confidence >= 5 else "⚠️ Weak"

    price = round(last['close'], 4)
    atr = last['atr']
    tp = price + atr * 1.2 if signal == "BUY 📈" else price - atr * 1.2
    sl = price - atr if signal == "BUY 📈" else price + atr

    return {
        "Symbol": symbol,
        "Datetime": str(last['datetime']),
        "Signal": signal,
        "Prob BUY": round(proba[1], 2),
        "RSI": round(last['rsi14'], 1),
        "Confidence": conf_label,
        "Price x100": round(price * MULTIPLIER, 2),
        "Plan": f"{price} / TP: {round(tp, 4)} / SL: {round(sl, 4)}"
    }

def run_signal_engine():
    results = []
    for symbol in SYMBOLS:
        df = fetch_data(symbol)
        if df.empty or len(df) < 150:
            continue
        df = add_features(df)
        model = train_model(df)
        if model:
            res = predict(df, model, symbol)
            results.append(res)
    return pd.DataFrame(results)

# RUN
output = run_signal_engine()
print(output.to_markdown(index=False))

        
