# === Imports ===
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from concurrent.futures import ThreadPoolExecutor
from tabulate import tabulate

# === Configuration ===
API_KEY = '1d22dac65cd34ddbbc18ba3094668237'
INTERVAL = '1min'
SYMBOLS = ['EUR/USD', 'USD/JPY', 'GBP/USD']  # Add more if needed
CONFIDENCE_THRESHOLD = 0.70
MAX_WORKERS = 5

# === Technical Indicators ===
def compute_indicators(df):
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMA10'] = df['close'].ewm(span=10, adjust=False).mean()
    df['RSI'] = compute_rsi(df['close'], 14)
    df['Momentum'] = df['close'].diff(4)
    df.dropna(inplace=True)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(window=period).mean()
    ma_down = down.rolling(window=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# === Fetch Live Forex Data ===
def fetch_data(symbol):
    base_url = f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={INTERVAL}&apikey={API_KEY}&outputsize=200"
    response = requests.get(base_url)
    try:
        df = pd.DataFrame(response.json()['values'])
        df = df.rename(columns={'datetime': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.astype({'open': float, 'high': float, 'low': float, 'close': float})
        df = df.sort_values(by='timestamp')
        return compute_indicators(df)
    except Exception as e:
        print(f"[ERROR] Failed to fetch data for {symbol}: {e}")
        return None

# === Signal Generator ===
def generate_signal(symbol):
    df = fetch_data(symbol)
    if df is None or len(df) < 50:
        return None

    df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)
    X = df[['EMA5', 'EMA10', 'RSI', 'Momentum']]
    y = df['target']

    # === Cross-validation Accuracy Check ===
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    accuracies = []
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        accuracies.append(acc)

    avg_acc = np.mean(accuracies)

    # Reject models below accuracy threshold
    if avg_acc < CONFIDENCE_THRESHOLD:
        return {
            "Symbol": symbol,
            "Accuracy": f"{avg_acc:.2f}",
            "Signal": "REJECTED (Low Accuracy)",
            "Confidence": "N/A"
        }

    # Final training on all data
    model.fit(X, y)
    latest_features = X.iloc[-1:]
    prob = model.predict_proba(latest_features)[0]
    signal = "BUY" if prob[1] > prob[0] else "SELL"
    confidence = max(prob)

    return {
        "Symbol": symbol,
        "Accuracy": f"{avg_acc:.2f}",
        "Signal": signal,
        "Confidence": f"{confidence:.2f}"
    }

# === Parallel Execution ===
def run_all_signals():
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(generate_signal, SYMBOLS))
    results = [r for r in results if r]  # remove None
    print("\n" + tabulate(results, headers="keys", tablefmt="fancy_grid"))

# === Main Loop (Optional live loop) ===
if __name__ == "__main__":
    while True:
        print(f"\n=== Forex Signal Update - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        run_all_signals()
        
