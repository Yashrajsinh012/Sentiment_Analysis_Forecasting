import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Dropout
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

import random
import tensorflow as tf
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)


'''Load JSONL'''
def load_jsonl(path):
    with open(path) as f:
        return pd.DataFrame([json.loads(line) for line in f])

train_df = load_jsonl("train.jsonl")
test_df = load_jsonl("test.jsonl")

'''Convert timestamps to datetime and sort'''
for df in [train_df, test_df]:
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.sort_values("timestamp", inplace=True)

SEQUENCE_LENGTH = 10
TARGET_COLUMNS = ["pos_ratio", "neg_ratio", "neu_ratio", "avg_sentiment"]


 '''Helper to create sequences'''
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

full_df = pd.concat([train_df, test_df], axis=0).sort_values("timestamp").reset_index(drop=True)

'''Final Forecast for Next 10 Days'''
future_predictions = {}

for target in TARGET_COLUMNS:
    print(f"\n============================")
    print(f"Final Training + Forecasting 10 Days Ahead: {target}")
    print(f"============================")

    '''Scale the full target column'''
    scaler = MinMaxScaler()
    full_scaled = scaler.fit_transform(full_df[[target]])

    '''Create sequences'''
    X_full, y_full = create_sequences(full_scaled, SEQUENCE_LENGTH)
    X_full = X_full.reshape((X_full.shape[0], SEQUENCE_LENGTH, 1))

    '''Build model'''
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(SEQUENCE_LENGTH, 1)),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=Huber(), metrics=['mae'])

    '''Train model on all data'''
    model.fit(X_full, y_full, epochs=100, batch_size=16, verbose=0)

    '''Rolling prediction for next 10 time steps'''
    input_seq = full_scaled[-SEQUENCE_LENGTH:]
    future_scaled_preds = []

    for _ in range(10):
        input_seq_reshaped = input_seq.reshape((1, SEQUENCE_LENGTH, 1))
        pred = model.predict(input_seq_reshaped, verbose=0)
        future_scaled_preds.append(pred[0][0])
        input_seq = np.vstack([input_seq[1:], pred])

    '''Inverse scale'''
    future_scaled_preds = np.array(future_scaled_preds).reshape(-1, 1)
    future_preds = scaler.inverse_transform(future_scaled_preds).flatten()

    '''Future timestamps'''
    last_date = full_df["timestamp"].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=10, freq='D')

    '''Store predictions'''
    future_predictions[target] = future_preds
    if "timestamp" not in future_predictions:
        future_predictions["timestamp"] = future_dates

    '''Plot forecast'''
    plt.figure(figsize=(12, 4))
    plt.plot(future_dates, future_preds, label="Forecast", marker='o')
    plt.xlabel("Date")
    plt.ylabel(target)
    plt.title(f"10-Day Forecast: {target}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

'''Results'''
forecast_df = pd.DataFrame(future_predictions)
forecast_df.to_json("10_day_forecast.jsonl", orient="records", lines=True)
print("Forecast saved to '10_day_forecast.jsonl'")

# the data of the past and forecasted sentiment scores are concatenated
''' HMM '''
df_combined = pd.read_json(r"combined_past_and_forecast.jsonl", lines  = True)
scaler = StandardScaler()
X = scaler.fit_transform(df_combined[['pos_ratio', 'neu_ratio', 'neg_ratio']].values)
n_states = 3

# Fit 
hmm_model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=1000, random_state=42, init_params='kmeans')
hmm_model.fit(X)
# Predict 
df_combined['regime'] = hmm_model.predict(X)
df = df_combined.copy()
df = df.sort_values('timestamp')

# Plot
data = df['avg_sentiment'].values
regimes = df['regime'].values
timestamps = df['timestamp'].values
plt.figure(figsize=(16, 6))
colors = ['red', 'green', 'blue']
labels = {0: 'Declining', 1: 'Neutral', 2: 'Optimistic'}

for i in range(3):  
    mask = (regimes == i)
    plt.plot(np.where(mask)[0], data[mask], '.', label=f'{labels[i]}', color=colors[i])

plt.plot(data, color='black', alpha=0.3, label='Avg Sentiment')
plt.title("HMM Regime Detection on Avg Sentiment")
plt.xlabel("Time Index")
plt.ylabel("Avg Sentiment")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
