import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import streamlit as st
import os

st.title("Spindle Runout Predictive Maintenance")

st.sidebar.header("User Input Parameters")
target_nose = st.sidebar.number_input("Target Nose Runout Value", value=0.10)
target_300 = st.sidebar.number_input("Target 300mm Runout Value", value=0.20)
time_steps = st.sidebar.number_input("Time Steps for LSTM", value=3, min_value=1)

nose_input = st.sidebar.text_area("Enter Nose Runout Values (comma-separated)", "")
dist_300_input = st.sidebar.text_area("Enter 300mm Runout Values (comma-separated)", "")

def process_input(input_str):
    try:
        values = np.array([float(x) for x in input_str.split(',') if x.strip()]).reshape(-1, 1)
        return values
    except ValueError:
        st.error("Invalid input! Ensure all values are numerical and comma-separated.")
        st.stop()

if nose_input and dist_300_input:
    nose_values = process_input(nose_input)
    dist_300_values = process_input(dist_300_input)
else:
    st.warning("Please enter values for both Nose and 300mm Runout.")
    st.stop()

if len(nose_values) <= time_steps or len(dist_300_values) <= time_steps:
    st.error(f"Insufficient data! Enter at least {time_steps + 1} values for meaningful predictions.")
    st.stop()

scaler_nose = MinMaxScaler(feature_range=(0, 1))
scaler_300 = MinMaxScaler(feature_range=(0, 1))
nose_values_scaled = scaler_nose.fit_transform(nose_values)
dist_300_values_scaled = scaler_300.fit_transform(dist_300_values)

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    if len(X) == 0:
        return np.zeros((1, time_steps, 1)), np.zeros((1, 1))
    return np.array(X), np.array(y)

X_nose, y_nose = create_sequences(nose_values_scaled, time_steps)
X_300, y_300 = create_sequences(dist_300_values_scaled, time_steps)

@st.cache_resource
def get_trained_model(X, y, model_path):
    if os.path.exists(model_path):
        return load_model(model_path, custom_objects={'mse': tf.keras.losses.MeanSquaredError()})
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)
    model.save(model_path)
    return model

model_nose = get_trained_model(X_nose, y_nose, "lstm_nose_model.h5")
model_300 = get_trained_model(X_300, y_300, "lstm_300_model.h5")

def predict_values(model, X, scaler, target_value):
    if len(X) == 0:
        return np.array([])
    predictions = list(X[-1].flatten())
    while scaler.inverse_transform(np.array([[predictions[-1]]]))[0, 0] < target_value:
        last_sequence = np.array(predictions[-time_steps:]).reshape(1, time_steps, 1)
        next_value = model.predict(last_sequence, verbose=0)[0, 0]
        predictions.append(next_value)
        if len(predictions) > 1000:
            break
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

if st.sidebar.button("Run Predictions"):
    st.write("Processing predictions...")
    progress_bar = st.progress(0)

    nose_predictions = predict_values(model_nose, X_nose, scaler_nose, target_nose)
    progress_bar.progress(50)
    dist_300_predictions = predict_values(model_300, X_300, scaler_300, target_300)
    progress_bar.progress(100)

    st.subheader("Predicted Runout Values")
    predictions_df = pd.DataFrame({
        "Nose Runout Predictions": nose_predictions,
        "300mm Runout Predictions": dist_300_predictions[:len(nose_predictions)]
    })
    st.dataframe(predictions_df)

    st.subheader("Nose Runout Prediction Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(nose_predictions, label='Nose Runout Prediction', color='blue')
    ax.axhline(y=target_nose, color='r', linestyle='--', label=f'Target ({target_nose})')
    ax.set_title("Spindle Runout Prediction at Nose")
    ax.legend()
    st.pyplot(fig)

    st.subheader("300mm Runout Prediction Plot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dist_300_predictions, label='300mm Runout Prediction', color='green')
    ax.axhline(y=target_300, color='r', linestyle='--', label=f'Target ({target_300})')
    ax.set_title("Spindle Runout Prediction at 300mm")
    ax.legend()
    st.pyplot(fig)