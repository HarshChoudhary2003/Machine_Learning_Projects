import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import load_model

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Tesla Stock Price Prediction")
st.caption(
    "Deep Learning based Time-Series Forecasting using SimpleRNN and LSTM models"
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("TSLA.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df[["Close"]]

data = load_data()

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["LSTM", "SimpleRNN"]
)

prediction_days = st.sidebar.selectbox(
    "Prediction Horizon (Days)",
    [1, 5, 10]
)

window_map = {1: 30, 5: 60, 10: 90}
window_size = window_map[prediction_days]

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
data["MA20"] = data["Close"].rolling(20).mean()
data["MA50"] = data["Close"].rolling(50).mean()

# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[["Close"]])

# --------------------------------------------------
# Create Sequences
# --------------------------------------------------
def create_sequences(data, window):
    X = []
    for i in range(window, len(data)):
        X.append(data[i - window : i])
    return np.array(X)

X = create_sequences(scaled_data, window_size)
X = X.reshape(X.shape[0], X.shape[1], 1)

# --------------------------------------------------
# Load Model (compile=False FIX)
# --------------------------------------------------
model_path = (
    "models/lstm_model.h5"
    if model_choice == "LSTM"
    else "models/rnn_model.h5"
)

model = load_model(model_path, compile=False)

# --------------------------------------------------
# Predictions
# --------------------------------------------------
pred_scaled = model.predict(X)
predicted_prices = scaler.inverse_transform(pred_scaled)
actual_prices = data["Close"].values[window_size:]

# --------------------------------------------------
# Metrics
# --------------------------------------------------
rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))

# --------------------------------------------------
# Future Dates
# --------------------------------------------------
last_date = data.index[-1]
future_dates = [
    last_date + timedelta(days=i)
    for i in range(1, len(predicted_prices) + 1)
]

# --------------------------------------------------
# Confidence Band
# --------------------------------------------------
upper_band = predicted_prices.flatten() + rmse
lower_band = predicted_prices.flatten() - rmse

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Visualization", "📈 Results", "📥 Data"]
)

# --------------------------------------------------
# TAB 1: Visualization
# --------------------------------------------------
with tab1:
    st.subheader("Actual vs Predicted Prices with Confidence Band")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        data.index,
        data["Close"],
        label="Actual Price",
        color="blue"
    )

    ax.plot(
        future_dates,
        predicted_prices,
        label="Predicted Price",
        color="orange"
    )

    ax.fill_between(
        future_dates,
        lower_band,
        upper_band,
        color="orange",
        alpha=0.25,
        label="Confidence Band"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()

    st.pyplot(fig)

# --------------------------------------------------
# TAB 2: Results & Model Comparison
# --------------------------------------------------
with tab2:
    st.subheader("Prediction Summary")

    st.metric("RMSE", f"{rmse:.2f}")

    final_price = predicted_prices[-1][0]
    st.success(
        f"Predicted Tesla Closing Price after "
        f"{prediction_days} day(s): "
        f"**${final_price:.2f}**"
    )

    st.subheader("Model Comparison")

    try:
        rnn_model = load_model(
            "models/rnn_model.h5",
            compile=False
        )
        rnn_pred = rnn_model.predict(X)
        rnn_pred = scaler.inverse_transform(rnn_pred)
        rnn_rmse = np.sqrt(
            mean_squared_error(actual_prices, rnn_pred)
        )

        comparison_df = pd.DataFrame({
            "Model": ["LSTM", "SimpleRNN"],
            "RMSE": [rmse, rnn_rmse]
        })

        st.bar_chart(comparison_df.set_index("Model"))

    except:
        st.info(
            "SimpleRNN model not available. "
            "Only LSTM evaluated."
        )

# --------------------------------------------------
# TAB 3: Data Download
# --------------------------------------------------
with tab3:
    result_df = pd.DataFrame({
        "Actual Price": actual_prices.flatten(),
        "Predicted Price": predicted_prices.flatten()
    })

    st.dataframe(result_df.tail(20))

    csv = result_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Prediction Data (CSV)",
        data=csv,
        file_name="tesla_stock_predictions.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption(
    "⚠️ Educational project only. "
    "This application is not financial advice."
)
