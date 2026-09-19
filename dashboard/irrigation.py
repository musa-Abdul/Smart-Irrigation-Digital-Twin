import streamlit as st
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import joblib
import plotly.graph_objects as go

st.set_page_config(
    page_title="Smart Irrigation Digital Twin",
    page_icon="🌱",
    layout="wide"
)

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .main-title {
            font-size: 42px;
            font-weight: 700;
            margin-bottom: 0px;
        }

        .subtitle {
            font-size: 17px;
            opacity: 0.75;
            margin-top: 4px;
            margin-bottom: 25px;
        }

        .status-box {
            padding: 18px;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            min-height: 110px;
        }

        .status-label {
            font-size: 14px;
            opacity: 0.7;
            margin-bottom: 5px;
        }

        .status-value {
            font-size: 25px;
            font-weight: 600;
        }
    </style>

    <div class="main-title">
        🌱 Smart Irrigation Digital Twin
    </div>

    <div class="subtitle">
        Physics-guided soil-moisture forecasting,
        uncertainty-aware monitoring and irrigation decision support
    </div>
    """,
    unsafe_allow_html=True
)
# Load the processed field data

project_root = Path(__file__).resolve().parent.parent
model_path = (
    project_root
    / "models"
    / "physics_guided_lstm.pth"
)

feature_scaler_path = (
    project_root
    / "models"
    / "physics_feature_scaler.pkl"
)

target_scaler_path = (
    project_root
    / "models"
    / "physics_target_scaler.pkl"
)

feature_scaler = joblib.load(
    feature_scaler_path
)

target_scaler = joblib.load(
    target_scaler_path
)

checkpoint = torch.load(
    model_path,
    map_location="cpu",
    weights_only=False
)
st.sidebar.markdown("### Model Status")


st.sidebar.success("Saved forecasting model loaded")

with st.sidebar.expander("Model details"):
    st.write(f"Model inputs: {checkpoint['input_size']}")
    st.write(
        f"History window: {checkpoint['sequence_length']} observations"
    )
    st.write(
        f"Forecast horizon: {checkpoint['forecast_horizon_minutes']} minutes"
    )

data_path = (
    project_root
    / "data"
    / "processed"
    / "irrigation_model_data.csv"
)

data = pd.read_csv(data_path)

data["Timestamps"] = pd.to_datetime(
    data["Timestamps"]
)

data = (
    data
    .sort_values("Timestamps")
    .drop_duplicates(
        subset="Timestamps",
        keep="first"
    )
    .reset_index(drop=True)
)

st.markdown("### System Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Field Observations", f"{len(data):,}")
col2.metric("Soil Sensors", "6")
col3.metric("Forecast Horizon", "1 hour")
col4.metric("Prediction Interval", "90%")


st.markdown("### Soil Moisture Monitoring")

soil_ports = [
    "Port1",
    "Port2",
    "Port3",
    "Port4",
    "Port5",
    "Port6"
]

chart_end = data["Timestamps"].max()
chart_start = chart_end - pd.Timedelta(days=30)

soil_chart = (
    data.loc[
        data["Timestamps"].between(chart_start, chart_end),
        ["Timestamps"] + soil_ports
    ]
    .sort_values("Timestamps")
    .copy()
)


fig = go.Figure()

for port in soil_ports:
    fig.add_trace(
        go.Scatter(
            x=soil_chart["Timestamps"],
            y=soil_chart[port],
            mode="lines",
            name=port,
            line=dict(width=2)
        )
    )

fig.update_layout(
    height=430,
    margin=dict(l=10, r=10, t=20, b=10),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    ),
    xaxis_title=None,
    yaxis_title="Soil moisture"
)

fig.update_xaxes(
    range=[chart_start, chart_end],
    showgrid=False
)

fig.update_yaxes(
    range=[0.10, 0.40]
)

st.plotly_chart(
    fig,
    width="stretch",
    config={"displaylogo": False}
)

st.caption(
    f"Showing the most recent 30 days of recorded field data: "
    f"{chart_start:%d %b %Y} to {chart_end:%d %b %Y}"
)

st.markdown("### Latest Sensor State")

latest = data.iloc[-1]

sensor_cols = st.columns(6)

for i, port in enumerate(soil_ports):
    sensor_cols[i].metric(
        port,
        f"{latest[port]:.3f}"
    )

st.caption(
    f"Latest recorded observation: "
    f"{latest['Timestamps'].strftime('%d %b %Y, %H:%M')}"
)

st.markdown("### Weather and Water Inputs")

input_col1, input_col2, input_col3 = st.columns(3)

input_col1.metric(
    "Temperature",
    f"{latest['Temperature']:.2f}"
)

input_col2.metric(
    "Rainfall",
    f"{latest['Rainfall']:.3f}"
)

if pd.isna(latest["Irrigation"]):
    irrigation_display = "Missing"
else:
    irrigation_display = f"{latest['Irrigation']:.3f}"

input_col3.metric(
    "Recorded Irrigation",
    irrigation_display
)

class PhysicsInformedLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()

        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.fc = torch.nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])


forecast_model = PhysicsInformedLSTM(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"]
)

forecast_model.load_state_dict(
    checkpoint["model_state_dict"]
)

forecast_model.eval()

st.sidebar.success("Forecasting system ready")

physics_features = checkpoint["physics_features"]


model_data = data.copy()

model_data["Irrigation_missing"] = (
    model_data["Irrigation"].isna().astype(int)
)

model_data["Irrigation_filled"] = (
    model_data["Irrigation"].fillna(0)
)

time_gap = model_data["Timestamps"].diff()

model_data["continuous_group"] = (
    time_gap.ne(pd.Timedelta(minutes=15))
    .cumsum()
)

model_data["Rainfall_prev_1h"] = (
    model_data
    .groupby("continuous_group")["Rainfall"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=4,
            min_periods=4
        ).sum()
    )
)

model_data["Irrigation_prev_1h"] = (
    model_data
    .groupby("continuous_group")["Irrigation_filled"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=4,
            min_periods=4
        ).sum()
    )
)

model_data["water_input_prev_1h"] = (
    model_data["Rainfall_prev_1h"]
    + model_data["Irrigation_prev_1h"]
)

available_model_rows = model_data.dropna(
    subset=physics_features
).copy()

sequence_length = checkpoint["sequence_length"]

latest_group = available_model_rows[
    "continuous_group"
].iloc[-1]

latest_sequence = available_model_rows[
    available_model_rows["continuous_group"] == latest_group
].tail(sequence_length)


sequence_values = latest_sequence[
    physics_features
].to_numpy()

sequence_scaled = feature_scaler.transform(
    sequence_values
)

sequence_tensor = torch.tensor(
    sequence_scaled,
    dtype=torch.float32
).unsqueeze(0)

with torch.no_grad():
    predicted_scaled = forecast_model(
        sequence_tensor
    ).cpu().numpy()

predicted_delta = target_scaler.inverse_transform(
    predicted_scaled
)[0, 0]

current_port1 = latest_sequence["Port1"].iloc[-1]

predicted_port1 = (
    current_port1 + predicted_delta
)


uncertainty_margin = 0.0008818308522459128

forecast_lower = predicted_port1 - uncertainty_margin
forecast_upper = predicted_port1 + uncertainty_margin


st.markdown("### Soil Moisture Forecast")

forecast_col1, forecast_col2, forecast_col3 = st.columns(3)

forecast_col1.metric(
    "Current Port1",
    f"{current_port1:.3f}"
)

forecast_col2.metric(
    "Predicted in 1 Hour",
    f"{predicted_port1:.3f}",
    delta=f"{predicted_delta:+.6f}"
)

forecast_col3.metric(
    "90% Prediction Range",
    f"{forecast_lower:.3f} – {forecast_upper:.3f}"
)
forecast_fig = go.Figure()

forecast_fig.add_trace(
    go.Scatter(
        x=["Current", "1-hour forecast"],
        y=[current_port1, predicted_port1],
        mode="lines+markers",
        name="Port1 soil moisture",
        line=dict(width=3),
        marker=dict(size=10)
    )
)

forecast_fig.add_trace(
    go.Scatter(
        x=["1-hour forecast"],
        y=[predicted_port1],
        mode="markers",
        name="Forecast uncertainty",
        error_y=dict(
            type="data",
            symmetric=False,
            array=[forecast_upper - predicted_port1],
            arrayminus=[predicted_port1 - forecast_lower],
            visible=True
        ),
        marker=dict(size=10)
    )
)

forecast_fig.update_layout(
    height=300,
    margin=dict(l=10, r=10, t=30, b=10),
    yaxis=dict(
    title="Soil moisture",
    range=[0.18, 0.27]
),
    showlegend=True
)

st.plotly_chart(
    forecast_fig,
    width="stretch",
    config={"displaylogo": False}
)

if predicted_delta < 0:
    st.info(
        "The model expects a small decrease in Port1 soil moisture "
        "during the next hour."
    )
elif predicted_delta > 0:
    st.info(
        "The model expects an increase in Port1 soil moisture "
        "during the next hour."
    )
else:
    st.info(
        "The model expects Port1 soil moisture to remain approximately stable "
        "during the next hour."
    )

    st.markdown("### Irrigation Decision Support")

# Experimental operating references derived only from training data
theta_lower_reference = 0.180
theta_target = 0.249

decision_col1, decision_col2 = st.columns(2)

decision_col1.metric(
    "Operating Target",
    f"{theta_target:.3f}"
)

decision_col2.metric(
    "Lower Reference",
    f"{theta_lower_reference:.3f}"
)

if forecast_lower < theta_lower_reference:
    irrigation_status = "Irrigation attention recommended"
    decision_reason = (
        "The lower bound of the one-hour forecast falls below the "
        "experimental lower operating reference."
    )
    st.error(f"💧 {irrigation_status}")
elif forecast_lower < theta_target:
    irrigation_status = "Monitor moisture conditions"
    decision_reason = (
        "The forecast remains above the lower reference, but the uncertainty "
        "range falls below the experimental operating target."
    )
    st.warning(f"🌱 {irrigation_status}")
else:
    irrigation_status = "No immediate irrigation signal"
    decision_reason = (
        "The one-hour forecast and its uncertainty range remain above "
        "the experimental operating target."
    )
    st.success(f"✓ {irrigation_status}")

st.write(decision_reason)

st.caption(
    "The operating references are experimental values derived from the "
    "training data. They are not crop-specific agronomic thresholds."
)

