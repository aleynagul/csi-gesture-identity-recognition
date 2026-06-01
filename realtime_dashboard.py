import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import serial
import numpy as np
import pandas as pd
import plotly.express as px
import time

from collections import deque
from tensorflow.keras.models import load_model


PORT = "/dev/ttyUSB1"
BAUD = 115200

WINDOW_SIZE = 30
SUBCARRIERS = 64


user_labels = [
    "aleyna",
    "damla",
    "deniz",
    "derya",
    "empty",
    "hilal",
    "huseyin",
    "zehra"
]


st.set_page_config(
    page_title="CSI User Identification",
    layout="wide"
)

st.title(" CSI Realtime User Identification")

@st.cache_resource
def load_csi_model():

    model = load_model(
        "models/MULTI_SESSION_MODEL_EMPTY_V5.h5"
    )

    return model

model = load_csi_model()

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

buffer = deque(maxlen=WINDOW_SIZE)

def parse_csi(line):

    if "CSI:" not in line:
        return None

    try:

        data = line.split("CSI:")[1]

        values = [
            float(x)
            for x in data.strip().split(",")
            if x.strip()
        ]

        if len(values) < SUBCARRIERS:
            return None

        values = values[:SUBCARRIERS]

        return values

    except:
        return None


def normalize_sample(sample):

    sample_mean = np.mean(sample)

    sample_std = np.std(sample)

    sample = (
        sample - sample_mean
    ) / (sample_std + 1e-8)

    return sample

user_placeholder = st.empty()

confidence_placeholder = st.empty()

margin_placeholder = st.empty()

plot_placeholder = st.empty()

prob_placeholder = st.empty()


while True:

    line = ser.readline().decode(
        "utf-8",
        errors="ignore"
    )

    values = parse_csi(line)

    if values is None:
        continue

    buffer.append(values)

    if len(buffer) == WINDOW_SIZE:

        sample = np.array(
            buffer,
            dtype=np.float32
        )

        sample = sample[:, :64]

        sample = normalize_sample(sample)

        sample_input = np.expand_dims(
            sample,
            axis=0
        )

        prediction = model.predict(
            sample_input,
            verbose=0
        )

        pred_class = np.argmax(prediction)

        confidence = np.max(prediction)

        sorted_probs = np.sort(
            prediction[0]
        )[::-1]

        top1 = sorted_probs[0]
        top2 = sorted_probs[1]

        margin = top1 - top2

        predicted_user = user_labels[pred_class]

        if confidence < 0.90 or margin < 0.04:

            final_user = "BELIRSIZ"

        else:

            final_user = predicted_user.upper()


        user_placeholder.markdown(
            f"""
            ## CURRENT USER:
            # {final_user}
            """
        )

        confidence_placeholder.metric(
            "Confidence",
            f"{confidence*100:.2f}%"
        )

        margin_placeholder.metric(
            "Margin",
            f"{margin*100:.2f}%"
        )

        latest_frame = sample[-1]

        df_signal = pd.DataFrame({
            "Subcarrier": np.arange(len(latest_frame)),
            "Amplitude": latest_frame
        })

        fig_signal = px.line(
            df_signal,
            x="Subcarrier",
            y="Amplitude",
            title="Realtime CSI Signal"
        )

        plot_placeholder.plotly_chart(
            fig_signal,
            use_container_width=True
        )

        df_prob = pd.DataFrame({

            "User": user_labels,

            "Probability": prediction[0]

        })

        fig_prob = px.bar(
            df_prob,
            x="User",
            y="Probability",
            title="Prediction Probabilities"
        )

        prob_placeholder.plotly_chart(
            fig_prob,
            use_container_width=True
        )

        time.sleep(0.1)