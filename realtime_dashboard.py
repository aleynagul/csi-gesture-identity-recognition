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

from collections import deque, Counter
from tensorflow.keras.models import load_model


PORT = "/dev/ttyUSB0"
BAUD = 115200

WINDOW_SIZE = 100
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
        "models/MULTI_SESSION_MODEL_EMPTY_V6.h5"
    )

    return model

model = load_csi_model()

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

buffer = deque(maxlen=WINDOW_SIZE)
prediction_history = deque(maxlen=10)

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

    
    #print(line)
    print("ilk 10 değer ", values[:10])
    print("Mean",np.mean(values))
    print("Mestdan",np.std(values))
    time.sleep(1)


    buffer.append(values) 

    if len(buffer) == WINDOW_SIZE:

        sample = np.array(
            buffer,
            dtype=np.float32
        )

        sample = sample[:, :64]

        sample = normalize_sample(sample)
        np.save(
            "realtime_sample.npy",
            sample
        )

        #print("\nRealtime sample stats")
        #print("Shape:", sample.shape)
        #print("Min:", np.min(sample))
        #print("Max:", np.max(sample))
        #print("Mean:", np.mean(sample))
        #print("Std:", np.std(sample))

        sample_input = np.expand_dims(
            sample,
            axis=0
        )

        prediction = model.predict(
            sample_input,
            verbose=0
        )
        print("\nPrediction:")
        print(prediction[0])

        pred_class = np.argmax(prediction)

        confidence = np.max(prediction)

        sorted_probs = np.sort(
            prediction[0]
        )[::-1]

        top1 = sorted_probs[0]
        top2 = sorted_probs[1]

        margin = top1 - top2

        predicted_user = user_labels[pred_class]

        # Empty güçlü gelirse geçmişi sıfırla
        if predicted_user == "empty" and confidence > 0.60:

            prediction_history.clear()
            prediction_history.append("empty")

        else:

            if confidence < 0.50 or margin < 0.03:

                prediction_history.append("belirsiz")

            else:

                prediction_history.append(predicted_user)

        # Son tahminleri oylama ile stabilize et
        if len(prediction_history) >= 5:

            final_user = Counter(
                prediction_history
            ).most_common(1)[0][0]

        else:

            final_user = "bekleniyor"

        final_user = final_user.upper()


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