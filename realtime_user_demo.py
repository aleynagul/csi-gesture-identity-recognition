import serial
import numpy as np
from collections import deque

from tensorflow.keras.models import load_model

PORT = "/dev/ttyUSB0"
BAUD = 921600

WINDOW_SIZE = 100
SUBCARRIERS = 64

user_labels = [
    "aleyna",
    "damla",
    "deniz",
    "derya"
]

model = load_model(
    "models/MULTI_SESSION_MODEL.h5"
)

print("MODEL YUKLENDI!")

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

print("CSI DINLENIYOR...")

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

        return values[:SUBCARRIERS]

    except:
        return None

def normalize_sample(sample):

    mean = np.mean(sample)

    std = np.std(sample)

    return (
        sample - mean
    ) / (std + 1e-8)

while True:

    try:

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        )

        values = parse_csi(line)

        if values is None:
            continue

        buffer.append(values)

        if len(buffer) == WINDOW_SIZE:

            sample = np.array(buffer)

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

            if confidence < 0.90:

                print(
                    "KULLANICI: BELIRSIZ"
                )

            else:

                print(
                    f"KULLANICI: "
                    f"{user_labels[pred_class]}"
                )

                print(
                    f"GUVEN: "
                    f"%{confidence*100:.2f}"
                )

                print("-" * 40)

    except KeyboardInterrupt:

        print("CIKILIYOR...")

        break