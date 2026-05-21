import serial
import numpy as np
import time

from collections import deque
from collections import Counter

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
    "empty"
]

model = load_model(
    "models/MULTI_SESSION_MODEL_EMPTY_V3.h5"
)

print("MODEL YUKLENDI!")



ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

print("CSI DINLENIYOR...")

buffer = deque(maxlen=WINDOW_SIZE)

prediction_history = deque(maxlen=5)

last_prediction_time = 0

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

while True:

    try:

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        )

        values = parse_csi(line)
        #print(line[:100])

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

            predicted_user = user_labels[pred_class]

            prediction_history.append(
                predicted_user
            )

            if confidence < 0.85:

                current_time = time.time()

                if current_time - last_prediction_time < 2:
                    continue

                last_prediction_time = current_time

                print("\n" + "="*40)

                print(
                    "KULLANICI: BELIRSIZ"
                )

                print(
                    f"GUVEN DUSUK: %{confidence*100:.2f}"
                )

                print("="*40)

            else:

                most_common = Counter(
                    prediction_history
                ).most_common(1)[0][0]

                same_count = prediction_history.count(
                    most_common
                )

                stability = (
                    same_count / len(prediction_history)
                ) * 100

                current_time = time.time()

                if current_time - last_prediction_time < 2:
                    continue

                last_prediction_time = current_time

                print("\n" + "="*40)

                print(
                    f"KULLANICI: {most_common.upper()}"
                )

                print(
                    f"GUVEN: %{confidence*100:.2f}"
                )

                print(
                    f"STABILITY: %{stability:.1f}"
                )

                print("="*40)
            
            if predicted_user == "empty":
                print("ODA BOŞ")

    except KeyboardInterrupt:

        print("CIKILIYOR...")

        break