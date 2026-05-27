import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings
warnings.filterwarnings("ignore")

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

NORMAL_DURATION = 10
EXIT_WAIT = 20


user_labels = [
    "aleyna",
    "damla",
    "deniz",
    "derya",
    "empty",
    "huseyin"
]


states = [
    "NORMAL DUR",
    "SAGA DON",
    "SOLA DON",
    "1 ADIM GERI GIT",
    "1 ADIM YAKLAS",
    "ODADAN CIK"
]

current_state_idx = 0
state_start_time = time.time()

exit_processed = False

model = load_model(
    "models/MULTI_SESSION_MODEL_EMPTY_V2.h5"
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

    #print(line[:100])
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


def clear_screen():

    os.system("clear")


def show_state():

    clear_screen()

    print("="*50)
    print("CSI REALTIME USER IDENTIFICATION")
    print("="*50)

    print(
        f"\nTEST ADIMI [{current_state_idx+1}/{len(states)}]"
    )

    print(
        f"YAPILACAK: {states[current_state_idx]}"
    )

    print("\n" + "="*50)

show_state()

while True:

    try:

        current_state = states[current_state_idx]

        if current_state == "ODADAN CIK" and not exit_processed:

            clear_screen()

            print("="*50)
            print("ODADAN CIK")
            print(f"{EXIT_WAIT} SANIYE BEKLENIYOR...")
            print("="*50)

            time.sleep(EXIT_WAIT)

            prediction_history.clear()
            buffer.clear()

            print("\nBUFFER TEMIZLENDI!")
            print("CSI TEKRAR BASLADI!")

            time.sleep(2)

            exit_processed = True

            state_start_time = time.time()

            continue

        if current_state != "ODADAN CIK":

            elapsed = time.time() - state_start_time

            if elapsed > NORMAL_DURATION:

                current_state_idx += 1

                if current_state_idx >= len(states):
                    current_state_idx = len(states) - 1

                state_start_time = time.time()

                prediction_history.clear()

                show_state()

                continue

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

            prediction_history.append(
                predicted_user
            )


            if confidence < 0.90 or margin < 0.04:

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

                print(
                    f"MARGIN: %{margin*100:.2f}"
                )

                print("="*40)

            else:

                # most_common = Counter(
                #     prediction_history
                # ).most_common(1)[0][0]

                # same_count = prediction_history.count(
                #     most_common
                # )

                # stability = (
                #     same_count / len(prediction_history)
                # ) * 100

                current_time = time.time()

                if current_time - last_prediction_time < 2:
                    continue

                last_prediction_time = current_time

                print("\n" + "="*40)

                print(
                    f"KULLANICI: {predicted_user.upper()}"
                )

                print(
                    f"GUVEN: %{confidence*100:.2f}"
                )

                print(
                    f"MARGIN: %{margin*100:.2f}"
                )

                # print(
                #     f"STABILITY: %{stability:.1f}"
                # )

                print("="*40)

            if predicted_user == "empty":

                print("ODA BOS")


    except KeyboardInterrupt:

        print("\nCIKILIYOR...")

        break

ser.close()