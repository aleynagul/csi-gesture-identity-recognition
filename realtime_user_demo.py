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


# =========================
# SETTINGS
# =========================

PORT = "/dev/ttyUSB1"
BAUD = 115200

WINDOW_SIZE = 30
SUBCARRIERS = 64

NORMAL_DURATION = 10
EXIT_WAIT = 20


# =========================
# LABELS
# =========================

user_labels = [
    "aleyna",
    "damla",
    "deniz",
    "derya",
    "empty"
]


# =========================
# TEST STATES
# =========================

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


# =========================
# MODEL LOAD
# =========================

model = load_model(
    "models/MULTI_SESSION_MODEL_EMPTY_V2.h5"

)

print("MODEL YUKLENDI!")


# =========================
# SERIAL START
# =========================

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=1
)

print("CSI DINLENIYOR...")


# =========================
# BUFFERS
# =========================

buffer = deque(maxlen=WINDOW_SIZE)

prediction_history = deque(maxlen=5)

last_prediction_time = 0


# =========================
# CSI PARSER
# =========================

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


# =========================
# NORMALIZATION
# =========================

def normalize_sample(sample):

    sample_mean = np.mean(sample)

    sample_std = np.std(sample)

    sample = (
        sample - sample_mean
    ) / (sample_std + 1e-8)

    return sample


# =========================
# CLEAR SCREEN
# =========================

def clear_screen():

    os.system("clear")


# =========================
# SHOW STATE
# =========================

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


# =========================
# START SCREEN
# =========================

show_state()


# =========================
# REALTIME LOOP
# =========================

while True:

    try:

        current_state = states[current_state_idx]


        # =========================
        # ODADAN CIK STATE
        # =========================

        if current_state == "ODADAN CIK" and not exit_processed:

            clear_screen()

            print("="*50)
            print("ODADAN CIK")
            print(f"{EXIT_WAIT} SANIYE BEKLENIYOR...")
            print("="*50)

            # inference pause
            time.sleep(EXIT_WAIT)

            # eski CSI temizleniyor
            prediction_history.clear()
            buffer.clear()

            print("\nBUFFER TEMIZLENDI!")
            print("CSI TEKRAR BASLADI!")

            time.sleep(2)

            exit_processed = True

            state_start_time = time.time()

            continue


        # =========================
        # NORMAL STATE TIMER
        # =========================

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


        # =========================
        # SERIAL READ
        # =========================

        line = ser.readline().decode(
            "utf-8",
            errors="ignore"
        )

        values = parse_csi(line)

        if values is None:
            continue

        buffer.append(values)


        # =========================
        # PREDICTION
        # =========================

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


            # =========================
            # LOW CONFIDENCE
            # =========================

            if confidence < 0.93:

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


            # =========================
            # NORMAL RESULT
            # =========================

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


            # =========================
            # EMPTY INFO
            # =========================

            if predicted_user == "empty":

                print("ODA BOŞ")


    except KeyboardInterrupt:

        print("\nCIKILIYOR...")

        break


# =========================
# CLOSE SERIAL
# =========================

ser.close()