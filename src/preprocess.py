import os
import numpy as np
from scipy.signal import butter, filtfilt

DATA_DIR = "../new_data"  
TIMESTEPS = 100 
SUBCARRIERS = 64

def butter_lowpass_filter(data, cutoff=4, fs=40, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    
    smoothed = np.copy(data)
    for i in range(data.shape[1]):
        if len(data) > 3 * order:
            smoothed[:, i] = filtfilt(b, a, data[:, i])
    return smoothed

def parse_csv_robustly(filepath, max_sc=SUBCARRIERS):
    matrix_data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    if len(lines) <= 1:
        return None

    header = lines[0].strip().split(',')
    sc_indices = [i for i, col in enumerate(header) if col.startswith('sc_')]

    if len(sc_indices) < max_sc:
        sc_indices = list(range(1, min(len(header), max_sc + 1)))

    sc_indices = sc_indices[:max_sc]

    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) <= max(sc_indices):
            continue
        try:
            row_values = [float(parts[idx]) for idx in sc_indices]
            matrix_data.append(row_values)
        except ValueError:
            continue

    if len(matrix_data) == 0:
        return None

    return np.array(matrix_data, dtype=np.float32)

def pad_or_truncate_matrix(data, max_len=TIMESTEPS):
    if len(data) < max_len:
        pad_width = max_len - len(data)
        data = np.pad(data, ((0, pad_width), (0, 0)), mode='edge')
    else:
        data = data[:max_len, :]
    return data

def load_dataset_final():
    X_final = []
    y_user_final = []
    y_gesture_final = []
    valid_files = []

    if not os.path.exists(DATA_DIR):
        print(f" Kritik Hata: '{DATA_DIR}' dizini bulunamadı!")
        return

    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.startswith(".")]

    if len(files) == 0:
        print(f" Kritik Hata: '{DATA_DIR}' içinde .csv dosyası bulunamadı!")
        return

    print(f"Klasörde toplam {len(files)} dosya var. Analiz ediliyor...")

    user_map = {"s01": 0, "s02": 1, "s03": 2, "s04": 3, "empty": 4}
    gesture_map = {"still": 0, "hand_clap": 1, "horizontal_arm_wave": 2, "bend": 3, "empty": 4}

    success_count = 0

    for filename in files:
        name_without_ext = filename.rsplit('.', 1)[0]
        parts = name_without_ext.split("_")

        if len(parts) < 3:
            continue

        first_part = parts[0].lower()
        
        if first_part == "empty":
            user = "empty"
            gesture = "empty"
        else:
            user = first_part
            gesture = "_".join(parts[1:-2]).strip().lower()

        if user not in user_map or gesture not in gesture_map:
            continue

        valid_files.append((filename, user, gesture))

    for filename, user, gesture in valid_files:
        filepath = os.path.join(DATA_DIR, filename)
        raw_matrix = parse_csv_robustly(filepath, max_sc=SUBCARRIERS)

        if raw_matrix is not None:
            #filtered_matrix = butter_lowpass_filter(raw_matrix, cutoff=4, fs=40, order=4)
            #processed_matrix = pad_or_truncate_matrix(filtered_matrix, max_len=TIMESTEPS)
            
            processed_matrix = pad_or_truncate_matrix(raw_matrix, max_len=TIMESTEPS)

            X_final.append(processed_matrix)
            y_user_final.append(user_map[user])
            y_gesture_final.append(gesture_map[gesture])
            success_count += 1

    print(f"\n İşlem Tamamlandı: {success_count}/{len(files)} dosya başarıyla standardize edildi.")

    return (
        np.array(X_final, dtype=np.float32),
        np.array(y_user_final, dtype=np.int32),
        np.array(y_gesture_final, dtype=np.int32),
        user_map,
        gesture_map
    )

if __name__ == "__main__":
    X, y_user, y_gesture, u_map, g_map = load_dataset_final()

    if len(X) > 0:
        print("\n--- Çıktı Matris Boyutları ---")
        print(f"Girdi Boyutu (X)                    : {X.shape}")
        print(f"Kullanıcı Etiket Boyutu (y_user)    : {y_user.shape}")
        print(f"Hareket Etiket Boyutu (y_gesture)  : {y_gesture.shape}")

        np.save("X_identity_100.npy", X)
        np.save("y_user_identity_100.npy", y_user)
        np.save("y_gesture_identity_100.npy", y_gesture)