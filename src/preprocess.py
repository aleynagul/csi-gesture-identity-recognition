import os
import numpy as np

DATA_DIR = "data"
TIMESTEPS = 100   
SUBCARRIERS = 64  
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
    X = []
    raw_users = []
    raw_gestures = []
    valid_files = []
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv") and not f.startswith(".")]
    
    if len(files) == 0:
        print(f"Kritik Hata: '{DATA_DIR}' klasöründe .csv dosyası bulunamadı!")
        return
        
    print(f"Klasörde toplam {len(files)} dosya var. Kurşun geçirmez yöntemle analiz ediliyor...")

    for filename in files:
        name_without_ext = filename.rsplit('.', 1)[0]
        parts = name_without_ext.split("_")
        
        if len(parts) < 3:
            continue
            
        user = parts[0].strip().lower()
        gesture = "_".join(parts[1:-1]).strip().lower()
        
        raw_users.append(user)
        raw_gestures.append(gesture)
        valid_files.append((filename, user, gesture))
                
    unique_users = sorted(list(set(raw_users)))
    unique_gestures = sorted(list(set(raw_gestures)))
    
    user_map = {user: idx for idx, user in enumerate(unique_users)}
    gesture_map = {gesture: idx for idx, gesture in enumerate(unique_gestures)}
    
    print("\n--- Otomatik Algılanan Kullanıcılar ---")
    for u, idx in user_map.items(): print(f"'{u}' -> Etiket ID: {idx}")
        
    print("\n--- Otomatik Algılanan Hareketler ---")
    for g, idx in gesture_map.items(): print(f"'{g}' -> Etiket ID: {idx}\n")
    
    X_final = []
    y_user_final = []
    y_gesture_final = []
    
    success_count = 0
    
    for filename, user, gesture in valid_files:
        filepath = os.path.join(DATA_DIR, filename)
        
        raw_matrix = parse_csv_robustly(filepath, max_sc=SUBCARRIERS)
        
        if raw_matrix is not None:
            processed_matrix = pad_or_truncate_matrix(raw_matrix, max_len=TIMESTEPS)
            
            X_final.append(processed_matrix)
            y_user_final.append(user_map[user])
            y_gesture_final.append(gesture_map[gesture])
            success_count += 1

    print(f"--Filtreleme Sonucu: {success_count}/{len(files)} dosya başarıyla standardize edildi.")

    return np.array(X_final, dtype=np.float32), np.array(y_user_final, dtype=np.int32), np.array(y_gesture_final, dtype=np.int32), user_map, gesture_map

if __name__ == "__main__":
    X, y_user, y_gesture, u_map, g_map = load_dataset_final()
    
    if len(X) > 0:
        print("DATASET BAŞARIYLA STANDARTLAŞTIRILDI VE YÜKLENDI!")
        print(f"Girdi Boyutu (X): {X.shape} -> (Toplam Örnek, Zaman Adımı, Sabit Subcarrier)")
        print(f"Kullanıcı Etiket Boyutu (y_user): {y_user.shape}")
        print(f"Hareket Etiket Boyutu (y_gesture): {y_gesture.shape}")
        
        np.save("X_data.npy", X)
        np.save("y_user.npy", y_user)
        np.save("y_gesture.npy", y_gesture)
        print("Ön işlenmiş veriler 'X_data.npy', 'y_user.npy' ve 'y_gesture.npy' olarak ana dizine kaydedildi!")