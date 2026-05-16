import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import build_user_model, build_gesture_model

def train_separate_models():
    print("Ön işlenmiş veriler yükleniyor...")
    X = np.load("X_data.npy")
    y_user = np.load("y_user.npy")
    y_gesture = np.load("y_gesture.npy")
    
    input_shape = (X.shape[1], X.shape[2]) # (100, 64)
    
    # ----------------------------------------------------
    # 1. KULLANICI MODELİNİN EĞİTİMİ
    # ----------------------------------------------------
    print("\n=== 1. KULLANICI TANIMA MODELİ EĞİTİMİ BAŞLIYOR ===")
    X_train_u, X_test_u, y_train_u, y_test_u = train_test_split(
        X, y_user, test_size=0.2, random_state=42, stratify=y_user
    )
    
    user_model = build_user_model(input_shape, num_users=4)
    # Learning rate 0.001'den 0.0005'e çekildi (Daha kararlı adımlar için)
    user_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    early_stop_u = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
    
    user_model.fit(
        X_train_u, y_train_u,
        validation_data=(X_test_u, y_test_u),
        epochs=40,
        batch_size=16, # Batch size küçültüldü, genelleme yeteneğini artırır
        callbacks=[early_stop_u]
    )
    user_model.save("models/user_model.keras")
    print("✓ Kullanıcı modeli kaydedildi: models/user_model.keras")
    
    # ----------------------------------------------------
    # 2. HAREKET MODELİNİN EĞİTİMİ
    # ----------------------------------------------------
    print("\n=== 2. HAREKET ALGILAMA MODELİ EĞİTİMİ BAŞLIYOR ===")
    X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(
        X, y_gesture, test_size=0.2, random_state=42, stratify=y_gesture
    )
    
    gesture_model = build_gesture_model(input_shape, num_gestures=9)
    gesture_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    early_stop_g = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    gesture_model.fit(
        X_train_g, y_train_g,
        validation_data=(X_test_g, y_test_g),
        epochs=50,
        batch_size=16, # Batch size küçültüldü
        callbacks=[early_stop_g]
    )
    gesture_model.save("models/gesture_model.keras")
    print("✓ Hareket modeli kaydedildi: models/gesture_model.keras")
    
    print("\n🎉 Model Duyarlılık Ayarları Yapıldı ve Eğitim Tamamlandı! 🎉")

if __name__ == "__main__":
    train_separate_models()