import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPool1D, LSTM, Dense, Dropout

def build_user_model(input_shape, num_users=4):
    """Ezberleme karşıtı, hafifletilmiş kullanıcı modeli"""
    inputs = Input(shape=input_shape)
    
    x = Conv1D(filters=16, kernel_size=5, padding="same")(inputs) # Filtre küçüldü, kernel büyüdü
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(pool_size=2)(x)
    
    x = LSTM(16, return_sequences=False)(x)
    x = Dropout(0.5)(x) # Dropout %50'ye çıkarıldı ki ezberleyemesin
    
    x = Dense(16, activation="relu")(x)
    outputs = Dense(num_users, activation="softmax", name="user_output")(x)
    return Model(inputs=inputs, outputs=outputs, name="WiSense_User_Model")

def build_gesture_model(input_shape, num_gestures=9):
    """Zaman serisi akışına odaklanan, regülasyonlu hareket modeli"""
    inputs = Input(shape=input_shape)
    
    x = Conv1D(filters=32, kernel_size=5, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(pool_size=2)(x)
    
    x = Conv1D(filters=64, kernel_size=3, padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool1D(pool_size=2)(x)
    
    x = LSTM(32, return_sequences=False)(x)
    x = Dropout(0.5)(x) # Güçlü regülasyon
    
    x = Dense(32, activation="relu")(x)
    outputs = Dense(num_gestures, activation="softmax", name="gesture_output")(x)
    return Model(inputs=inputs, outputs=outputs, name="WiSense_Gesture_Model")