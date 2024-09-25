import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = np.load(file_path)
    return data['features'], data['labels']

def build_model(input_shape):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_shape,), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # Multi-class classification
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    features, labels = load_data("features_multiclass.npz")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    model.save("cnc_sound_model_multiclass.h5")
