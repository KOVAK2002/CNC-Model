import librosa
import numpy as np
import tensorflow as tf

def extract_mfccs(file_path, n_mfcc=40):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_sound(model, file_path):
    features = extract_mfccs(file_path)
    features = np.expand_dims(features, axis=0)  # Reshape for model input
    prediction = model.predict(features)
    return np.argmax(prediction)

if __name__ == "__main__":
    model = load_model("cnc_sound_model_multiclass.h5")
    test_file = "test_audio/dt.wav"
    prediction = predict_sound(model, test_file)
    labels = {0: 'Normal Background', 1: 'Sharp Tool', 2: 'Dull Tool'}
    print(f"The predicted sound is: {labels[prediction]}")
