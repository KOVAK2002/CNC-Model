import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def load_data(file_path):
    data = np.load(file_path)
    return data['features'], data['labels']

def evaluate_model(model_path, X_test, y_test):
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    features, labels = load_data("features_multiclass.npz")
    _, X_test, _, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    evaluate_model("cnc_sound_model_multiclass.h5", X_test, y_test)
