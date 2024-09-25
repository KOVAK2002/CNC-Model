import os
import numpy as np
import librosa

def extract_mfccs(file_path, n_mfcc=40):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

def process_directory(input_dir, label, feature_list, label_list):
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_dir, filename)
            features = extract_mfccs(file_path)
            feature_list.append(features)
            label_list.append(label)

def save_features_and_labels(features, labels, output_file):
    np.savez(output_file, features=np.array(features), labels=np.array(labels))

if __name__ == "__main__":
    data_dirs = {
        'normal_background': 'preprocessed_audio/normal_background/',
        'sharp_tool': 'preprocessed_audio/sharp_tool/',
        'dull_tool': 'preprocessed_audio/dull_tool/'
    }
    labels = {
        'normal_background': 0,
        'sharp_tool': 1,
        'dull_tool': 2
    }
    
    feature_list = []
    label_list = []

    for label_name, directory in data_dirs.items():
        process_directory(directory, labels[label_name], feature_list, label_list)

    output_file = 'features_multiclass.npz'
    save_features_and_labels(feature_list, label_list, output_file)
