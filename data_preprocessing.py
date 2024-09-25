import os
import librosa
import numpy as np
import soundfile as sf

def load_audio(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    return audio, sample_rate

def normalize_audio(audio):
    return librosa.util.normalize(audio)

def segment_audio(audio, sample_rate, segment_duration=1):
    segment_length = int(segment_duration * sample_rate)
    segments = [audio[i:i + segment_length] for i in range(0, len(audio), segment_length)]
    return segments

def save_segments(segments, output_dir, base_name, sample_rate):
    os.makedirs(output_dir, exist_ok=True)
    for i, segment in enumerate(segments):
        segment_path = os.path.join(output_dir, f"{base_name}_segment_{i}.wav")
        sf.write(segment_path, segment, sample_rate)

if __name__ == "__main__":
    input_dir = "raw_audio/"  # Specify the path to your raw audio files here
    output_dir = "preprocessed_audio/"  # Directory to save preprocessed files
    
    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            file_path = os.path.join(input_dir, filename)
            audio, sample_rate = load_audio(file_path)
            normalized_audio = normalize_audio(audio)
            segments = segment_audio(normalized_audio, sample_rate)
            save_segments(segments, output_dir, os.path.splitext(filename)[0], sample_rate)
