# CNC-Model
Using ML we can detect sharp vs dull tool
CNC Machine Sound Classification
This project aims to classify the sounds produced by a CNC machine into three categories: Normal Background, Sharp Tool, and Dull Tool, using machine learning techniques. It involves audio preprocessing, feature extraction using MFCC (Mel Frequency Cepstral Coefficients), training a neural network, and deploying the model on devices with limited resources like a Raspberry Pi.

Table of Contents
Project Overview
Technologies Used
Installation
Data Preprocessing
Feature Extraction
Model Training
Model Evaluation
Deploying the Model
Future Improvements
Contributing

Project Overview
The goal of this project is to develop a system that can automatically classify the operational sound of CNC machines. By detecting abnormal sounds (such as a dull tool), the system can alert operators for maintenance, thereby reducing downtime and increasing production efficiency.

Key features of the project include:

Preprocessing audio files for better feature extraction.
Feature extraction using MFCC, which captures key information about the sound.
Training a machine learning model (Neural Network and Logistic Regression).
Deploying the model for real-time sound classification on devices with limited resources, such as Raspberry Pi.
Technologies Used
Python 3.x
Librosa for audio processing and feature extraction.
TensorFlow/Keras for training and deploying the neural network.
Scikit-learn for logistic regression and evaluation.
Sounddevice for real-time audio recording.
Raspberry Pi for model deployment.
Installation
To set up the project on your local machine or Raspberry Pi, follow these steps:

Clone the repository:


git clone https://github.com/your-repo/cnc-sound-classification.git
cd cnc-sound-classification

Install required dependencies:


pip install numpy tensorflow librosa soundfile scikit-learn matplotlib seaborn sounddevice
(Optional) If you are using Raspberry Pi, ensure you have FFmpeg installed:

sudo apt install ffmpeg
Data Preprocessing
To preprocess the raw audio files, use the data_preprocessing.py script. This script normalizes the audio and segments it into 1-second chunks.

python data_preprocessing.py
Parameters:
Input directory: raw_audio/
Output directory: preprocessed_audio/
The output will be normalized and segmented audio files saved in the preprocessed_audio/ directory.

Feature Extraction
For feature extraction, use the feature_extraction.py script, which extracts MFCC features from the preprocessed audio files.


python feature_extraction.py
The features will be saved in features_multiclass.npz for further use in model training.

Model Training
To train the machine learning model (neural network or logistic regression), run the model_training.py script.


python model_training.py
This script will:

Load the extracted MFCC features.
Split the dataset into training and testing sets.
Train a neural network model.
Save the trained model as cnc_sound_model_multiclass.h5.
Model Architecture
Input: MFCC features
Hidden Layers: Dense layers with ReLU activation and dropout for regularization.
Output Layer: Softmax for multiclass classification (Normal, Sharp Tool, Dull Tool).
Model Evaluation
After training, you can evaluate the model's performance using the model_evaluation.py script.


python model_evaluation.py
This script will:

Load the test dataset and the trained model.
Generate a classification report showing precision, recall, and F1-score.
Display a confusion matrix to analyze the model’s accuracy for each class.
Deploying the Model
To deploy the trained model in real-time for classifying CNC machine sounds, use the deploy_model.py script.


python deploy_model.py
This script:

Continuously listens for sound using a microphone.
Extracts MFCC features from the recorded audio.
Classifies the sound into one of the three categories: Normal Background, Sharp Tool, or Dull Tool.
Displays the classification results in real-time.
For Raspberry Pi, ensure that the microphone is properly configured and detected.

Future Improvements
Optimization for Raspberry Pi: The current model might be computationally intensive for Raspberry Pi, so TensorFlow Lite or model pruning can be applied for better performance.
Additional Classes: Introduce more classes such as "Overheated Tool" or "Loose Component" to improve predictive maintenance.
Data Augmentation: Improve the robustness of the model by adding more varied data, such as different CNC machines or operating environments.
Contributing

Feel free to submit issues or pull requests for improvements. All contributions are welcome.
