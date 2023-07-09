Python 3.10.11 (tags/v3.10.11:7d4cc5a, Apr  5 2023, 00:38:17) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Step 1: Prepare your dataset
# Load audio files and extract audio features (e.g., MFCC)
def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
...     mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
...     mfccs_processed = np.mean(mfccs.T, axis=0)
...     return mfccs_processed
... 
... # Example dataset
... audio_files = [
...     {'file': 'audio_files/anger.wav', 'label': 'anger'},
...     {'file': 'audio_files/happiness.wav', 'label': 'happiness'},
...     # Add more audio files with corresponding labels
... ]
... 
... # Extract features from audio files and create dataset
... X = np.array([extract_features(file['file']) for file in audio_files])
... y = np.array([file['label'] for file in audio_files])
... 
... # Step 2: Split the dataset into training and testing sets
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
... 
... # Step 3: Train a classifier
... classifier = MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42)
... classifier.fit(X_train, y_train)
... 
... # Step 4: Use the trained classifier for emotion detection
... def detect_emotion(file_path):
...     features = extract_features(file_path)
...     emotion = classifier.predict([features])[0]
...     return emotion
... 
... # Example usage
... audio_file = 'audio_files/speech.wav'
... detected_emotion = detect_emotion(audio_file)
... print(f"Detected emotion: {detected_emotion}")
... 
... 
... 
... 
... 
... 
