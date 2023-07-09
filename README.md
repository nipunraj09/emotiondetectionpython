# emotiondetectionpython
Emotion prediction on basis of voice.
In this Code, the extract_features() function loads an audio file using Librosa, extracts the Mel-frequency cepstral coefficients (MFCCs) as audio features, and returns the averaged MFCCs. The dataset consists of audio files with corresponding emotion labels. The extracted features are then split into training and testing sets using train_test_split().

Next, an MLPClassifier from scikit-learn is trained on the training set. You can adjust the architecture and hyperparameters of the classifier as needed.

Finally, the detect_emotion() function takes an audio file path, extracts features from the file, and uses the trained classifier to predict the emotion label. The predicted emotion is returned as the output.

Please note that this is a basic example, and the accuracy of emotion detection depends on various factors like the quality of the dataset, feature extraction techniques, and the performance of the chosen classifier. You might need a larger and more diverse dataset, as well as additional preprocessing steps or advanced techniques for better results.
