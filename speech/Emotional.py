import os
import numpy as np
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to extract features from audio
def extract_features(file_name, sr=22050):
    audio, sample_rate = librosa.load(file_name, sr=sr)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13).T, axis=0)
    return mfccs

# Example dataset directories
directories = ["E:/Actor_21", "E:/Actor_22", "E:/Actor_23", "E:/Actor_24"]  # Replace with your actual directory paths
emotions = ['happy', 'sad', 'angry', 'neutral']  # Corresponding labels for each directory

# Extract features and labels
features = []
labels = []

for i, directory in enumerate(directories):
    if os.path.exists(directory):
        # Iterate over each file in the directory
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                try:
                    mfccs = extract_features(file_path)
                    features.append(mfccs)
                    labels.append(emotions[i])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    else:
        print(f"Directory not found: {directory}")

# Proceed with training and prediction if files are available
if features:
    features = np.array(features)
    labels = np.array(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Predict emotion from a new file
    new_file =r"E:\Actor_22\03-01-02-01-02-01-22.wav" # Replace with the actual path to your file
    if os.path.exists(new_file):
        try:
            new_features = extract_features(new_file).reshape(1, -1)
            predicted_emotion = clf.predict(new_features)
            print(f"Predicted Emotion: {predicted_emotion[0]}")
        except Exception as e:
            print(f"Error processing {new_file}: {e}")
    else:
        print(f"File not found: {new_file}")
else:
    print("No valid audio files found for training.")
