import os
import soundfile
import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Function to extract MFCC features from audio files
def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    X, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    if chroma:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    return result

# Load the trained model
model = load_model('Weights/audio_emotion_model5.h5')

# Function to predict emotion from an audio file
def predict_emotion(audio_path):
    # Extract features from the audio file
    features = extract_features(audio_path)
    # Reshape the features to match the input shape expected by the model
    features = np.reshape(features, (1, features.shape[0]))
    # Predict the emotion using the loaded model
    predicted_probabilities = model.predict(features)
    # Get the index of the emotion with the highest probability
    predicted_emotion_index = np.argmax(predicted_probabilities)
    # Map the index to the corresponding emotion label
    emotions = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust','surprise']
    predicted_emotion = emotions[predicted_emotion_index]
    return predicted_emotion

# Example usage:
audio_path = 'live_audios/laughter-29686.wav'
predicted_emotion = predict_emotion(audio_path)
print("Predicted emotion:", predicted_emotion)
