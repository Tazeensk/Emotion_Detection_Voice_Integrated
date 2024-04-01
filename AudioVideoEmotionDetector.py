import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json, load_model
from moviepy.editor import VideoFileClip
import os
import soundfile
import librosa

# Define emotion dictionaries
face_emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
audio_emotion_dict = {0: 'angry', 1: 'happy', 2: 'sad', 3: 'neutral', 4: 'fear', 5: 'disgust', 6: 'surprise'}

# Load face emotion detection model
json_file = open('Weights/emotion_model5.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
face_emotion_model = model_from_json(loaded_model_json)
face_emotion_model.load_weights("Weights/emotion_model5.h5")

# Load audio emotion detection model
audio_emotion_model = load_model('Weights/audio_emotion_model5.h5')

# Function to extract features from audio files
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

# Function to predict emotion from an audio file
def predict_audio_emotion(audio_path):
    features = extract_features(audio_path)
    features = np.reshape(features, (1, features.shape[0]))
    predicted_probabilities = audio_emotion_model.predict(features)
    predicted_emotion_index = np.argmax(predicted_probabilities)
    predicted_emotion = audio_emotion_dict[predicted_emotion_index]
    return predicted_emotion

def detect_emotions(video_path):
    clip = VideoFileClip(video_path)
    audio_path = 'temp_audio.wav'  # Temporary audio file path

    def process_frame(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Detect faces
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        # Process each detected face
        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # Predict emotions from face
            emotion_prediction = face_emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, face_emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        return frame

    def process_audio(audio_clip):
        audio_clip.write_audiofile(audio_path)
        predicted_emotion = predict_audio_emotion(audio_path)
        os.remove(audio_path)  # Remove temporary audio file
        return predicted_emotion

    def process_frame_with_audio(frame):
        frame_with_emotions = process_frame(frame)
        audio_emotion = process_audio(clip.audio)
        cv2.putText(frame_with_emotions, f"Audio Emotion: {audio_emotion}", (50, frame_with_emotions.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)  # Change color to yellow (BGR value: 0, 255, 255)
        return frame_with_emotions


    processed_clip = clip.fl_image(process_frame_with_audio)
    processed_video_path = 'processed_video8.mp4'
    processed_clip.write_videofile(processed_video_path, codec='libx264', fps=clip.fps)
    processed_clip.close()
    os.startfile(processed_video_path)  # Open the processed video file

def select_video():
    file_path = filedialog.askopenfilename()
    detect_emotions(file_path)

# GUI
root = tk.Tk()
root.title("Combined Emotion Detection")

button = tk.Button(root, text="Select Video", command=select_video)
button.pack()

root.mainloop()
