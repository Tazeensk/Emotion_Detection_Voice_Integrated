# Emotion_Detection_Voice_Integrated
This project combines both facial and audio emotion detection to predict emotions in videos. It utilizes convolutional neural networks (CNNs) for facial emotion detection and recurrent neural networks (RNNs) for audio emotion detection.

## Features
Detects faces in videos using OpenCV and Haar cascades.
Predicts facial emotions (Angry, Disgusted, Fearful, Happy, Neutral, Sad, Surprised) using a pre-trained CNN model.
Extracts audio from videos and predicts emotions (Angry, Happy, Sad, Neutral, Fear, Disgust, Surprise) using a trained CNN model.
Combines both facial and audio predictions to provide a comprehensive analysis of emotions in the video.

## Usage
Install the required libraries using pip install -r requirements.txt.
Run the main script emotion_detection.py.
Select a video file.
The processed video with detected emotions will be displayed.

## Credits
Facial Emotion Detection Model: link to model - Trained on : https://www.kaggle.com/datasets/msambare/fer2013 dataset.
Audio Emotion Detection Model: link to model - Trained on https://www.kaggle.com/datasets/ejlok1/cremad dataset.
Haar Cascade Classifier for face detection: OpenCV


## Outputs
