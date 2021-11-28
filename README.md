---
# hand-gesture-recognition-using-mediapipe
Estimate hand pose using MediaPipe(Python version).<br> This is a sample program that recognizes hand signs and finger gestures with a simple MLP using the detected key points.

This repository contains the following contents.
* Sample program
* Hand sign recognition model(TFLite)
* Learning data for hand sign recognition and notebook for learning

# Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix) 
* matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

# Demo
Here's how to run the demo using your webcam.
```bash
python main.py
```
https://youtu.be/UR0TmihRfRA
