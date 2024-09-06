# fatigueAnalysis.js

fatigueAnalysis.js is a TensorFlow-powered JavaScript project designed to detect fatigue in real-time through an
integrated webcam. By training and utilizing machine learning models, the system accurately analyzes facial features to
determine tiredness, offering valuable insights for health and safety.

Imagine a long road trip with friends and family—wouldn't it be nice to check if the driver is okay? With
fatigueAnalysis.js, you can monitor the driver's alertness and ensure everyone stays safe on the road.

## Feature Engineering

### Labels

- Eyes opened
- Eyes closed
- Yawning

### Collection

- Create for each label a video
- Get each frame of the video

### Preprocessing

- Use mediapipe face detection to detect the face
- Save the face in a different file

## Training

- Load the faces and train the model

## Application

- Use the video stream
- Use mediapipe face detection to detect the face
- Predict the face

## Operation

- Explain why the images of one person are enough (tailored, fast release)
- Write about performance and user behaviour analysis
