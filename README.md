# fatigueAnalysis.js

fatigueAnalysis.js is a TensorFlow-powered JavaScript project designed to detect fatigue in real-time through an
integrated webcam. By training and utilizing machine learning models, the system accurately analyzes facial features to
determine tiredness, offering valuable insights for health and safety.

Imagine a long road trip with friends and family—wouldn't it be nice to check if the driver is okay? With
fatigueAnalysis.js, you can monitor the driver's alertness and ensure everyone stays safe on the road.

## Feature Engineering

### Labels

- **Eyes open:** Represents when the eyes are open and visible.
- **Eyes closed:** Represents when the eyes closed.
- **Yawning:** Represents when the mouth is covered by the hand.

### Data Collection

- **Video Recording:** For each label a 30 seconds video has been recorded in car.
- **Frame Extraction:** Every frame has been extracted. Note: Large dataset may cause git issues.

### Data Preprocessing

- **Face Extraction**: Using the Blazeface model from TensorFlow.js, the face has been extracted from each frame. The
  image size has been reduced to max. 35 kB

## Training

- Load the faces and train the model

## Application

- Use the video stream
- Use mediapipe face detection to detect the face
- Predict the face

## Operation

- Explain why the images of one person are enough (tailored, fast release)
- Write about performance and user behaviour analysis
