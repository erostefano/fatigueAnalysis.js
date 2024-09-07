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
- **Frame Extraction:** Every frame has been extracted. Note: Large dataset may cause Git issues.

### Data Preprocessing

- **Face Extraction**: Using the Blazeface model from TensorFlow.js, the face has been extracted from each frame. The
  image size has been reduced to max. 35 kB

## Training

### Convolutional Neural Network

The model is a Convolutional Neural Network (CNN) built using TensorFlow.js for classifying close-up face images into
three categories: Eyes open, closed, and yawning.

- **Input Layer:** Accepts images of size 110x190 with 3 color channels (RGB).
- **Convolutional Layers:**
    - **First Convolutional Layer:** 32 filters, 3x3 kernel, specified activation function.
    - **Second Convolutional Layer:** 64 filters, 3x3 kernel, specified activation function.
- **Max-Pooling Layers:** Applied after each convolutional layer with a 2x2 pool size.
- **Flatten Layer:** Converts the 2D feature maps into 1D.
- **Dense Layer:** Fully connected layer with 256 units and specified activation function.
- **Dropout Layer:** Applies dropout with a rate to prevent overfitting.
- **Output Layer:** Produces three output units with softmax activation for multi-class classification.

The model is compiled with the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric.

### Hyperparameter Tuning

The initial model had a suboptimal accuracy. To find the optimal configuration, hyperparameter tuning was implemented:

- **Activation Functions:** relu, elu, tanh, sigmoid
- **Dropout Rates:** 0.2, 0.5, 0.8
- **Learning Rates:** 0.01, 0.001, 0.0001, 0.00001, 0.000001

In this setup, 60 models have been trained, tested, and stored.

**Note:** Git has limitations for storing large files. It is recommended to store models in an artifactory. For convenience, the best-performing model is stored in this repository.

### Model Results

// TODO: add table

### Transfer Learning

// TODO: try it

## Application

- Use the video stream
- Use mediapipe face detection to detect the face
- Predict the face

## Operation

- Explain why the images of one person are enough (tailored, fast release)
- Write about performance and user behaviour analysis

## Critics

- **Use Existing Model:** Face detection models return face points that can be used to calculate the distance between
  the upper and lower eyelids, as well as the mouth. This may potentially be accurate enough to detect signs of fatigue.
