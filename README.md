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

**Note:** Git has limitations for storing large files. It is recommended to store models in an artifactory. For
convenience, the best-performing model is stored in this repository.

### Model Results

During training, each model result has been logged. The script analyze-logs.js summarizes and sorts them by test
accuracy. The following picture shows the first 20 results:

![model-results.png](assets%2Fmodel-results.png)

At first glance, the model with the tanh activation function, a dropout rate of 0.2, and a learning rate of 0.000001
appears to be the best choice, as it achieved a test accuracy of 76.9% and shows slight overfitting.

However, a closer examination of the confusion matrix reveals that the eyesOpen label has been frequently misclassified
as eyesClosed, which renders the model unsuitable.

```
{
  "eyesOpen": {
    "eyesOpen": 273,
    "eyesClosed": 288,
    "yawning": 0
  },
  "eyesClosed": {
    "eyesOpen": 47,
    "eyesClosed": 460,
    "yawning": 54
  },
  "yawning": {
    "eyesOpen": 0,
    "eyesClosed": 0,
    "yawning": 561
  }
}
```

Therefore, the best model is the one at index 3. It has the most balanced confusion matrix and uses the tanh activation
function, a dropout rate of 0.5, and a learning rate of 0.0001. Although it is clearly overfitted, it demonstrates more
balanced classification performance.

```
{
  "eyesOpen": {
    "eyesOpen": 325,
    "eyesClosed": 236,
    "yawning": 0
  },
  "eyesClosed": {
    "eyesOpen": 176,
    "eyesClosed": 307,
    "yawning": 78
  },
  "yawning": {
    "eyesOpen": 0,
    "eyesClosed": 0,
    "yawning": 561
  }
}
```

Overall, the accuracy is not satisfactory for the use case. Therefore, the next approach will be to try transfer
learning.

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
- **Randomize Pictures:** Even though the video captures multiple poses and directions, it is still necessary to
  randomize the pictures. For example, the last second of the video might show the face consistently facing the phone
  while trying to find the stop button.
