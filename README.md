# Model runs in the browser

Browser apps offer a user-friendly experience without the need for installation, making them accessible on a wide range
of devices. They provide instant updates and can be tailored to your needs through extensions. Smart caching ensures
that the browsing experience remains quick and responsive.

High performance is achieved by running models directly on your device, removing the need for powerful servers. This
method supports endless scalability, allowing apps to manage growing demands efficiently while maintaining speed.

## Use Case

Imagine a long road trip with friends and family—wouldn't it be nice to check if the driver is okay? With
fatigueAnalysis.js, you can monitor the driver's alertness and ensure everyone stays safe on the road.

This project is designed to detect fatigue in real-time through an integrated webcam. By training and utilizing machine
learning models, the system accurately analyzes facial features to determine tiredness, offering valuable insights for
health and safety.

## Feature Engineering

The label **"Eyes open"** represents instances when the eyes are open. **"Eyes closed"** indicates when the
eyes are closed. **"Yawning"** refers to situations where the mouth is covered by the hand.

### Data Collection

- **Video Recording:** For each label a 30 seconds video has been recorded in car.
- **Frame Extraction:** Every frame has been extracted. Note: Large dataset may cause Git issues.

### Data Preprocessing

- **Face Extraction**: Using the Blazeface model from TensorFlow.js, the face has been extracted from each frame. The
  image size has been reduced to max. 35 kB

## Training

For image analysis, Convolutional Neural Networks (CNNs) are recommended. They excel in feature extraction and pattern
recognition through their convolutional layers, making them ideal for tasks like image classification and object
detection.

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

The initial model showed suboptimal accuracy. To improve performance, hyperparameter tuning was applied, as these
parameters greatly influence model outcomes.

- **Activation Functions:** ReLU, ELU, tanh, sigmoid
- **Dropout Rates:** 0.2, 0.5, 0.8
- **Learning Rates:** 0.01, 0.001, 0.0001, 0.00001, 0.000001

In this setup, 60 models have been trained, tested, and stored.

**Note:** Git has limitations for storing large files. It is recommended to store models in an artifactory. For
convenience, the best-performing model is stored in this repository.

### Model Results

During training, each model result has been logged. The script analyze-logs.js summarizes and sorts them by test
accuracy. The following picture shows the first 20 results:

![Model Results](assets%2Fmodel-results.png)

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

Overall, the accuracy is not satisfactory for the use case. To improve it, more data and parameter tuning are needed.
Due to performance limitations, this approach won't be feasible. Instead, alternative transfer learning solutions should
be explored.

### Transfer Learning using Face Landmarks

A better solution is to use a model to extract face landmarks, as this approach does not require additional image
processing.

![Face Landmarks](assets%2Fface-landmarks.png)

- **Feature Extraction:** Use a model with a webcam to extract face landmarks and label the data.
- **Training:** Train a neural network on the labeled face landmarks.
- **Benefits:** This approach requires less computational power, enabling the use of much more data without
  significantly increasing the computational load.
- **Size**: Approximately 8.7kB, Source: https://codepen.io/mediapipe-preview/pen/OJBVQJm

Face landmarks typically occupy around 9 kB, which is about four times smaller than images. Additionally, they can be
further reduced to include only the necessary features. For example, ears are not crucial for analyzing fatigue.

## Application

The webapp captures the video stream from the camera, using the Blazeface model from feature engineering to detect and
process faces during preprocessing. The trained model then predicts outcomes based on the preprocessed image.

Overall, the performance is bad. Anyway, here are some working examples:

- ![Prediction: Eyes open](assets%2Feyes-open.png)
- ![Prediction: Eyes closed](assets%2Feyes-closed.png)
- ![Prediction: Yawning](assets%2Fyawning.png)

The application is stored [index.html](application%2Findex.html) and can be run simply by opening it in a browser.

**Note:** Since the model and the application do not perform well, there was no deployment.

## Operation

The application and model run entirely in the browser. Aside from serving files, the server has no additional
functionality. Therefore, user and performance analysis must be implemented in the browser.

- **Data Collection**: In the early stage of the project, models are created for each customer. Once enough data is
  collected, a global model can be introduced and shared.
- **Measurements**: Use Google Analytics to gather insights into user behavior, such as browser type, device, browser
  version, location, session duration, retention and page views. Additionally, track custom events like user
  interactions, clicks, form submissions, and logging events.
- **User Feedback**: Allow users to provide feedback and submit images.
- **Model Updates**: As users may change their appearance, models need to be updated. Therefore, the application should
  continuously gather data to track changes and update models accordingly.

## Critics

- **Calculation instead of Training:** Face landmarks provide precise and complete predictions, which can be used to
  determine if the eyes are open.
- **Data Augmentation:** Because of performance issues there was no data augmentation.
- **Randomize Pictures:** Even though the video captures multiple poses and directions, it is still necessary to
  randomize the pictures. For example, the last second of the video might show the face consistently facing the phone
  while trying to find the stop button.
- **Missing Seed:** No seed has been set, making the models non-reproducible.
- **Testing:** There are no formal tests to confirm the functions. Everything has been tested manually through
  exploratory testing.

## Learning and Outlook

- **Model runs in the browser:** It is absolutely possible to run one or multiple models in the browser, though there
  may be some performance limitations at times.
- **Train a model in the browser:** It's also worth exploring how well a model can be trained in the browser while
  keeping the user experience smooth. This will be analyzed in a future project.
