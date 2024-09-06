const tf = require('@tensorflow/tfjs-node');

const cnn = tf.sequential();

// Input layer with shape for face close-ups
cnn.add(tf.layers.conv2d({
    inputShape: [110, 190, 3],  // 128x128 RGB input for close-up face images
    filters: 32,                // Number of filters
    kernelSize: 3,              // 3x3 filter size
    activation: 'relu'
}));

// Max-pooling layer
cnn.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

// Second convolutional layer for feature extraction
cnn.add(tf.layers.conv2d({
    filters: 64,
    kernelSize: 3,
    activation: 'relu'
}));

// Max-pooling layer
cnn.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

// Flatten the output
cnn.add(tf.layers.flatten());

// Fully connected layer for feature combination
cnn.add(tf.layers.dense({
    units: 256,                 // Number of neurons
    activation: 'relu'
}));

// Dropout layer to prevent overfitting
cnn.add(tf.layers.dropout({rate: 0.5}));

// Output layer with three classes
cnn.add(tf.layers.dense({
    units: 3,                   // Output units for three labels
    activation: 'softmax'
}));

// Compile the model
cnn.compile({
    optimizer: tf.train.adam(),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
});

module.exports = {cnn};
