const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');

function loadImage(filePath) {
    const buffer = fs.readFileSync(filePath); // Read the image file
    const imageTensor = tf.node.decodeImage(buffer, 3); // Decode the image to a tensor
    const resizedImage = tf.image.resizeBilinear(imageTensor, [110, 190]); // Resize to 224x224
    return resizedImage.div(255);
}

function loadImagesFromFolder(folderPath) {
    const files = fs.readdirSync(folderPath);
    return files.map(file => loadImage(path.join(folderPath, file)));
}

module.exports = {loadImagesFromFolder};
