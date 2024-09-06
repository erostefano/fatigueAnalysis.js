const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const logger = require("./logger");

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

function getDataSet(label) {
    const images = loadImagesFromFolder(`../feature-engineering/${label.label}/faces`);

    logger.info(`Total of ${label.label} images`, images.length);

    const labels = new Array(images.length).fill(label.encoding);
    logger.info(`Total of ${label.label} labels`, labels.length);

    return {images, labels};
}

function splitIntoTrainingData(data, label) {
    const splitAtIndex = 1089; // 1089 is 66% of 1650

    const images = data.images.slice(0, splitAtIndex);
    logger.info(`Total of ${label.label} training images`, images.length);

    const labels = data.labels.slice(0, splitAtIndex);
    logger.info(`Total of ${label.label} training labels`, images.length);

    return {
        images,
        labels
    }
}

module.exports = {getDataSet, splitIntoTrainingData};
