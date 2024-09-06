const fs = require('fs');
const path = require('path');
const sharp = require('sharp');
const { createCanvas, loadImage } = require('canvas');
const tf = require('@tensorflow/tfjs-node');
const blazeface = require('@tensorflow-models/blazeface');

const label = 'yawning'
const inputDirectory = `../${label}/frames`;
const outputDirectory = `../${label}/faces`;

// Load the BlazeFace model
let model;

async function loadModel() {
    model = await blazeface.load();
    console.log('BlazeFace model loaded');
}

async function processImage(imagePath, imageName) {
    try {
        // Load the model
        if (!model) await loadModel();

        // Read the image file
        const imageBuffer = fs.readFileSync(imagePath);

        // Load image into canvas
        const img = await loadImage(imageBuffer);
        const canvas = createCanvas(img.width, img.height);
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        // Convert canvas to tensor
        const imageTensor = tf.browser.fromPixels(canvas);

        // Detect faces
        const predictions = await model.estimateFaces(imageTensor, false);

        if (predictions.length > 0) {
            // Process the first detected face
            const prediction = predictions[0];
            const [x, y, width, height] = prediction.topLeft.concat(prediction.bottomRight);
            const faceWidth = width - x;
            const faceHeight = height - y;

            // Crop face region using sharp
            const faceImageBuffer = await sharp(imageBuffer)
                .extract({
                    left: Math.floor(x),
                    top: Math.floor(y),
                    width: Math.floor(faceWidth),
                    height: Math.floor(faceHeight)
                })
                .toBuffer();

            // Remove 'frame' from the imageName and format output filename
            const cleanImageName = imageName.replace(/frame/, '');
            const faceImagePath = path.join(outputDirectory, `face${cleanImageName}.jpg`);
            fs.writeFileSync(faceImagePath, faceImageBuffer);

            console.log(`First face detected and saved as face${cleanImageName}.jpg.`);
        } else {
            console.log(`No faces detected in image ${imageName}.`);
        }
    } catch (err) {
        console.error('Error processing image:', err);
    }
}

// Process all images in the input directory
async function processAllImages() {
    try {
        // Load the model
        if (!model) await loadModel();

        // Read all files in the input directory
        const files = fs.readdirSync(inputDirectory);

        for (const file of files) {
            const filePath = path.join(inputDirectory, file);

            // Only process image files
            if (fs.statSync(filePath).isFile() && /\.(jpg|jpeg|png)$/i.test(file)) {
                const imageName = path.parse(file).name;
                await processImage(filePath, imageName);
            }
        }
    } catch (err) {
        console.error('Error processing images:', err);
    }
}

// Start processing images
processAllImages();
