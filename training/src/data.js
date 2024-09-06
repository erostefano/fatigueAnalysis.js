const {splitIntoTrainingData, getDataSet} = require("./util");
const logger = require("./logger");
const tf = require("@tensorflow/tfjs-node");
const {eyesOpen, eyesClosed, yawning} = require("./labels");

const eyesOpenSet = getDataSet(eyesOpen);
const eyesClosedSet = getDataSet(eyesClosed);
const yawningSet = getDataSet(yawning);

const eyesOpenTraining = splitIntoTrainingData(eyesOpenSet, eyesOpen);
const eyesClosedTraining = splitIntoTrainingData(eyesClosedSet, eyesClosed);
const yawningTraining = splitIntoTrainingData(yawningSet, yawning);

const xTrain = tf.stack(eyesOpenTraining.images.concat(eyesClosedTraining.images).concat(yawningTraining.images));
logger.info('xTrain shape', xTrain.shape);

const yTrain = tf.tensor2d(eyesOpenTraining.labels.concat(eyesClosedTraining.labels).concat(yawningTraining.labels), [xTrain.shape[0], 3]);
logger.info('yTrain shape', yTrain.shape);

module.exports = {train: {x: xTrain, y: yTrain}};
