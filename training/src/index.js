const tf = require('@tensorflow/tfjs-node');
const {createCnn} = require("./model");
const logger = require("./logger");

async function hyperParamTuning() {
    const {train, test} = require("./data");

    const modelPerformances = [];

    for (const activation of ['relu', 'elu', 'tanh', 'sigmoid']) {
        for (const dropoutRate of [0.2, 0.5, 0.8]) {
            for (const learningRate of [0.01, 0.001, 0.0001, 0.00001, 0.000001]) {
                const cnn = createCnn(activation, dropoutRate, learningRate);
                const history = await cnn.fit(
                    train.x,
                    train.y,
                    {
                        epochs: 10,
                        batchSize: 32,
                        callbacks: tf.callbacks.earlyStopping({monitor: 'acc', patience: 3})
                    }
                );

                const [lossTensor, accuracyTensor] = cnn.evaluate(test.x, test.y);

                const performance = {
                    activation,
                    dropoutRate,
                    learningRate: learningRate.toString(),
                    trainingLoss: history.history.loss,
                    trainingAccuracy: history.history.acc,
                    testLoss: lossTensor.dataSync(),
                    testAccuracy: accuracyTensor.dataSync(),
                };

                logger.info('Model Performance', JSON.stringify(performance));

                modelPerformances.push(performance);

                const predictions = cnn.predict(test.x);
                const predictedLabels = predictions.arraySync();

                const confusionMatrix = test.y.arraySync().reduce((acc, row, index) => {
                        const isEyesOpen = row[0] === 1;
                        const isEyesClosed = row[1] === 1;
                        const isYawning = row[2] === 1;

                        const prediction = predictedLabels[index];
                        const highestPrediction = Math.max(...prediction);

                        if (isEyesOpen) {
                            if (prediction[0] === highestPrediction) {
                                acc.eyesOpen.eyesOpen++
                            } else if (prediction[1] === highestPrediction) {
                                acc.eyesOpen.eyesClosed++
                            } else if (prediction[2] === highestPrediction) {
                                acc.eyesOpen.yawning++
                            }
                        }

                        if (isEyesClosed) {
                            if (prediction[0] === highestPrediction) {
                                acc.eyesClosed.eyesOpen++
                            } else if (prediction[1] === highestPrediction) {
                                acc.eyesClosed.eyesClosed++
                            } else if (prediction[2] === highestPrediction) {
                                acc.eyesClosed.yawning++
                            }
                        }

                        if (isYawning) {
                            if (prediction[0] === highestPrediction) {
                                acc.yawning.eyesOpen++
                            } else if (prediction[1] === highestPrediction) {
                                acc.yawning.eyesClosed++
                            } else if (prediction[2] === highestPrediction) {
                                acc.yawning.yawning++
                            }
                        }

                        return acc;
                    },
                    {
                        eyesOpen: {
                            eyesOpen: 0,
                            eyesClosed: 0,
                            yawning: 0
                        },
                        eyesClosed: {
                            eyesOpen: 0,
                            eyesClosed: 0,
                            yawning: 0
                        },
                        yawning: {
                            eyesOpen: 0,
                            eyesClosed: 0,
                            yawning: 0
                        }
                    }
                );

                logger.info('Confusion Matrix', JSON.stringify(confusionMatrix))

                await cnn.save(`file://models/${activation}-${dropoutRate}-${learningRate}`);
            }
        }
    }

    logger.info('Model Performances', JSON.stringify(modelPerformances));
}

hyperParamTuning()
