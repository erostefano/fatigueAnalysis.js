const tf = require('@tensorflow/tfjs-node');
const {createCnn} = require("./model");
const logger = require("./logger");

async function trainTestAndSave(train, test, activation, dropoutRate) {
    // const predictions = cnn.predict(xTest);
    // const predictedLabels = predictions.arraySync();
    //
    // logger.info('Predicted Labels', predictedLabels)
    // logger.info('Predicted Labels Size', predictedLabels.length)
    // logger.info('Expected Labels', yTest.arraySync())
    // logger.info('Expected Labels', yTest.arraySync().length)
    //
    // const summary = yTest.arraySync().map((labels, index) => {
    //     const isWithSunglasses = labels[0] === labelsWithEncoding.withSunglasses.encoding;
    //
    //     const label = isWithSunglasses
    //         ? labelsWithEncoding.withSunglasses.label
    //         : labelsWithEncoding.withoutSunglasses.label;
    //
    //     /*
    //         Every test picture starts at Nr. 661 (not the index)
    //
    //         Calculation to get the pictures using the index:
    //         - With Sunglasses:
    //             - First el : 0 + 661 = 661
    //             - Last el  : 339 + 661 = 1000
    //         - Without Sunglasses:
    //             - First el : 340 + 321 = 661
    //             - Last el  : 679 + 321 = 1000
    //      */
    //     const pictureIndex = isWithSunglasses
    //         ? Math.floor(index / 2) + 661
    //         : Math.floor(index / 2) + 321
    //
    //     return {
    //         label,
    //         withSunglassesPrediction: predictedLabels[index][0],
    //         withoutSunglassesPrediction: predictedLabels[index][1],
    //         file: `${isWithSunglasses ? 'with-sunglasses' : 'without-sunglasses'}-${pictureIndex}`
    //     };
    // });
    //
    // const withSunglassesNegative = summary
    //     .filter(row => row.label === labelsWithEncoding.withSunglasses.label)
    //     .filter(row => row.withSunglassesPrediction < row.withoutSunglassesPrediction);
    //
    // logger.info('withSunglassesNegative', JSON.stringify(withSunglassesNegative))
    // console.table(withSunglassesNegative)
    //
    // const withoutSunglassesNegative = summary
    //     .filter(row => row.label === labelsWithEncoding.withoutSunglasses.label)
    //     .filter(row => row.withoutSunglassesPrediction < row.withSunglassesPrediction);
    //
    // logger.info('withoutSunglassesNegative', JSON.stringify(withoutSunglassesNegative))
    // console.table(withoutSunglassesNegative)
    //
    // const confusionMatrix = summary.reduce(
    //     (acc, row) => {
    //         if (row.label === labelsWithEncoding.withSunglasses.label) {
    //             row.withSunglassesPrediction > row.withoutSunglassesPrediction
    //                 ? acc.withSunglassesPositive++
    //                 : acc.withSunglassesNegative++;
    //         }
    //
    //         if (row.label === labelsWithEncoding.withoutSunglasses.label) {
    //             row.withoutSunglassesPrediction > row.withSunglassesPrediction
    //                 ? acc.withoutSunglassesPositive++
    //                 : acc.withoutSunglassesNegative++
    //         }
    //
    //         return acc;
    //     },
    //     {
    //         withSunglassesPositive: 0,
    //         withSunglassesNegative: 0,
    //         withoutSunglassesPositive: 0,
    //         withoutSunglassesNegative: 0
    //     }
    // );
    //
    // logger.info('confusionMatrix', JSON.stringify(confusionMatrix))
    // console.table(confusionMatrix)

    // await cnn.save('file://model');
}

async function hyperParamTuning() {
    const {train, test} = require("./data");

    const modelPerformances = [];

    for (const activation of ['relu', 'elu', 'tanh', 'sigmoid']) {
        for (const dropoutRate of [0.2, 0.5, 0.8]) {
            for (const learningRate of [0.01, 0.001, 0.0001, 0.00001, 0.000001]) {
                logger.info(`Starting with Activation: ${activation}, Dropout: ${dropoutRate}, Learning Rate: ${learningRate}`);

                const cnn = createCnn(activation, dropoutRate, learningRate);
                const history = await cnn.fit(
                    train.x,
                    train.y,
                    {
                        epochs: 10,                                                  // Number of epochs
                        batchSize: 32,                                              // Number of samples per gradient update
                        callbacks: tf.callbacks.earlyStopping({monitor: 'acc', patience: 3})
                    }
                );

                const [lossTensor, accuracyTensor] = cnn.evaluate(test.x, test.y);

                const performance = {
                    activation,
                    dropoutRate,
                    trainingLoss: history.history.loss,
                    trainingAccuracy: history.history.acc,
                    testLoss: lossTensor.dataSync(),
                    testAccuracy: accuracyTensor.dataSync(),
                };

                logger.info('Model Performance', JSON.stringify(performance));
                console.table(performance);

                modelPerformances.push(performance);

                await cnn.save(`file://models/${activation}-${dropoutRate}-${learningRate}`);
            }
        }
    }

    logger.info('Model Performances', JSON.stringify(modelPerformances));
    console.table(modelPerformances);
}

hyperParamTuning()
