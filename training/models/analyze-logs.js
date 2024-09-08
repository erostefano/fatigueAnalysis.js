const fs = require('fs');

const data = fs.readFileSync('../logs.txt', 'utf8');

const performances = data.split('\n')
    .filter(line => line.includes('Model Performance'))
    .map(line => JSON.parse(line.substring(line.indexOf('{'))));


const results = performances
    .map(result => {
            return {
                activation: result.activation,
                dropoutRate: result.dropoutRate,
                learningRate: result.learningRate,
                trainingLoss: +result.trainingLoss.at(-1).toFixed(3),
                trainingAccuracy: +result.trainingAccuracy.at(-1).toFixed(3),
                testLoss: +result.testLoss[0].toFixed(3),
                testAccuracy: +result.testAccuracy[0].toFixed(3),
                fitting: +(result.trainingAccuracy.at(-1) - result.testAccuracy[0]).toFixed(3)
            }

        }
    )
    .sort((a, b) => b.testAccuracy - a.testAccuracy);

console.table(results);
