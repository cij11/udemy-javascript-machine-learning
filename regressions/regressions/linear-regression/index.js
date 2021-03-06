require('@tensorflow/tfjs-node')
const LinearRegression = require('./linear-regression')
const plot = require('node-remote-plot')

const tf = require('@tensorflow/tfjs')

const loadCSV = require('../load-csv')

let { features, labels, testFeatures, testLabels } = loadCSV('../data/cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1, 
  iterations: 3,
  batchSize: 10
})

regression.train()

plot({
  x: regression.mseHistory.reverse(),
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error'
})

const r2 = regression.test(testFeatures, testLabels)

console.log('R2: ', r2)

regression.predict([
  [120, 2, 380]
]).print()