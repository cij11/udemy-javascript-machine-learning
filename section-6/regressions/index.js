require('@tensorflow/tfjs-node')
const LinearRegression = require('./linear-regression')

const tf = require('@tensorflow/tfjs')

const loadCSV = require('./load-csv')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower'],
  labelColumns: ['mpg']
})

const regression = new LinearRegression(features, labels, {
  learningRate: 0.0001, 
  iterations: 100
})

regression.train()

console.log('Updated M is: ', regression.m, 'Updated B is: ', regression.b)