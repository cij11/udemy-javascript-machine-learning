require('@tensorflow/tfjs-node') // Instruct tensorflow to run on cpu by default
const tf = require('@tensorflow/tfjs-node')
const loadCSV = require('./load-csv')

function knn(features, labels, predictionPoint, k) {
  return features.sub(predictionPoint)
  .pow(2)
  .sum(1)
  .pow(0.5)
  .expandDims(1)
  .concat(labels, 1)
  .unstack() // Unpack tensor into a regular javascript ARRAY of tensors
  .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
  .slice(0, k) // array version of slice
  .reduce((acc, pair) => acc + pair.get(1), 0) / k
}

let { features, labels, testFeatures, testLabels } = loadCSV('kc_house_data.csv', {
  shuffle: true,
  splitTest: 10,
  dataColumns: ['lat', 'long'],
  labelColumns: ['price']
});

console.log('---------------')


const featuresT = tf.tensor(features)
const labelsT = tf.tensor(labels)

const result = knn(featuresT, labelsT, tf.tensor(testFeatures[0]), 10)

console.log('Guess', result)
console.log(testLabels[0][0])

console.log('Accuracy: ' )