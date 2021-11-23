const tf = require('@tensorflow/tfjs-node')
require('@tensorflow/tfjs-node') // Instruct tensorflow to run on cpu by default
const loadCSV = require('./load-csv')

function knn(features, labels, predictionPoint, k) {
  return features.sub(predictionPoint)
  .pow(2)
  .sum(1)
  .pow(0.5)
  .expandDims(1)
  .concat(labels, 1)
  .unstack() // Unpack tensor into a regular javascript ARRAY of tensors
  .sort((a, b) => a.arraySync()[0] > b.arraySync()[0] ? 1 : -1) //a.arraySync()[0], because a.get(0) not defined in this version of tf
  .slice(0, k) // array version of slice
  .reduce((acc, pair) => acc + pair.arraySync()[1], 0) / k // Average the values. Again, using arraysync, because no get()
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

testFeatures.forEach((testPoint, i) => {
  const result = knn(featuresT, labelsT, tf.tensor(testPoint), 10)
  const err = (testLabels[i][0] - result) / testLabels[i][0]
  console.log('Error: ', err)
})

// console.log('Guess', result)
