const tf = require('@tensorflow/tfjs-node'); // only need to put require @tensorflow/node-tfjs in one file of the project. Preferably index.js
const _ = require('lodash')

class LogisticRegression {
  constructor(features, labels, options) { // assume features and labels are already tensorflow tensors
    this.features = this.processFeatures(features)
    this.labels = tf.tensor(labels);
    this.costHistory = [] // Cost (Here calculated from cross entropy) history.

    this.options = Object.assign({
      learningRate: 0.1, iterations: 1000, batchSize: 10, decisionBoundary: 0.5
    }, options)

    this.weights = tf.zeros([this.features.shape[1], this.labels.shape[1]])
  }

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).softmax()
    const differences = currentGuesses.sub(labels)


    const slopes = features // calculate the gradients
    .transpose() // Need to transpose so that shapes allow multiplication
    .matMul(differences)
    .div(features.shape[0]) // divide by n

    return this.weights.sub(slopes.mul(this.options.learningRate)) // Subtract gradients * learning rates from previous weights
  }

  train() {
    const { batchSize } = this.options
    const batchQuantity = Math.floor(this.features.shape[0] / batchSize)
    for (let i = 0; i < this.options.iterations; i++) {
      for(let j = 0; j < batchQuantity; j++) {
        const startIndex = j * batchSize;

        this.weights = tf.tidy(() => { // Wrap in tf.tidy to clean up all tensors, except those that are returned function
          const featureSlice = this.features.slice(
            [startIndex, 0], 
            [batchSize, -1]
          )
          const labelSlice = this.labels.slice([
            startIndex, 0], 
            [batchSize, -1]
          )
  
          return this.gradientDescent(featureSlice, labelSlice) // Slice: (startcoords eg, current row: i * batchSize, first column: 0), (size-of-slice eg (batchSize (rows), -1 (all columns))))
        })
      }

      this.recordCost() // If cost is going up, learning rate needs to be reduced
      this.updateLearningRate()
    }
  }

  predict(observations) {
    return observations = this.processFeatures(observations)
      .matMul(this.weights)
      .softmax()
      .argMax(1)
  }

  test(testFeatures, testLabels) {
    // To test:
    // Round predictions (to turn fraction into 0 or 1)
    // Subtract + abs (so that 0 == correct and 1 = incorrect)
    // Sum and take incorrect / total
     const predictions = this.predict(testFeatures)
     testLabels = tf.tensor(testLabels).argMax(1) // Store the index of the column with the highest value for that row

     const incorrect = predictions.notEqual(testLabels) // If pred != actual, this will be 1
     .sum() // = number of incorrect guesses
     .arraySync()

     return (predictions.shape[0] - incorrect) / predictions.shape[0]
  }

  processFeatures(features) {
    features = tf.tensor(features)

    if (this.mean && this.variance) {
      features = features.sub(this.mean).div(this.variance.pow(0.5))
    } else {
      features = this.standardize(features)
    }

    features = tf.ones([features.shape[0], 1]).concat(features, 1)

    return features
  }

  standardize(features) {
    const { mean, variance } = tf.moments(features, 0)

    const filler =  variance.cast('bool') // Cast to bool, invert, cast back to float, then sum.
    .logicalNot()
    .cast('float32') 
  
    this.mean = mean
    this.variance = variance.add(filler) // Add filler. The effect of this is just to turn 0's to 1's. This prevents divide by 0's filling the tensor with NaN

    debugger

    return features.sub(this.mean).div(this.variance.pow(0.5))
  }

  recordCost() {
    // Cross Entropy = 
    // -(1/n) * (ActualTransposed * log(Guesses)) + (1 - Actual)Transposed * Log(1-Guesses)
    const guesses = this.features.matMul(this.weights)
    .softmax()

    const termOne = this.labels
      .transpose()
      .matMul(guesses.log())

    const termTwo = this.labels
      .mul(-1) // -Actual
      .add(1) // 1 - Actual
      .transpose()
      .matMul(
        guesses.mul(-1) // - guesses
        .add(1)
        .log() // log (1 - guesses)
      )

    const cost = termOne.add(termTwo)
        .div(this.features.shape[0])
        .mul(-1) // * -1/n
        .arraySync()[0,0]

    this.costHistory.unshift(cost)
  }

  updateLearningRate() {
    if (this.costHistory.length < 2) {
      return;
    }

    if (this.costHistory[0] > this.costHistory[1]) {
      this.options.learningRate /= 2
    } else {
      this.options.learningRate *= 1.05
    }
  }
}

module.exports = LogisticRegression