const tf = require('@tensorflow/tfjs-node'); // only need to put require @tensorflow/node-tfjs in one file of the project. Preferably index.js
const _ = require('lodash')

class LinearRegression {
  constructor(features, labels, options) { // assume features and labels are already tensorflow tensors
    this.features = this.processFeatures(features)
    this.labels = tf.tensor(labels);

    this.options = Object.assign({
      learningRate: 0.1, iterations: 1000
    }, options)

    this.weights = tf.zeros([2, 1])
    // 2 by 1 matrix
    // With the way we're calculating gradient descent:
    // 0, 0 = b
    // 1, 0 = m
  }

  gradientDescent() {
    // Slop of mean squared error (MSE) with respect to M and B:
    // =  Features * ( ( Features * Weights) - Labels) / n
    const currentGuesses = this.features.matMul(this.weights)
    const differences = currentGuesses.sub(this.labels)


    const slopes = this.features // calculate the gradients
    .transpose() // Need to transpose so that shapes allow multiplication
    .matMul(differences)
    .div(this.features.shape[0]) // divide by n

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate)) // Subtract gradients * learning rates from previous weights
  }

  // Iterative javsacript gradient descent. Replaced with tensorflow implementation.
  // gradientDescent() {
  //   const currentGuessesForMPG = this.features.map(row => {
  //     return this.m * row[0] + this.b
  //   })

  //   const bSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
  //     return guess - this.labels[i][0]
  //   })) * (2 / this.features.length)

  //   const mSlope = _.sum(currentGuessesForMPG.map((guess, i) => {
  //     return -1 * this.features[i][0] * (this.labels[i][0] - guess)
  //   })) * 2

  //   this.b = this.b - bSlope * this.options.learningRate; 
  //   this.m = this.m - mSlope * this.options.learningRate;
  // }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent()
    }
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures)
    testLabels = tf.tensor(testLabels)

     // Multiplying features by weights essentially does y = mx + b for entire dataset
     // Note: Features must be on the left, for dimensions/shapes to match
    const predictions = testFeatures.matMul(this.weights)

    predictions.print()

    const ssRes =  testLabels.sub(predictions) // Sum of Squares of Residuals
      .pow(2)
      .sum()
      .arraySync()

    const ssTot = testLabels.sub(testLabels.mean())// Total sum of squares
      .pow(2)
      .sum()
      .arraySync()

      //Coefficient of determination
      // Closer to 1 is closer to a perfect fit.
      // Negative is worse than just taking the mean
  
      return 1 - ssRes / ssTot
  }

  // Scale and prepend 1's. This needs to be applied to training and test features
  processFeatures(features) {
    features = tf.tensor(features)

    // If mean + variance aren't defined, need to do the first time standardisation (i.e, before/during training)
    // Otherwise, we're standardising test data, and need to re-apply the training standardisation (from the training mean+variance), and NOT use the test features.
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

    this.mean = mean
    this.variance = variance

    return features.sub(mean).div(variance.pow(0.5))
  }
}

module.exports = LinearRegression