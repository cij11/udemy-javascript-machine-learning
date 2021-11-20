

const outputs = [];


function distance(pointA, pointB) {
  return _.chain(pointA)
    .zip(pointB) // Create pairs of values for the same feature
    .map(([a, b]) => (a - b) ** 2) // square
    .sum() 
    .value() ** 0.5 // sqrt
}


function knn(data, point, k) {
 return _.chain(data)
 .map(row => {
   return [
     distance(_.initial(row), point),  // don't do initial(point) because we may pass in points that don't have a 
     _.last(row)
    ]
}) // map to distance from to predition point : bucket it landed in
 .sortBy(row => row[0]) // sort by distnance to prediction point
 .slice(0, k) // select top k rows
 .countBy(row => row[1]) // Count by bucket they fell in
 .toPairs() // Convert object containing keys: counts to array of [key, count] 
 .sortBy(row => row[1]) // Sort by count, fewest to most
 .last() // Get last (highest count)
 .first() // Get the bucket number
 .parseInt() 
 .value()
}


function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([
    dropPosition, bounciness, size, bucketLabel
  ])
}

function runAnalysis() {
  const testSetSize = 50
  const [testSet, trainingSet] = splitDataset(outputs, testSetSize);

  _.range(1, 20).forEach(k => {

    const accuracy = _.chain(testSet)
    .filter(testPoint => 
      knn(trainingSet, _.initial(testPoint), k) === testPoint[3]
    )
    .size() // Same as length for a collectin
    .divide(testSetSize)
    .value()

    console.log(`Accuracy for k ${k} = `, accuracy)

  })



}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data)

  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount)

  return [testSet, trainingSet]
}