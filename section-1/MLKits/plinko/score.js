

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
  const testSetSize = 100
  const k = 10;
  
  _.range(0, 3).forEach(feature => {
    const data = _.map(outputs, row => [row[feature], _.last(row)])

    const [testSet, trainingSet] = splitDataset(minMax(data, 1), testSetSize);

    const accuracy = _.chain(testSet)
    .filter(testPoint => 
      knn(trainingSet, _.initial(testPoint), k) === _.last(testPoint)
    )
    .size() // Same as length for a collection
    .divide(testSetSize)
    .value()

    console.log(`Accuracy for feature ${feature} = `, accuracy)

  })
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data)

  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount)

  return [testSet, trainingSet]
}

function minMax(data, featureCount) {
  const clonedData = _.cloneDeep(data)

  for(let i = 0; i < featureCount; i++) {
    const column = clonedData.map(row => row[i]) // Extract the i'th column
    
    const min = _.min(column)
    const max = _.max(column)

    for(let j = 0; j < clonedData.length; j++) {
      clonedData[j][i] = (clonedData[j][i] - min) / (max - min)
    }
  }

  return clonedData
}