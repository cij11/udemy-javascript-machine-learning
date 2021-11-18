

const outputs = [];


function distance(pointA, pointB) {
  return Math.abs(pointA - pointB)
}

const k = 10


function knn(data, point) {
 return _.chain(data)
 .map(row => [distance(row[0], point), row[3]]) // map to distance from to predition point : bucket it landed in
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
  const testSetSize = 10
  const [testSet, trainingSet] = splitDataset(outputs, testSetSize);

  const accuracy = _.chain(testSet)
  .filter(testPoint => 
    knn(trainingSet, testPoint[0]) === testPoint[3]
  )
  .size() // Same as length for a collectin
  .divide(testSetSize)
  .value()

  console.log('Accuracy: ', accuracy)
}

function splitDataset(data, testCount) {
  const shuffled = _.shuffle(data)

  const testSet = _.slice(shuffled, 0, testCount);
  const trainingSet = _.slice(shuffled, testCount)

  return [testSet, trainingSet]
}