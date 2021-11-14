const outputs = [];

const predictionPoint = 300

function distance(point) {
  return Math.abs(point - predictionPoint)
}

const k = 10




function onScoreUpdate(dropPosition, bounciness, size, bucketLabel) {
  outputs.push([
    dropPosition, bounciness, size, bucketLabel
  ])
}

function runAnalysis() {
  const bucket = _.chain(outputs)
  .map(row => [distance(row[0]), row[3]]) // map to distance from to predition point : bucket it landed in
  .sortBy(row => row[0]) // sort by distnance to prediction point
  .slice(0, k) // select top k rows
  .countBy(row => row[1]) // Count by bucket they fell in
  .toPairs() // Convert object containing keys: counts to array of [key, count] 
  .sortBy(row => row[1]) // Sort by count, fewest to most
  .last() // Get last (highest count)
  .first() // Get the bucket number
  .parseInt() 
  .value()

  console.log("Your point will probably fall into", bucket)

}

