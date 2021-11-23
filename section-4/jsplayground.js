const features = tf.tensor([
  [-121, 47],
  [-121.2, 46.5],
  [-122, 46.6],
  [-120.9, 46.7]
])

const labels = tf.tensor([
  [200],
  [250],
  [215],
  [240]
])

const predictionPoint = tf.tensor([-121, 47])

const k = 2


const av = features.sub(predictionPoint)
  .pow(2)
  .sum(1)
  .pow(0.5)
  .expandDims(1)
  .concat(labels, 1)
  .unstack() // Unpack tensor into a regular javascript ARRAY of tensors
  .sort((a, b) => a.get(0) > b.get(0) ? 1 : -1)
  .slice(0, k) // array version of slice
  .reduce((acc, pair) => acc + pair.get(1), 0) / k
  

