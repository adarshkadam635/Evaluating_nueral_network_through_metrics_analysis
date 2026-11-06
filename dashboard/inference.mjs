// Exact trained dense network: tokenization → masked mean → Dense/ReLU → softmax.
// No remote API and no sentiment heuristics.
export function tokenize(text, vocabulary, sequenceLength = 64) {
  const index = vocabulary instanceof Map ? vocabulary : new Map(vocabulary.map((word, i) => [word, i]));
  // Match Keras lower_and_strip_punctuation, including stripping apostrophes.
  const standardized = text.toLowerCase().replace(/[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g, '');
  const words = standardized.trim().split(/\s+/u).filter(Boolean).slice(0, sequenceLength);
  return words.map(word => index.get(word) ?? 1);
}

export function createPredictor(metadata, vocabulary, buffer) {
  const tensors = metadata.tensors.map(tensor => ({...tensor, values: new Float32Array(buffer, tensor.offset * 4, tensor.length)}));
  const [embedding, hiddenKernel, hiddenBias, outputKernel, outputBias] = tensors;
  if (tensors.length !== 5 || embedding.shape[1] !== metadata.embedding_dim || outputBias.length !== 3) throw new Error('Unsupported dense model layout. Re-export the dashboard model.');
  const index = new Map(vocabulary.map((word, i) => [word, i]));
  return text => {
    const ids = tokenize(text, index, metadata.sequence_length);
    if (!ids.length) throw new Error('Enter a message with at least one word.');
    const mean = new Float64Array(metadata.embedding_dim);
    for (const id of ids) for (let j = 0; j < mean.length; j++) mean[j] += embedding.values[id * mean.length + j] / ids.length;
    const hidden = new Float64Array(hiddenBias.length);
    for (let j = 0; j < hidden.length; j++) {
      let sum = hiddenBias.values[j];
      for (let i = 0; i < mean.length; i++) sum += mean[i] * hiddenKernel.values[i * hidden.length + j];
      hidden[j] = Math.max(0, sum);
    }
    const logits = Array.from(outputBias.values);
    for (let j = 0; j < 3; j++) for (let i = 0; i < hidden.length; i++) logits[j] += hidden[i] * outputKernel.values[i * 3 + j];
    const max = Math.max(...logits);
    const exponentials = logits.map(value => Math.exp(value - max));
    const sum = exponentials.reduce((a, b) => a + b, 0);
    return exponentials.map(value => value / sum);
  };
}
