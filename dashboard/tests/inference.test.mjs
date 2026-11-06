import {readFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createPredictor, tokenize} from '../inference.mjs';

const dataPath = new URL('../data/', import.meta.url);
const readJSON = async name => JSON.parse(await readFile(new URL(`${name}.json`, dataPath), 'utf8'));
const metadata = await readJSON('inference');
const vocabulary = await readJSON('vocabulary');
const fixtures = await readJSON('inference_fixtures');
const weights = await readFile(new URL('dense_weights.bin', dataPath));
const buffer = weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength);
const predict = createPredictor(metadata, vocabulary, buffer);

test('JavaScript tokenizer matches saved Keras vectorizer', () => {
  for (const fixture of fixtures) {
    const tokens = tokenize(fixture.text, vocabulary);
    assert.deepEqual(tokens, fixture.token_ids.filter(id => id !== 0), fixture.text);
  }
  assert.equal(tokenize('hello '.repeat(100), vocabulary).length, 64);
});

test('browser dense predictions reproduce actual TensorFlow outputs', () => {
  for (const fixture of fixtures) {
    const actual = predict(fixture.text);
    for (let i = 0; i < actual.length; i++) assert.ok(Math.abs(actual[i] - fixture.probabilities[i]) < 2e-6, `${fixture.text}: class ${i}`);
    assert.ok(Math.abs(actual.reduce((a, b) => a + b, 0) - 1) < 1e-10);
  }
});

test('blank or punctuation-only text is rejected rather than returning invalid scores', () => {
  for (const text of ['', '  ', '!!!...']) assert.throws(() => predict(text), /at least one word/);
});
