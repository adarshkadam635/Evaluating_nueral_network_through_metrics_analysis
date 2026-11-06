import {createPredictor} from './inference.mjs';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const number = new Intl.NumberFormat('en-US');
const percent = value => `${(value * 100).toFixed(2)}%`;
const score = value => value.toFixed(4);
const names = {dense: 'Dense', cnn: 'CNN', lstm: 'LSTM'};
const colors = {dense: '#2855d9', cnn: '#087f8c', lstm: '#7955c8', train: '#2855d9', validation: '#d88b36', negative: '#c86152', neutral: '#5683c9', positive: '#309781'};
const state = {model: 'dense', split: 'test', normalized: false};
let data;
let predictorPromise;

async function getJSON(name) {
  const response = await fetch(`data/${name}.json`);
  if (!response.ok) throw new Error(`Unable to load ${name}: HTTP ${response.status}`);
  return response.json();
}

function legend(items) {
  return `<div class="legend">${items.map(item => `<span><i style="background:${item.color}"></i>${item.label}</span>`).join('')}</div>`;
}

function renderKpis() {
  const bestAccuracy = Math.max(...Object.values(data.metrics).map(m => m.test.accuracy));
  const bestF1 = Math.max(...Object.values(data.metrics).map(m => m.test.macro_f1));
  const trainCount = data.manifest.deduplication.train.kept_rows;
  $('#kpis').innerHTML = [
    {label: 'Best test accuracy', icon: '↗', value: `${(bestAccuracy * 100).toFixed(2)}<small>%</small>`, foot: `<b>+${((bestAccuracy - data.summary.majority_baseline.test_accuracy) * 100).toFixed(2)} pp</b> vs majority baseline`, tag: 'CNN'},
    {label: 'Best test macro F1', icon: '◎', value: bestF1.toFixed(4), foot: 'Equal weight to every sentiment', tag: 'LSTM'},
    {label: 'Training examples', icon: '▤', value: number.format(trainCount), foot: 'Cleaned · 3 sentiment classes'},
    {label: 'Held-out test tweets', icon: '⊞', value: number.format(data.manifest.deduplication.test.kept_rows), foot: 'Official test split · unchanged'},
  ].map(kpi => `<article class="kpi"><div class="kpi-label">${kpi.label}<span class="kpi-icon" aria-hidden="true">${kpi.icon}</span></div><div class="kpi-value">${kpi.value}</div><div class="kpi-foot">${kpi.tag ? `<span class="mini-pill">${kpi.tag}</span>` : ''}${kpi.foot}</div></article>`).join('');
  $('#selection-score').textContent = score(data.metrics.dense.validation.macro_f1);
}

function renderComparison() {
  const entries = Object.entries(data.metrics);
  $('#comparison-chart').innerHTML = legend([{label: 'Accuracy', color: colors.train}, {label: 'Macro F1', color: '#96b4f4'}]) +
    `<div class="comparison-bars">${entries.map(([name, model]) => `<div class="bar-row"><span class="bar-name">${names[name]}</span><div class="paired-bars">${['accuracy', 'macro_f1'].map((metric, i) => `<div class="bar-track ${i ? 'alt' : ''}" role="img" aria-label="${names[name]} ${metric}: ${percent(model[state.split][metric])}"><div class="bar-fill" style="width:${model[state.split][metric] * 100}%"></div><span>${percent(model[state.split][metric])}</span></div>`).join('')}</div></div>`).join('')}</div>` +
    `<div class="baseline"><span>Majority baseline · test accuracy</span><strong>${percent(data.summary.majority_baseline.test_accuracy)}</strong></div>`;
  const descriptions = {dense: 'Mean pooling → Dense 64 → Softmax', cnn: 'Conv1D 64 × 3 → Max pool → Softmax', lstm: 'LSTM 32 → Softmax'};
  $('#architecture-cards').innerHTML = entries.map(([name, model]) => `<div class="architecture"><div class="model-letter ${name}" aria-hidden="true">${name === 'dense' ? 'D' : name === 'cnn' ? 'C' : 'L'}</div><div><div class="architecture-title">${names[name]} ${name === data.summary.best_model ? '<span class="mini-pill">SELECTED</span>' : ''}</div><p>${descriptions[name]}</p></div><div class="architecture-meta">${(model.parameters / 1e6).toFixed(2)}M<small>${model.training_seconds.toFixed(1)}s training</small></div></div>`).join('');
  const columns = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1', 'roc_auc_ovr_macro', 'log_loss'];
  $('#metrics-table tbody').innerHTML = entries.map(([name, model]) => `<tr class="${name === state.model ? 'current' : ''}"><td><button class="model-name-button" data-model="${name}" aria-label="Explore ${names[name]} model">${names[name]} ${name === data.summary.best_model ? '<span class="mini-pill">SELECTED</span>' : ''}</button></td>${columns.map(key => {
    const value = model[state.split][key];
    const best = (key === 'log_loss' ? Math.min : Math.max)(...entries.map(([, m]) => m[state.split][key]));
    return `<td class="${value === best ? 'metric-best' : ''}">${key === 'accuracy' ? percent(value) : score(value)}</td>`;
  }).join('')}</tr>`).join('');
  $$('.current-split').forEach(el => { el.textContent = state.split === 'test' ? 'test' : 'validation'; });
}

function lineChart(container, series, options = {}) {
  const W = 530, H = 252, left = 42, right = 16, top = 14, bottom = 38;
  const innerW = W - left - right, innerH = H - top - bottom;
  const xs = series.flatMap(s => s.points.map(p => p[0]));
  const ys = series.flatMap(s => s.points.map(p => p[1]));
  const xmin = options.xmin ?? Math.min(...xs), xmax = options.xmax ?? Math.max(...xs);
  const ymin = options.ymin ?? Math.max(0, Math.floor((Math.min(...ys) - .03) * 10) / 10);
  const ymax = options.ymax ?? Math.min(1.2, Math.ceil((Math.max(...ys) + .03) * 10) / 10);
  const x = value => left + (value - xmin) / (xmax - xmin || 1) * innerW;
  const y = value => top + (1 - (value - ymin) / (ymax - ymin || 1)) * innerH;
  const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${options.label}"><title>${options.label}</title><desc>${escape(series.map(s => `${s.label}: ${s.points.length} plotted observations`).join('. '))}</desc>`;
  for (let i = 0; i <= 4; i++) {
    const value = ymin + (ymax - ymin) * i / 4;
    const yy = y(value);
    svg += `<line x1="${left}" y1="${yy}" x2="${W - right}" y2="${yy}" stroke="#eaf0f6" stroke-dasharray="3 4"/><text x="${left - 9}" y="${yy + 3}" text-anchor="end" class="chart-axis">${options.percent ? Math.round(value * 100) + '%' : value.toFixed(2)}</text>`;
  }
  const ticks = options.epochs ? Array.from({length: xmax}, (_, i) => i + 1) : Array.from({length: 5}, (_, i) => xmin + (xmax - xmin) * i / 4);
  ticks.forEach(value => { svg += `<text x="${x(value)}" y="${H - 17}" text-anchor="middle" class="chart-axis">${options.epochs ? value : value.toFixed(2)}</text>`; });
  svg += `<text x="${left + innerW / 2}" y="${H - 1}" text-anchor="middle" class="chart-axis">${options.xlabel || 'Epoch'}</text>`;
  if (options.diagonal) svg += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#c5cfdf" stroke-dasharray="4 5"/>`;
  if (options.bestEpoch) svg += `<line x1="${x(options.bestEpoch)}" x2="${x(options.bestEpoch)}" y1="${top}" y2="${top + innerH}" stroke="#bccbe6" stroke-dasharray="3 4"><title>Restored checkpoint: epoch ${options.bestEpoch}</title></line>`;
  for (const s of series) {
    const path = s.points.map(([xx, yy], i) => `${i ? 'L' : 'M'}${x(xx).toFixed(2)},${y(yy).toFixed(2)}`).join(' ');
    svg += `<path d="${path}" fill="none" stroke="${s.color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>`;
    if (options.epochs) for (const [xx, yy] of s.points) svg += `<circle cx="${x(xx)}" cy="${y(yy)}" r="3.5" fill="white" stroke="${s.color}" stroke-width="2"><title>${s.label} · epoch ${xx}: ${yy.toFixed(4)}</title></circle>`;
  }
  svg += '</svg>';
  container.innerHTML = svg + legend(series);
}

function renderTraining() {
  const model = data.metrics[state.model];
  const history = data.histories[state.model];
  for (const metric of ['accuracy', 'loss']) {
    lineChart($(`#${metric}-chart`), [
      {label: 'Training', color: colors.train, points: history[metric].map((value, i) => [i + 1, value])},
      {label: 'Validation', color: colors.validation, points: history[`val_${metric}`].map((value, i) => [i + 1, value])},
    ], {label: `${names[state.model]} training and validation ${metric} by epoch`, epochs: true, percent: metric === 'accuracy', bestEpoch: model.best_epoch});
  }
  $('#epoch-badge').textContent = `${model.epochs_run} epochs run`;
  $('#best-epoch').textContent = `Restored epoch ${model.best_epoch}`;
  $('#training-strip').innerHTML = `<span>Training time <strong>${model.training_seconds.toFixed(1)} seconds</strong></span><span>Parameters <strong>${number.format(model.parameters)}</strong></span><span>Early stopping <strong>val_loss · patience 2</strong></span><span>Optimizer <strong>Adam · 0.001</strong></span>`;
  $$('.current-model').forEach(el => { el.textContent = names[state.model]; });
  $$('[data-model]').forEach(el => {
    const selected = el.dataset.model === state.model;
    if (el.closest('.segmented')) { el.classList.toggle('selected', selected); el.setAttribute('aria-pressed', selected); }
  });
}

function renderEvaluation() {
  const metrics = data.metrics[state.model][state.split];
  const matrix = metrics.confusion_matrix;
  const max = Math.max(...matrix.flat());
  let html = '<div class="matrix-axis">PREDICTED SENTIMENT →</div><div class="confusion-wrap"><div></div>' + data.config.labels.map(label => `<span class="matrix-label">${label}</span>`).join('');
  matrix.forEach((row, i) => {
    html += `<span class="matrix-label">${data.config.labels[i]}</span>`;
    const total = row.reduce((a, b) => a + b, 0);
    row.forEach((count, j) => {
      const normalized = count / total;
      const intensity = state.normalized ? normalized : count / max;
      const alpha = 0.05 + intensity * 0.9;
      const value = state.normalized ? (normalized * 100).toFixed(1) + '%' : number.format(count);
      html += `<div class="matrix-cell" style="background:rgba(40,85,217,${alpha});color:${alpha > .57 ? '#fff' : '#324d82'}" title="True ${data.config.labels[i]}, predicted ${data.config.labels[j]}: ${number.format(count)} tweets (${percent(normalized)} of row)">${value}<small>${i === j ? 'correct' : 'misclassified'}</small></div>`;
    });
  });
  $('#confusion-matrix').innerHTML = html + '</div><div class="matrix-axis">True sentiment in rows · hover a cell for counts and percentages</div>';
  const curves = data.roc[state.model];
  lineChart($('#roc-chart'), curves.map(curve => ({label: `${curve.label[0].toUpperCase() + curve.label.slice(1)} · ${curve.auc.toFixed(3)}`, color: colors[curve.label], points: curve.points})), {label: `${names[state.model]} test one-vs-rest ROC curves; legend gives full-resolution AUC`, xmin: 0, xmax: 1, ymin: 0, ymax: 1, diagonal: true, xlabel: 'False positive rate → (y: true positive rate)'});
  $('#class-table tbody').innerHTML = data.config.labels.map(label => {
    const row = metrics.classification_report[label];
    return `<tr><td><span class="sentiment ${label}">${label}</span></td><td>${score(row.precision)}</td><td>${score(row.recall)}</td><td>${score(row['f1-score'])}</td><td>${number.format(row.support)}</td></tr>`;
  }).join('');
}

function renderDataset() {
  $('#distribution').innerHTML = ['train', 'val', 'test'].map(split => {
    const counts = data.manifest.effective_class_counts[split];
    const total = Object.values(counts).reduce((a, b) => a + b, 0);
    return `<div class="dist-row"><div class="dist-heading"><strong>${{train: 'Training', val: 'Validation', test: 'Test'}[split]}</strong><span>${number.format(total)} tweets</span></div><div class="stacked-bar">${data.config.labels.map(label => `<span class="${label}" style="width:${counts[label] / total * 100}%" title="${label}: ${number.format(counts[label])} (${percent(counts[label] / total)})">${(counts[label] / total * 100).toFixed(1)}%</span>`).join('')}</div></div>`;
  }).join('');
  $('#config-details').innerHTML = `<div class="config-grid"><div>Python <b>${data.config.python}</b></div><div>Seed <b>${data.config.seed}</b></div><div>Embeddings <b>${data.config.embedding_dim} dimensions</b></div><div>Maximum epochs <b>${data.config.epochs}</b></div><div>Vocabulary <b>${number.format(data.config.vocabulary_size)} tokens</b></div><div>Sequence <b>${data.config.sequence_length} tokens</b></div></div><p class="config-hash">Upstream revision: ${data.manifest.revision}</p>`;
}

function render() { renderComparison(); renderTraining(); renderEvaluation(); }

async function loadPredictor() {
  if (!predictorPromise) predictorPromise = Promise.all([getJSON('inference'), getJSON('vocabulary'), fetch('data/dense_weights.bin').then(response => {
    if (!response.ok) throw new Error('The trained model weights could not be loaded. Please try again.');
    return response.arrayBuffer();
  })]).then(([metadata, vocabulary, buffer]) => createPredictor(metadata, vocabulary, buffer)).catch(error => { predictorPromise = undefined; throw error; });
  return predictorPromise;
}

$('#prediction-form').addEventListener('submit', async event => {
  event.preventDefault();
  const text = $('#message').value.trim();
  const button = $('#predict-button');
  $('#prediction-error').hidden = true;
  if (!text) { $('#prediction-error').textContent = 'Enter an English message to analyze.'; $('#prediction-error').hidden = false; $('#message').focus(); return; }
  const original = button.innerHTML;
  button.disabled = true;
  button.textContent = 'Loading model & analyzing…';
  $('#prediction-result').setAttribute('aria-busy', 'true');
  try {
    const predict = await loadPredictor();
    const probabilities = predict(text);
    const winner = probabilities.indexOf(Math.max(...probabilities));
    $('#prediction-result').innerHTML = `<div class="prediction-caption">DENSE MODEL · VALIDATION-SELECTED CHECKPOINT</div><div class="predicted-sentiment" style="color:${colors[data.config.labels[winner]]}">${data.config.labels[winner]}</div>${data.config.labels.map((label, i) => `<div class="prediction-score-row ${label}"><span>${label}</span><div class="score-track"><div style="width:${probabilities[i] * 100}%"></div></div><strong>${percent(probabilities[i])}</strong></div>`).join('')}<p class="prediction-caption">Your message stays in this browser. Trained weights load once (~3.9 MB).</p>`;
  } catch (error) {
    $('#prediction-error').textContent = error.message;
    $('#prediction-error').hidden = false;
  } finally { button.disabled = false; button.innerHTML = original; $('#prediction-result').removeAttribute('aria-busy'); }
});

$$('[data-example]').forEach(button => button.addEventListener('click', () => {
  $('#message').value = button.dataset.example;
  $('#message').dispatchEvent(new Event('input'));
  $('#message').focus();
}));
$('#message').addEventListener('input', () => { $('#char-count').textContent = `${number.format($('#message').value.length)} / 5,000 characters`; });
document.addEventListener('click', event => {
  const modelButton = event.target.closest('[data-model]');
  if (modelButton && data) { state.model = modelButton.dataset.model; render(); }
  const splitButton = event.target.closest('[data-split]');
  if (splitButton && data) {
    state.split = splitButton.dataset.split;
    $$('[data-split]').forEach(button => { const selected = button === splitButton; button.classList.toggle('selected', selected); button.setAttribute('aria-pressed', selected); });
    render();
  }
});
$('#normalize').addEventListener('change', event => { state.normalized = event.target.checked; if (data) renderEvaluation(); });
$('#export').addEventListener('click', () => {
  if (!data) return;
  const fields = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1', 'weighted_f1', 'roc_auc_ovr_macro', 'log_loss'];
  const rows = [['model', 'split', ...fields], ...Object.entries(data.metrics).flatMap(([name, model]) => ['validation', 'test'].map(split => [name, split, ...fields.map(field => model[split][field])]))];
  const blob = new Blob([rows.map(row => row.join(',')).join('\n') + '\n'], {type: 'text/csv;charset=utf-8'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url; link.download = 'neural-metrics-evaluation.csv'; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

const observer = new IntersectionObserver(entries => {
  const visible = entries.filter(entry => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
  if (visible) $$('nav a').forEach(link => link.classList.toggle('active', link.hash === `#${visible.target.id}`));
}, {rootMargin: '-5% 0px -65% 0px', threshold: [0, .1, .5]});
$$('main section').forEach(section => observer.observe(section));

try {
  const [metrics, summary, manifest, config, histories, roc] = await Promise.all(['metrics', 'summary', 'dataset_manifest', 'config', 'histories', 'roc'].map(getJSON));
  data = {metrics, summary, manifest, config, histories, roc};
  renderKpis(); render(); renderDataset();
  $('#export').disabled = false;
  $('#loading').hidden = true;
} catch (error) {
  $('#loading').hidden = true;
  $('#load-error').textContent = `The experiment data could not be loaded. Refresh to try again. ${error.message}`;
  $('#load-error').hidden = false;
}
