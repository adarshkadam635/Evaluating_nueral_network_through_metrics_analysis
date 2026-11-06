# Dataset provenance

The selected dataset is the **TweetEval sentiment task**, sourced from Cardiff NLP's
[official repository](https://github.com/cardiffnlp/tweeteval/tree/main/datasets/sentiment).
It matches this project's original Twitter sentiment classification task.

- Immutable source revision: `4fbd22cd78421f05b1ecdb4fc5725bc7a7bd8f66`.
- Original splits: 45,615 training, 2,000 validation, 12,284 test tweets.
- Labels: `0 = negative`, `1 = neutral`, `2 = positive`.
- The six downloaded UTF-8 text/label files are pinned by SHA-256 in
  `dataset_checksums.json`; `train.py` verifies those hashes on every run.
- Raw tweets are cached under ignored `data/`, rather than redistributed here.

## Attribution and use

TweetEval: Francesco Barbieri, Jose Camacho-Collados, Luis Espinosa-Anke, and
Leonardo Neves (2020), [TweetEval: Unified Benchmark and Comparative Evaluation
for Tweet Classification](https://aclanthology.org/2020.findings-emnlp.148/).

Sentiment source: Sara Rosenthal, Noura Farra, and Preslav Nakov (2017),
[SemEval-2017 Task 4: Sentiment Analysis in Twitter](https://aclanthology.org/S17-2088/).

The [upstream dataset card](https://huggingface.co/datasets/cardiffnlp/tweet_eval/blob/main/README.md)
lists the sentiment subset under **Creative Commons Attribution 3.0 Unported**;
other TweetEval tasks have different terms. Consult the upstream source and terms
for your intended use. No new license is asserted for the upstream data.

## Split hygiene and preprocessing

The official test set is kept intact, including any internal repeated test tweets.
For matching only, text is case-folded, whitespace collapsed, and stripped.
Validation excludes test overlaps, internal duplicates, and all examples whose
normalized text has conflicting labels within validation. Training likewise
excludes validation/test overlaps, internal duplicates, and conflicting labels.
The manifest records original, removed, and effective counts and class counts.
Consequently the cleaned training/validation results are not an exact reproduction
of the unmodified TweetEval benchmark protocol.

Keras TextVectorization learns at most 15,000 tokens exclusively from the cleaned
training set and pads/truncates to 64 tokens. It uses the built-in lowercase and
punctuation-stripping standardizer. Stopwords and negations are retained. Embeddings
are trained from scratch for each architecture; no GloVe download is needed.

This changes the original notebook's binary positive-versus-other mapping to the
dataset's three explicit labels. Historical English Twitter data and its topic,
class, and temporal distribution limit generalization. These models should not be
interpreted as an unbiased measure of individuals' emotions or intent.
