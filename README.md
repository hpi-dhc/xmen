[![Build](https://github.com/hpi-dhc/xmen/actions/workflows/python-app.yml/badge.svg)](https://github.com/hpi-dhc/xmen/actions/workflows/python-app.yml)
[![pypi Version](https://img.shields.io/pypi/v/xmen)](https://pypi.org/project/xmen/)
[![DOI](https://zenodo.org/badge/630365891.svg)](https://zenodo.org/doi/10.5281/zenodo.12699189)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/ambv/black)


# ✖️MEN

xMEN is an extensible toolkit for Cross-lingual (**x**) **M**edical **E**ntity **N**ormalization.
Through its compatibility with the [BigBIO (BigScience Biomedical)](https://github.com/bigscience-workshop/biomedical) framework, it can be used out-of-the box to run experiments with many open biomedical datasets. It can also be easily integrated with existing Named Entity Recognition (NER) pipelines.

### Installation

xMEN is available through [PyPi](https://pypi.org/project/xmen/): 

`pip install xmen`

**Note**:
If you encounter issues installing `xmen` from `pip`, please see here: https://github.com/hpi-dhc/xmen/issues/37

### Development

We use [Poetry](https://python-poetry.org/) for building, testing and dependency management (see [pyproject.toml](pyproject.toml)).

## 🚀 Getting Started

A very simple pipeline highlighting the main components of xMEN can be found in [notebooks/00_Getting_Started.ipynb](notebooks/00_Getting_Started.ipynb)

## 🎓 Examples

For more advanced use cases, check out the [examples](examples) folder. 

## 📂 Data Loading

Usually, BigBIO-compatible datasets can just be loaded from the Hugging Face Hub:

```python
from datasets import load_dataset
dataset = load_dataset("distemist", "distemist_linking_bigbio_kb")
```

### Integration with NER Tools

To use xMEN with existing NER pipelines, you can also create a dataset at runtime.

#### Span-based Formats

Any span-based annotation format (i.e., based on character offsets), can be converted to a xMEN-compatible dataset.
For instance, using [SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) predictions:

```python
from span_marker import SpanMarkerModel
sentences = ... # list of sentences
model = SpanMarkerModel.from_pretrained(...)
preds = model.predict(sentences)

from xmen.data import from_spans
dataset = from_spans(preds, sentences)
```

#### [spaCy](https://spacy.io/)

```python
from xmen.data import from_spacy
docs = ... #  list of spaCy docs with entity spans
dataset = from_spacy(docs)
```

for an example, see: [examples/02_spaCy_German.ipynb](examples/02_spaCy_German.ipynb)


## 🔧 Configuration and CLI

xMEN provides a convenient command line interface to prepare entity linking pipelines by creating target dictionaries and pre-computing indices to link to concepts in them.

Run `xmen help` to get an overview of the available commands.

Configuration is done through `.yaml` files. For examples, see the [/examples/conf](/examples/conf) folder.

## 📕 Creating Knowledge Bases / Dictionaries

Run `xmen dict` to create KBs to link against. Although the most common use case is to create subsets of the UMLS, it also supports passing custom parser scripts for non-UMLS dictionaries.

**Note**: Creating UMLS subsets requires a local installation of the [UMLS metathesaurus](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) (not only MRCONSO.RRF). In the examples, we assume that the environment variable `$UMLS_HOME` points to the installation path. You can either set this variable, or replace the path with your local installation.

### UMLS Subsets

Example configuration for [Medmentions](https://github.com/chanzuckerberg/MedMentions):

```yaml
name: medmentions

dict:
  umls:
    lang: 
      - en
    meta_path: ${oc.env:UMLS_HOME}/2017AA/META
    version: 2017AA
    semantic_types:
      - T005
      - T007
      - T017
      - T022
      - T031
      - T033
      - T037
      - T038
      - T058
      - T062
      - T074
      - T082
      - T091
      - T092
      - T097
      - T098
      - T103
      - T168
      - T170
      - T201
      - T204
    sabs:
      - CPT
      - FMA
      - GO
      - HGNC
      - HPO
      - ICD10
      - ICD10CM
      - ICD9CM
      - MDR
      - MSH
      - MTH
      - NCBI
      - NCI
      - NDDF
      - NDFRT
      - OMIM
      - RXNORM
      - SNOMEDCT_US
```

Running `xmen dict examples/conf/medmentions.yaml` creates a `.jsonl` file from the described UMLS subset.

### Using Custom KB Scripts

Parsing scripts for custom KBs can be provided with the `--code` option (examples can be found in the [dicts](/dicts) folder).

Example configuration for [DisTEMIST](https://temu.bsc.es/distemist/):

```yaml
name: distemist

dict:
  custom:
    lang: 
      - es
    gazetteer_path: local_files/dictionary_distemist.tsv
```

Running `xmen dict examples/conf/distemist.yaml --code examples/dicts/bsc_gazetteer.py` creates a `.jsonl` KB file from the custom DisTEMIST gazetteer (which you can download from [Zenodo](https://zenodo.org/record/6505583) and put into any folder, e.g., `local_files`). The script `bsc_gazetteer.py` can use any custom keys like `gazetteer_path` in the example to construct the custom KB.

## 🔎 Candidate Generation

The `xmen index` command is used to compute term indices from a dictionary created through the `dict` command.
If an index already exists, you will be prompted to overwrite the existing file (or pass `--overwrite`).

xMEN provides implementations of different neural and non-neural candidate generators

### TF-IDF Weighted Character N-grams

Based on the implementation from [scispaCy](https://github.com/allenai/scispacy).

Run `xmen index my_config.yaml --ngram` or `xmen index my_config.yaml --all` to create the index.

To use the linker at runtime, pass the index folder as an argument:

```python
from xmen.linkers import TFIDFNGramLinker

ngram_linker = TFIDFNGramLinker(index_base_path="/path/to/my/index/ngram", k=100)
predictions = ngram_linker.predict_batch(dataset)
```

### SapBERT

Dense Retrieval based on [SapBERT](https://github.com/cambridgeltl/sapbert) embeddings.

YAML file (optional, if you want to configure another Transformer model):

```yaml
linker:
  candidate_generation:
    sapbert:
      model_name: cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR
```

Run `xmen index my_config.yaml --sapbert` or `xmen index my_config.yaml --all` to create the [FAISS](https://github.com/facebookresearch/faiss) index.

To use the linker at runtime, pass the index folder as an argument. To make predictions on a batch of documents, you have to pass a batch size, as the SapBERT linker runs on the GPU by default:

```python
from xmen.linkers import SapBERTLinker

sapbert_linker = SapBERTLinker(
    index_base_path = "/path/to/my/index/sapbert",
    k = 1000
)
predictions = sapbert_linker.predict_batch(dataset, batch_size=128)
```

If you have loaded a yaml-config as a dictionary-like object, you may also just pass it as kwargs:

```python
sapbert_linker = SapBERTLinker(**config)
```

By default, SapBERT assumes a CUDA device is available. If you want to disable cuda, pass `cuda=False` to the constructor.


### Ensemble

Different candidate generators often work well for different kinds of entity mentions, and it can be helpful to combine their predictions.

In xMEN, this can be easily achieved with an `EnsembleLinker`:

```python
from xmen.linkers import EnsembleLinker

ensemble_linker = EnsembleLinker()
ensemble_linker.add_linker('sapbert', sapbert_linker, k=10)
ensemble_linker.add_linker('ngram', ngram_linker, k=10)
```

or (as a shortcut for the combination of `TFIDFNGramLinker` and `SapBERTLinker`):

```python
from xmen.linkers import default_ensemble

ensemble_linker = default_ensemble("/path/to/my/index/")
```

You can call `predict_batch` on the `EnsembleLinker` just as with any other linker.

Sometimes, you want to compare the ensemble performance to individual linkers and already have the candidate lists. To avoid recomputation, you can use the `reuse_preds` argument:

```python
prediction = ensemble_linker.predict_batch(dataset, 128, 100, reuse_preds={'sapbert' : predictions_sap, 'ngram' : predictions_ngram'})
```

## 🌀 Entity Rankers

### Cross-encoder Re-ranker

When labelled training data is available, a trainable re-ranker can improve ranking of candidate lists a lot.

To train a cross-encoder model, first create a dataset of mention / candidate pairs:

```python
from xmen.reranking.cross_encoder import CrossEncoderReranker, CrossEncoderTrainingArgs
from xmen import load_kb

# Load a KB from a pre-computed dictionary (jsonl) to obtain synonyms for concept encoding
kb = load_kb('path/to/my/dictionary.jsonl')

# Obtain prediction from candidate generator (see above)
candidates = linker.predict_batch(dataset)

ce_dataset = CrossEncoderReranker.prepare_data(candidates, dataset, kb)
```

Then you can use this dataset to train a supervised reranking model:

```python
# Number of epochs to train
n_epochs = 10

# Any BERT model, potentially language-specific
cross_encoder_model = 'bert-base-multilingual-cased'

args = CrossEncoderTrainingArgs(n_epochs, cross_encoder_model)

rr = CrossEncoderReranker()

# Fit the model
rr.fit(args, ce_dataset['train'].dataset, ce_dataset['validation'].dataset)

# Predict on test set
prediction = rr.rerank_batch(candidates['test'], ce_dataset['test'])
```

#### Note on Memory Usage
In most examples and benchmarks, we use 64 candidates as a batch size for the cross-encoder, which usually fit into 48GB of GPU memory. 
If you encounter memory issues, you can try reducing this number and/or using a smaller BERT model. See: [Issue #22](https://github.com/hpi-dhc/xmen/issues/22#issuecomment-1827648900)

### Pre-trained Cross-encoders

We provide pre-trained models, based on automatically translated versions of MedMentions (see [notebooks/01_Translation.ipynb](notebooks/01_Translation.ipynb)).

Instead of fitting the cross-encoder model, you can just load a pre-trained model, e.g., for French:

```python
rr = CrossEncoderReranker.load('phlobo/xmen-fr-ce-medmentions', device=0)
```

Models are available on the Hugging Face Hub: https://huggingface.co/models?library=xmen


## 💡 Pre- and Post-processing

We support various optional components for transforming input data and result sets in `xmen.data`:

- [Sampling](xmen/data/sampling.py)
- [Abbrevation expansion](xmen/data/abbrevations.py)
- [Filtering by UMLS semantic groups](xmen/data/semantic_groups.py)
- [Filtering by UMLS semantic types](xmen/data/semantic_types.py)
- [Replacement of retired CUIS](xmen/data/retired_cuis.py)

## 📊 Evaluation

xMEN provides implementations of common entity linking metrics (e.g., a wrapper for [neleval](https://github.com/wikilinks/neleval)) and utilities for error analysis.

```python
from xmen.evaluation import evaluate, error_analysis

# Runs the evaluation
eval_results = evaluate(ground_truth, predictions)

# Performs error analysis
error_dataframe = error_analysis(ground_truth, predictions))
```

## Citation

If you use xMEN in your work, please cite the following paper:

Florian Borchert, Ignacio Llorca, Roland Roller, Bert Arnrich, and Matthieu-P Schapranow. 
**xMEN: A Modular Toolkit for Cross-Lingual Medical Entity Normalization**. 
arXiv preprint arXiv:2310.11275 (2023). http://arxiv.org/abs/2310.11275.

BibTeX:

```bibtex
@article{
      borchert2023xmen,
      title={{xMEN}: A Modular Toolkit for Cross-Lingual Medical Entity Normalization}, 
      author={Florian Borchert and Ignacio Llorca and Roland Roller and Bert Arnrich and Matthieu-P. Schapranow},
      year={2023},
      url={https://arxiv.org/abs/2310.11275},
      journal={arXiv preprint arXiv:2310.11275}
}
```
