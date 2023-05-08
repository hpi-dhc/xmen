# :heavy_multiplication_x:MEN

xMEN is an extensible toolkit for Cross-lingual (**x**) **M**edical **E**ntity **N**ormalization.
Through its compatibility with the [BigBIO (BigScience Biomedical)](https://github.com/bigscience-workshop/biomedical) framework, it can be used out-of-the box with many open biomedical datasets.

### Installation

xMEN is available through PyPi: `pip install xmen`

### Development

We use [Poetry](https://python-poetry.org/) for building, testing and dependency management (see [pyproject.toml](pyproject.toml)).

## :open_file_folder: Data Loading

Usually, BigBIO-compatible datasets can just be loaded from the Hugging Face Hub:

```
from datasets import load_dataset
dataset = load_dataset("distemist", "distemist_linking_bigbio_kb")
```

### Integration with NER Tools

To use xMEN with existing NER pipelines, you can also create a dataset at runtime.

### spaCy

```
from xmen.data import from_spacy
docs = ... #  list of spaCy docs with entity spans
dataset = from_spacy(docs)
```

## :wrench: Configuration and CLI

xMEN provides a convenient command line interface to prepare entity linking pipelines by creating target dictionaries and pre-computing indices to link to concepts in them.

Run `xmen help` to get an overview of the available commands.

Configuration is done through `.yaml` files. For examples, see the [conf](/conf) folder.

## :closed_book: Creating Dictionaries

Run `xmen dict` to create dictionaries to link against. Although the most common use case is to create subsets of the UMLS, it also supports passing custom parser scripts for non-UMLS dictionaries.

**Note**: Creating UMLS subsets requires a local installation of the [UMLS metathesaurus](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) (not only MRCONSO.RRF). In the examples, we assume that the environment variable `$UMLS_HOME` points to the installation path. You can either set this variable, or replace the path with your local installation.

### UMLS Subsets

Example configuration for [Medmentions](https://github.com/chanzuckerberg/MedMentions):

```
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

Running `xmen --dict conf/medmentions.yaml` creates a `.jsonl` file from the described UMLS subset.

### Using Custom Dictionaries

Parsing scripts for custom dictionaries can be provided with the `--code` option (examples can be found in the [dicts](/dicts) folder).

Example configuration for [DisTEMIST](https://temu.bsc.es/distemist/):

```
name: distemist

dict:
  custom:
    lang: 
      - es
    distemist_path: path/to/dictionary_distemist.tsv
```

Running `xmen dict conf/distemist.yaml --code dicts/distemist.py --key distemist_gazetteer` creates a `.jsonl` file from the custom DisTEMIST gazetteer.

## :mag_right: Candidate Generation

The `xmen index` command is used to compute term indices from a dictionary created through the `dict` command.
If an index already exists, you will be prompted to overwrite the existing file (or pass `--overwrite`).

xMEN provides implementations of different neural and non-neural candidate generators

### TF-IDF Weighted Character N-grams

Based on the implementation from [scispaCy](https://github.com/allenai/scispacy).

YAML file:

```
linker:
  candidate_generation:
    ngram:
      k: 100
```


Run `xmen index my_config.yaml --ngram` or `xmen index my_config.yaml --all` to create the index.

Example usage: see [notebooks/BioASQ_DisTEMIST.ipynb](notebooks/BioASQ_DisTEMIST.ipynb)

### SapBERT

Dense Retrieval based on [SapBERT](https://github.com/cambridgeltl/sapbert) embeddings.

YAML file:

```
linker:
  candidate_generation:
    sapbert:
      embedding_model_name: cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR
      k: 1000
```

Run `xmen index my_config.yaml --sapbert` or `xmen index my_config.yaml --all` to create the [FAISS](https://github.com/facebookresearch/faiss) index.

Example usage: see [notebooks/BioASQ_DisTEMIST.ipynb](notebooks/BioASQ_DisTEMIST.ipynb)

### Ensemble

TODO

Example usage: see [notebooks/BioASQ_DisTEMIST.ipynb](notebooks/BioASQ_DisTEMIST.ipynb)

## :cyclone: Rerankers

### Cross-Encoder Reranker

TODO

Example usage:see [notebooks/BioASQ_DisTEMIST.ipynb](notebooks/BioASQ_DisTEMIST.ipynb)

## :bulb: Pre- and Post-Processing

We support various optional components for transforming input data and result sets:

- [Sampling](xmen/preprocessing/sampling.py)
- [Abbrevation expansion](xmen/preprocessing/abbrevations.py)
- [Filtering by UMLS semantic groups](xmen/preprocessing/semantic_groups.py)
- [Filtering by UMLS semantic types](xmen/preprocessing/semantic_types.py)
- [Replacement of retired CUIS](xmen/preprocessing/retired_cuis.py)

## :bar_chart: Evaluation

xMEN provides implementations of common entity linking metrics (e.g., a wrapper for [neleval](https://github.com/wikilinks/neleval))

Example usage: see [notebooks/BioASQ_DisTEMIST.ipynb](notebooks/BioASQ_DisTEMIST.ipynb)

## :chart_with_upwards_trend: Benchmark Results

TODO
