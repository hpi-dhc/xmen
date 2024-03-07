# Examples

|Link|Language|Description|
|---|---|---|
|[01_BRONCO_German.ipynb](01_BRONCO_German.ipynb)|🇩🇪|Candidate generation and supervised re-ranking using the BRONCO corpus.<br>Shows how you can configure multiple dictionaries in the same config file.|
|[02_spaCy_German.ipynb](02_spaCy_German.ipynb)|🇩🇪|Using a spaCy NER model with xMEN<br>Shows how to build a pipeline without labelled data using candidate generation, type filtering and pre-trained re-rankers|
|[03_SNOMED_Linking_German.ipynb](03_SNOMED_Linking_German.ipynb)|🇩🇪|Linking against codes in UMLS source vocabularies (here SNOMED CT)|
|[04_Drug_Names_ATC.ipynb](04_Drug_Names_ATC.ipynb)|🇩🇪|Normalization of medication mentions (without surrounding text) to ATC codes|

## External Links

|Link|Language|Description|
|---|---|---|
| https://github.com/hpi-dhc/symptemist | 🇪🇸 | BioCreative VIII SympTEMIST + [LLM-based Entity Simplification](../xmen/data/simplification.py)  |

## Benchmarks

More examples for configurations can be found in the [Benchmarks](../benchmarks/benchmark) folder.

|Benchmark|Language|
|---|---|
|[Quaero](../benchmarks/benchmark/quaero.yaml)|🇫🇷|
|[MedMentions](../benchmarks/benchmark/medmentions_en.yaml)|🇬🇧|
|[DisTEMIST](../benchmarks/benchmark/distemist.yaml)|🇪🇸|
|[SympTEMIST](../benchmarks/benchmark/symptemist.yaml)|🇪🇸|
|[BRONCO](../benchmarks/benchmark/bronco.yaml)|🇩🇪|
|[Mantra](../benchmarks/benchmark/mantra.yaml)|🇬🇧 🇫🇷 🇪🇸 🇩🇪 🇳🇱|
