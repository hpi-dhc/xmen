# Training a weakly supervised cross-encoder

(Assuming target language is French)

Translate MedMentions and project labels:
`python translate_dataset.py medmentions_st21pv en fr ../data/`

Train cross-encoder model:
`python run_benchmark.py benchmark=medmentions_fr output=./models/medmentions_fr linker.reranking.training.num_train_epochs=5`

# Benchmark run with fully-supervised cross-encoder

`python run_benchmark.py benchmark=quaero output=~/scratch/xmen`

# Benchmark run with weakly-supervised cross-encoder

`python run_benchmark.py benchmark=quaero output=~/scratch/xmen +linker.reranking.pre_trained_model=absolute/path/to/models/ce_ws_medmentions_fr`
