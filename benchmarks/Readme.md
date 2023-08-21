# Weakly Supervised Cross-Encoder

Assuming target language is French

- `python translate_dataset.py medmentions_st21pv en fr ../data/`
- `python run_benchmark.py benchmark=medmentions_fr output=../models/medmentions_fr linker.reranking.training.num_train_epochs=5`