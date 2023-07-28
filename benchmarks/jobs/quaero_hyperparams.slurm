#!/bin/bash
#SBATCH --job-name=xmen
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=florian.borchert@mattermost
#SBATCH --partition=longrun
#SBATCH --time=8-0:00:00
#SBATCH --mem=8gb
#SBATCH --output=xmen_%j.out

# Initialize conda:
eval "$(conda shell.bash hook)"

conda activate xmen100

cd ~/workspace/xmen/benchmarks
python run_benchmark.py -m hydra/launcher=gpupro benchmark=quaero output=~/scratch/xmen/ wandb_project=xmen-hyperparams linker.reranking.training.learning_rate=5e-6,1e-5,2e-5,5e-5,1e-4

#linker.reranking.training.rank_regularization=0.0,0.2,0.5,1.0,2.0 linker.reranking.data.context_length=0,32,64,128,256 
