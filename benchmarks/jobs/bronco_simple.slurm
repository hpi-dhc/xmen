#!/bin/bash
#SBATCH --job-name=xmen
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=florian.borchert@mattermost
#SBATCH --partition=longrun
#SBATCH --time=48:00:00
#SBATCH --mem=8gb
#SBATCH --output=xmen_%j.out

# Initialize conda:
eval "$(conda shell.bash hook)"

conda activate xmen100

cd ~/workspace/xmen/benchmarks
python run_benchmark.py -m hydra/launcher=gpupro benchmark=bronco_medications output=/scratch/florian.borchert/ linker.reranking.training.rank_regularization=0.0,1.0 linker.reranking.data.context_length=0,128,256
