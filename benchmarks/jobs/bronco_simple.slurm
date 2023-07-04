#!/bin/bash
#SBATCH --job-name=xmen
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=florian.borchert@mattermost
#SBATCH --partition=longrun
#SBATCH --time=192:00:00
#SBATCH --output=xmen_%j.out

# Initialize conda:
eval "$(conda shell.bash hook)"

conda activate xmen100

cd ~/workspace/xmen/benchmarks
python run_benchmark.py -m hydra/launcher=gpupro benchmark=bronco_medications output=/scratch/florian.borchert/ linker.reranking.training.score_regularization=0.0,0.5,1.0,2.0,4.0
