#!/bin/bash
#SBATCH --job-name=p10qsd_filing_dataset_creation
#SBATCH --output=logs/filing_%A.out
#SBATCH --error=logs/filing_%A.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd /home/sarthak.malla/Documents/P10QSD
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate p10qsd

python -m src.dataloader.filing_dataset