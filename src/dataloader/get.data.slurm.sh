#!/bin/bash
#SBATCH --job-name=p10qsd_data_collection
#SBATCH --output=logs/sec_%A_%a.out
#SBATCH --error=logs/sec_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --array=0-7%4

cd /home/sarthak.malla/Documents/P10QSD
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate p10qsd

SHARD=$(printf "ticker_shards/shard_%02d.txt" $SLURM_ARRAY_TASK_ID)

python -m src.dataloader.sec_loader data.tickers_file=$SHARD