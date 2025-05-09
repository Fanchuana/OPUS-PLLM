#!/bin/bash
#SBATCH --job-name=total_eval_keywords_IDFilter
#SBATCH --output=total_eval_keywords_IDFilter.out
#SBATCH --error=total_eval_keywords_IDFilter.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --partition=normal
#SBATCH --nodelist=g01n03


module load cuda11.8
source activate model


cd /path/to/multi_modality_model/multi_modality_v1/eval

accelerate launch  run_opus_ddp.py  \
--model-base-path /path/to/model/path/to/model \
--adapter-path /path/to/base \
--is_json True \
--json_path /path/to/file \
--save_path /path/to/save \
--load-4bit True