#!/bin/bash
#SBATCH --job-name=finetune_script
#SBATCH --output=finetune_script.out
#SBATCH --error=finetune_script.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=32G
#SBATCH --time=10-00:00:00
#SBATCH --partition=normal
#SBATCH --nodelist=g01n16

module load cuda/11.8
source activate model
cd /work/home/cryoem666/xyf/temp/pycharm/OPUS-BioLLM-CSTP/multi_modality_model/multi_modality_v1/train

deepspeed --num_nodes=1 --num_gpus=4 train_mem.py \
    --output_dir path/to/save \
    --data_path path/to/embed_dat.jsonl \
    --model_name_or_path path/to/model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --save_strategy "steps" \
    --save_steps 10000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --model_max_length 256 \
    --pretrain_protein_projector_ckpt path/to/cstp

