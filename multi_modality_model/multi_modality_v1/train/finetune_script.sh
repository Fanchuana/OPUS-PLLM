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
    --data_path path/to/embed_data.jsonl \
    --cache_dir path/to/cache \
    --deepspeed path/to/zero3 \
    --training_module "lora" \
    --lora_r 64 --lora_alpha 16 --lora_dropout 0.05 \
    --switch_projector_lr 1e-5 \
    --model_name_or_path path/to/model \
    --num_train_epochs 2 \
    --per_device_train_batch_size 6 \
    --tune_switch_projector False \
    --pretrain_switch_projector_ckpt path/to/switch_projector \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 4e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.006 \
    --lr_scheduler_type "cosine" \
    --model_max_length 256 \
    --pretrain_protein_projector_ckpt path/to/cstp

