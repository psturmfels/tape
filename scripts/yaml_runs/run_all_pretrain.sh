#!/bin/bash
cd /export/home/tape/
python -u run_distributed.py \
    transformer joint_mlm_profile \
    --data_dir /export/home/tape/data/alignment/ \
    --gradient_accumulation_steps 1 \
    --batch_size 512 \
    --learning_rate 2.5e-4 \
    --nproc_per_node 8 \
    --fp16 \
    --WANDB_API_KEY d715f4d161ee84314d5c2a84f8bc595d7781d2ab \
    --WANDB_USERNAME psturmfels \
    --WANDB_PROJECT protein_sequences \
    --tokenizer iupac \
    --model_config_file /export/home/tape/config/transformer_config.json \
    --max_sequence_length 270 \
    --num_workers 0 \

python -u run_distributed.py \
    transformer profile_prediction \
    --data_dir /export/home/tape/data/alignment/ \
    --gradient_accumulation_steps 1 \
    --batch_size 512 \
    --learning_rate 5e-4 \
    --nproc_per_node 8 \
    --fp16 \
    --WANDB_API_KEY d715f4d161ee84314d5c2a84f8bc595d7781d2ab \
    --WANDB_USERNAME psturmfels \
    --WANDB_PROJECT protein_sequences \
    --tokenizer iupac \
    --model_config_file /export/home/tape/config/transformer_config.json \
    --max_sequence_length 270 \
    --num_workers 0 \

python -u run_distributed.py \
    transformer masked_language_modeling \
    --data_dir /export/home/tape/data/alignment/ \
    --gradient_accumulation_steps 1 \
    --batch_size 512 \
    --learning_rate 2.5e-4 \
    --nproc_per_node 8 \
    --fp16 \
    --WANDB_API_KEY d715f4d161ee84314d5c2a84f8bc595d7781d2ab \
    --WANDB_USERNAME psturmfels \
    --WANDB_PROJECT protein_sequences \
    --tokenizer iupac \
    --model_config_file /export/home/tape/config/transformer_config.json \
    --max_sequence_length 270 \
    --num_workers 0 \
