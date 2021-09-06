#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

pythonpath='python'

${pythonpath} -u main.py --run_type train \
                            --data_class wizard \
                            --stage 0 \
                            --num_epoch 6 \
                            --lr 0.0005 \
                            --batch_size 32 \
                            --vocab_size 50000 \
                            --train_data_path ../data/wizard/train.json \
                            --dev_data_path ../data/wizard/dev.json \
                            --vocab_path ../data/wizard/vocab.txt \
                            --checkpoint_dir ./checkpoints/wizard/stage_0 \
                            --log_steps 100 \
                            --valid_steps 800

${pythonpath} -u main.py --run_type train \
                           --data_class wizard \
                           --stage 1 \
                           --num_epoch 20 \
                           --lr 0.0005 \
                           --batch_size 24 \
                           --vocab_size 50000 \
                           --train_data_path ../data/wizard/train.json \
                           --dev_data_path ../data/wizard/dev.json \
                           --vocab_path ../data/wizard/vocab.txt \
                           --checkpoint_dir ./checkpoints/wizard/stage_1 \
                           --log_steps 100 \
                           --valid_steps 800 \
                           --init_model_path ./checkpoints/wizard/stage_0/model_stage_0

# ${pythonpath} -u main.py --run_type train \
#                            --stage 0 \
#                            --data_class duconv \
#                            --num_epoch 5 \
#                            --lr 0.0005 \
#                            --batch_size 32 \
#                            --vocab_size 50000 \
#                            --train_data_path ../data/duconv/demo.train \
#                            --dev_data_path ../data/duconv/demo.dev \
#                            --vocab_path ../data/duconv/vocab.txt \
#                            --checkpoint_dir ./checkpoints/duconv/stage_0 \
#                            --log_steps 100 \
#                            --valid_steps 800
                           

#  ${pythonpath} -u main.py --run_type train \
#                             --stage 1 \
#                             --data_class duconv \
#                             --num_epoch 16 \
#                             --lr 0.0005 \
#                             --batch_size 24 \
#                             --vocab_size 50000 \
#                             --train_data_path ../data/duconv/demo.train \
#                             --dev_data_path ../data/duconv/demo.dev \
#                             --vocab_path ../data/duconv/vocab.txt \
#                             --checkpoint_dir ./checkpoints/duconv/stage_1 \
#                             --log_steps 100 \
#                             --valid_steps 800 \
#                             --init_model_path ./checkpoints/duconv/stage_0/model_stage_0