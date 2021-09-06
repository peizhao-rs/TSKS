#!/bin/bash

export CUDA_VISIBLE_DEVICES=0


pythonpath='python'


# ${pythonpath} -u main.py --run_type test \
#                            --use_gpu True \
#                            --batch_size 16 \
#                            --vocab_size 50000 \
#                            --test_data_path ../data/duconv/demo.test \
#                            --test_sample_path ../data/duconv/sample.test.json \
#                            --test_topic_path ../data/duconv/demo.test.topic \
#                            --vocab_path ../data/duconv/vocab.txt \
#                            --init_model_path ./checkpoints/duconv/stage_1/epoch_15_iter_55256_nll_34.46973 \
#                            --output_dir ./outputs/duconv/epoch_15_iter_55256_nll_34.46973

# ${pythonpath} eval.py ./outputs/duconv/epoch_15_iter_55256_nll_34.46973/eval.txt


test_file_name=epoch_19_iter_57128_nll_86.80552
${pythonpath} -u main.py --run_type test \
                           --data_class wizard \
                           --use_gpu True \
                           --batch_size 32 \
                           --vocab_size 50000 \
                           --test_data_path ../data/wizard/test_seen.json \
                           --test_sample_path ../data/wizard/test_seen.json \
                           --vocab_path ../data/wizard/vocab.txt \
                           --init_model_path ./checkpoints/wizard/stage_1/${test_file_name} \
                           --output_dir ./outputs/wizard/test_seen/${test_file_name}

${pythonpath} eval.py ./outputs/wizard/test_seen/${test_file_name}/eval.txt

${pythonpath} -u main.py --run_type test \
                           --data_class wizard \
                           --use_gpu True \
                           --batch_size 32 \
                           --vocab_size 50000 \
                           --test_data_path ../data/wizard/test_unseen.json \
                           --test_sample_path ../data/wizard/test_unseen.json \
                           --vocab_path ../data/wizard/vocab.txt \
                           --init_model_path ./checkpoints/wizard/stage_1/${test_file_name} \
                           --output_dir ./outputs/wizard/test_unseen/${test_file_name}

${pythonpath} eval.py ./outputs/wizard/test_unseen/${test_file_name}/eval.txt