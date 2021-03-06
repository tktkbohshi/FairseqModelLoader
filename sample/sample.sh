#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python sample.py /project/nakamura-lab09/Work/natsuno-t/additional_ex_ANLP2022/reinforce_data_20220208/bin/ \
 --arch transformer \
 --finetune-from-model /project/nakamura-lab09/Work/natsuno-t/span_extraction/savedmodel_1110_persona.pt/checkpoint_best.pt \
 --save-dir checkpoints/savedmodel.pt \
 --source-lang src \
 --target-lang dst \
 --tokenizer space \
 --bpe sentencepiece \
 --sentencepiece-model data/dicts/sp_oall_32k.model \
 --batch-size 16 \
 --encoder-embed-dim 1920 --decoder-embed-dim 1920 \
 --encoder-attention-heads 32 --decoder-attention-heads 32 \
 --encoder-ffn-embed-dim 7680 --decoder-ffn-embed-dim 7680 \
 --encoder-layers 2 --decoder-layers 24 \
 --encoder-normalize-before --decoder-normalize-before