#!/bin/bash -v
CUDA_VISIBLE_DEVICES="0,1,4,5" python pretrain.py -c configs/pretrain/config_pretrain_text.json --run_id text_roberta_large --overwrite