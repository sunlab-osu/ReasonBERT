#!/bin/bash -v
declare -A pretrain_models=( ["splinter"]="output/splinter" ["ours_roberta"]="osunlp/ReasonBERT-RoBERTa-base" )

for PRETRAIN_NAME in "${!pretrain_models[@]}"
do
    PRETRAIN_MODEL=${pretrain_models[$PRETRAIN_NAME]}
    for DATASET in "HotpotQA"
    do
        for SAMPLE in 16 128 1024
        do
            echo "test "$DATASET" "$SAMPLE
            # # roberta-base
            python train.py -c configs/MRQA/configQA.json --seed 1\
                --sample $SAMPLE\
                --dataset $DATASET\
                --pretrain $PRETRAIN_MODEL\
                --use_ours 0\
                --preprocess_only 1

            CUDA_VISIBLE_DEVICES=0 python train.py -c configs/MRQA/configQA.json --seed 11\
                --sample $SAMPLE\
                --dataset $DATASET\
                --pretrain $PRETRAIN_MODEL\
                --expr $DATASET"_"$SAMPLE"_"$PRETRAIN_NAME"_seed1_neg-1_5e-5_exp-score" &>/dev/null &
            CUDA_VISIBLE_DEVICES=1 python train.py -c configs/MRQA/configQA.json --seed 13\
                --sample $SAMPLE\
                --dataset $DATASET\
                --pretrain $PRETRAIN_MODEL\
                --expr $DATASET"_"$SAMPLE"_"$PRETRAIN_NAME"_seed3_neg-1_5e-5_exp-score" &>/dev/null &
            CUDA_VISIBLE_DEVICES=2 python train.py -c configs/MRQA/configQA.json --seed 15\
                --sample $SAMPLE\
                --dataset $DATASET\
                --pretrain $PRETRAIN_MODEL\
                --expr $DATASET"_"$SAMPLE"_"$PRETRAIN_NAME"_seed5_neg-1_5e-5_exp-score" &>/dev/null &
            CUDA_VISIBLE_DEVICES=3 python train.py -c configs/MRQA/configQA.json --seed 17\
                --sample $SAMPLE\
                --dataset $DATASET\
                --pretrain $PRETRAIN_MODEL\
                --expr $DATASET"_"$SAMPLE"_"$PRETRAIN_NAME"_seed7_neg-1_5e-5_exp-score"
            wait
        done
    done
done