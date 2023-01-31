#!/bin/bash
for SPLIT in 3 4
do
    sbatch --job-name=annotate_sentence_part_${SPLIT} --export=SPLIT=${SPLIT} /users/PAS1197/osu8727/workspace/hybrid_pretrain_preprocess/annotate_sentence.job
done