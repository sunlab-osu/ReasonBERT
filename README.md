# ReasonBERT
Code and pre-trained models for [*ReasonBert: Pre-trained to Reason with Distant Supervision*](https://arxiv.org/abs/2109.04912), EMNLP'2021

## Pretrained Models
The pretrained models are shared via Huggingface ModelHub (https://huggingface.co/osunlp). You can directly load them with Huggingface Transformers.
```
from transformers import AutoTokenizer, AutoModel
  
tokenizer = AutoTokenizer.from_pretrained("osunlp/ReasonBERT-RoBERTa-base")

model = AutoModel.from_pretrained("osunlp/ReasonBERT-RoBERTa-base")
```
Note that the tokenizers are identical to BERT/RoBERTa/TAPAS, with the extra `<QUESTION>` token appended in the end. Please refer to our paper for more details.

## Pretraining Data
The pretraining data for both text-only and hybrid settings are shared on https://zenodo.org/record/5612316.

## Environment Setup
Dependencies
```
pandas==1.2.4
torch
transformers==4.3.3
webdataset==0.1.40
torch-scatter # You need to install pytorch first, then install the correct version of torch-scatter
```

## MRQA
You can download the data for MRQA experiments from the [official repo](https://github.com/mrqa/MRQA-Shared-Task-2019). Note that we use the original training data for training (and dev if needed), and the original in-domain dev data for testing. The data directory should be specificed in `configs/MRQA/configQA.json -> datadir`.

We experiment under both few-shot and full-data setting. You can use the script below to run experiments for MRQA. See README under configs for more info.
```
MODEL=$1 #roberta-base; osunlp/ReasonBERT-RoBERTa-base; ...
DATASET=$2 #SQuAD; NewsQA; ...

MODEL_NAME=(${MODEL//// })
MODEL_NAME=${MODEL_NAME[-1]}

SEED=7 
SAMPLE=128 # number of training examples to sample, -1 will use all
echo $MODEL_NAME $DATASET
python train.py\
    --config ./configs/MRQA/configQA.json\
    --pretrain ${MODEL}\
    --sample ${SAMPLE}\
    --seed ${SEED}\
    --run_id ${MODEL_NAME}_${DATASET}_${SAMPLE}_${SEED}\
    --dataset ${DATASET}\
    --overwrite
```
*Tips: The results reported in the paper are based on five random runs with seed 1/3/5/7/9. To reproduce the exact same results for our pretrained models, you need to first initialize the model with base PLM (roberta-base or bert-base), then load the weights of ReasonBERT (This is what we do when running the experiments with local pretrained checkpoints). Directly load ReasonBERT(like pass osunlp/ReasonBERT-RoBERTa-base in the script above) will have slightly different results as the random initizlization of QA headers are different.*

## Citation
```
@inproceedings{deng-etal-2021-reasonbert,
    title = "{R}eason{BERT}: {P}re-trained to Reason with Distant Supervision",
    author = "Deng, Xiang  and
      Su, Yu  and
      Lees, Alyssa  and
      Wu, You  and
      Yu, Cong  and
      Sun, Huan",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.494",
    doi = "10.18653/v1/2021.emnlp-main.494",
    pages = "6112--6127",
}
```