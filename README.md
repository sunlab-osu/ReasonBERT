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

For details on preparing the pretraining data, please check `data_processing` and the paper.

## Environment Setup
Dependencies
```
pandas==1.2.4
torch
transformers==4.3.3
webdataset==0.2.20
torch-scatter # You need to install pytorch first, then install the correct version of torch-scatter
```
## Pretraining
You can pretrain the models with our, or your own pretraining data using `pretrain.py` (for text only setting), and `hybrid_pretrain.py` (for hybrid setting with tables). The configs are under `configs/pretrain`, you need to modify the `batch_size` based on your system configurations, and point the data/output path correctly. A demo script is under `scripts/pretrain.sh`. `xla_spawn.py` can be used to launch pretraining job on Google cloud TPUs, note that you need to set `fp16=false` for TPUs.

## QA
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

## TableQA
The data for TableQA on nqtables is under `data/nqtables`. We cast TableQA still as extractive QA task, but support using table-based encoder, e.g., tapas or ReasonBERT-tapas. The associated configs are under `configs/nqtables`, just change the `datadir` and `save_dir` accordingly. We use the same architecture, to avoid repeatitive preprocessing, you could download the tokenizer and point `tokenizer` to that location. Use `bert_version` to specify the encoder you want to use.

```
# This will use the tapas model, and train with 10% of data
python train.py -c configs/nqtables/configTableQA_tapas.json --run_id 0.1_1 --seed 1 --sample 0.1 --epochs 10 --warmup_steps 1 --eval_steps 3

# This will use basic roberta model, and treat tableQA as regular text QA task
python train.py -c configs/nqtables/configTableTextQA_roberta.json --run_id 0.1_1 --seed 1 --sample 0.1 --epochs 10 --warmup_steps 1 --eval_steps 3
```

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
