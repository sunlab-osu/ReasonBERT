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
