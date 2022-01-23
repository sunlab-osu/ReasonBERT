Here is an example of the config file
```
{
    "name": "Base_QA", # Name of the experiment when saving results
    # Number of gpus to use. Should match with the env (e.g. set with CUDA_VISIBLE_DEVICES). Will be used to calculate the real batch size etc.
    "n_gpus": 1, 
    "seed": 123,

    "arch": {
        "type": "SentencePairModelForQA", # The model to use, defined in model/model.py
        "args": {
            "bert_version": "roberta-base", # Base encoder to use
            "gradient_checkpointing": false
        } # Arguments to initilize the model
    },
    # Tokenizer used for processing the data. Should match with the bert_version. Some PLMs share the same tokenizer so the processed data can be shared as well.
    "tokenizer": "roberta-base", 
    "train_dataset": {
        "type": "MRQADataset", # The dataset class to use, defined in data_loader/data_loaders.py
        "args":{
            "datadir": "./data/MRQA", # Path to downloaded data
            "split": "train", # Data split, the data (e.g. jsonl file) should be saved under ./data/MRQA/{split}/
            "sample_num": 1024, # For few-shot experiments. -1 will use all training data, >1 will use the exact sample_num of training data, 0-1 will sample this fraction of training data
            "neg_ratio": 1, # pos:neg ratios, negative samples are those that does not contain answer.
            "overwrite": false,
            "skip_first_line": true
        }
    },
    "eval_dataset": {
        "type": "MRQADataset",
        "args":{
            "datadir": "./data/MRQA",
            "split": "eval",
            "overwrite": false,
            "skip_first_line": true
        }
    },
    "collator": {
        "type": "MRQA_collate",
        "args":{
            "use_token_type": false
        }
    },
    "trainer_type": {
        "type": "BaseTrainerForQA",
        "args":{}
    },
    "trainer": {
        "num_train_epochs": 10,
        "warmup_steps": 1.0,
        "weight_decay": 0.01,
        "gradient_accumulation_steps": 2,
        "learning_rate": 5e-5,
        "max_grad_norm": 1.0,
        "logging_steps": 100,
        "save_total_limit": 2,
        "fp16": true,
        "per_device_train_batch_size": 10,
        "per_device_eval_batch_size": 64,
        "evaluation_strategy": "steps"
    }, # Hyperparameters
    "save_dir": "./output/MRQA", # Directory to save the results
    "local_pretrained": "", # If this is not empty and use_our_pretrained=1, will try to load weights from a pre-trained model saved locally
    "use_our_pretrained": 0,
    "preprocess_only": 0,
    "do_test": 0,
    "test_only": 0
}
```