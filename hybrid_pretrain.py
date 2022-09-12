import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer.trainer import HybridTrainer

from transformers import TrainingArguments
from transformers import AutoModel
import transformers

import os
import pdb
import random

from tqdm import tqdm


def main(index=0):
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument("--run_id", default=0, type=str, help="run id")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target="trainer;learning_rate"
        ),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="train_dataset;args;batch_size"
        ),
    ]
    config = ConfigParser.from_args(args, options, index=index)

    # fix random seeds for reproducibility
    seed = config["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    transformers.logging.set_verbosity_info()

    # setup data_loader instances
    config["train_dataset"]["args"]["length"] = (
        config["train_dataset"]["args"]["length"]
        // config["train_dataset"]["args"]["batch_size"]
    )
    train_dataset = config.init_obj(
        "train_dataset",
        module_data,
        n_gpus=config["n_gpus"],
        tokenizer=config["tokenizer"],
        mlm_probability=config["mlm_probability"],
        index=index,
    )

    # build model architecture, then print to console
    model = config.init_obj(
        "arch", module_arch, vocab_size=len(train_dataset.tokenizer)
    )

    if config.resume is not None:
        last_trained = torch.load(
            os.path.join(config.resume.resolve().as_posix(), "pytorch_model.bin")
        )
        model.load_state_dict(last_trained)
        del last_trained
    elif "roberta" in config["tokenizer"]:
        # init the model with roberta
        roberta_model = AutoModel.from_pretrained("roberta-base")
        roberta_model.resize_token_embeddings(len(train_dataset.tokenizer))
        model_state_dict = model.state_dict()
        roberta_state_dict = roberta_model.state_dict()
        state_dict_to_load = {
            "bert.tapas." + k: v
            for k, v in roberta_state_dict.items()
            if "bert.tapas." + k in model_state_dict
        }
        model_state_dict.update(state_dict_to_load)
        model.load_state_dict(model_state_dict)
        del roberta_model
        del model_state_dict
        del roberta_state_dict
        del state_dict_to_load

    config["trainer"].update(
        {
            "per_device_train_batch_size": config["train_dataset"]["args"][
                "batch_size"
            ],
            "dataloader_num_workers": config["n_gpus"] * 2,
            "seed": seed,
            "output_dir": config.save_dir.resolve().as_posix(),
            "logging_dir": config.log_dir.resolve().as_posix(),
        }
    )
    training_args = TrainingArguments(**config["trainer"])

    trainer = HybridTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
    )

    if config.resume is None:
        trainer.train()
    else:
        trainer.train(config.resume.resolve().as_posix())


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main(index)


if __name__ == "__main__":
    main()
