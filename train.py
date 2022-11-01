import argparse
import collections
import json
import os
import pdb
import random

import numpy as np
import torch
import transformers
from transformers import AutoModel, TrainingArguments
from transformers.utils import logging

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import trainer.trainer as module_trainer
from parse_config import ConfigParser
from trainer.trainer import BaseTrainer, BaseTrainerForQA

logger = logging.get_logger(__name__)


def main(config, args):

    # fix random seeds for reproducibility
    seed = config["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    transformers.logging.set_verbosity_info()

    # setup data_loader instances
    if any(
        [name in config["arch"]["args"]["bert_version"].lower() for name in ["roberta"]]
    ):
        tokenizer = "roberta-base"
    elif any(
        [name in config["arch"]["args"]["bert_version"].lower() for name in ["bert"]]
    ):
        tokenizer = "bert-base-cased"
    elif "splinter" in config["arch"]["args"]["bert_version"].lower():
        tokenizer = config["arch"]["args"]["bert_version"]
    else:
        try:
            tokenizer = config["tokenizer"]
        except KeyError:
            tokenizer = config["arch"]["args"]["bert_version"]

    if config["train_dataset"]["args"].get("dataset", None) is None:
        train_dataset = config.init_obj(
            "train_dataset", module_data, tokenizer=tokenizer, dataset=config["dataset"]
        )
    else:
        train_dataset = config.init_obj(
            "train_dataset", module_data, tokenizer=tokenizer
        )
    # To save time, we only sample 4096 examples for evaluation during training
    if config["eval_dataset"]["args"].get("dataset", None) is None:
        eval_dataset = config.init_obj(
            "eval_dataset",
            module_data,
            tokenizer=tokenizer,
            dataset=config["dataset"],
            sample_num=4096,
        )
    else:
        eval_dataset = config.init_obj(
            "eval_dataset", module_data, tokenizer=tokenizer, sample_num=4096
        )
    if config["do_test"] != 0:
        try:
            if config["test_dataset"]["args"].get("dataset", None) is None:
                test_dataset = config.init_obj(
                    "test_dataset",
                    module_data,
                    tokenizer=tokenizer,
                    dataset=config["dataset"],
                )
            else:
                test_dataset = config.init_obj(
                    "test_dataset", module_data, tokenizer=tokenizer
                )
        except:
            pass

    if config["preprocess_only"] == 1:
        exit()

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    if config["local_pretrained"] and config["use_our_pretrained"] == 1:
        logger.info(f"load pretrained model from {config['local_pretrained']}")
        pretrained_model = torch.load(
            os.path.join(config["local_pretrained"], "pytorch_model.bin")
        )
        state_dict = model.state_dict()
        is_mlm_model = any(
            ["lm_head" in k or "cls.predictions" in k for k in pretrained_model.keys()]
        )
        model.bert.resize_token_embeddings(
            model.bert.config.vocab_size + 1
        )  # handle extra special token [QUESTION]
        not_used = []
        for k, v in pretrained_model.items():
            if not k.startswith("bert"):
                k = "bert." + k
            if args.use_all or k.startswith("bert"):
                if is_mlm_model:
                    if "roberta" in k:
                        k = k.replace("bert.roberta", "bert")
                    elif "tapas" in k:
                        k = k.replace("bert.tapas", "bert")
                    else:
                        k = k.replace("bert.bert", "bert")
                if k in state_dict:
                    state_dict[k] = v
                else:
                    not_used.append(k)
            else:
                not_used.append(k)
        print("unused parameters", not_used)
        model.load_state_dict(state_dict)
        del pretrained_model
        del state_dict
    model.bert.resize_token_embeddings(
        len(train_dataset.tokenizer)
    )  # handle extra special tokens like [DOC]

    if config.resume is not None:
        last_trained = torch.load(
            os.path.join(config.resume.resolve().as_posix(), "pytorch_model.bin")
        )
        model.load_state_dict(last_trained)
        del last_trained

    if (
        config["train_dataset"]["args"]["sample_num"] < 1
        and config["train_dataset"]["args"]["sample_num"] != -1
    ):
        config["train_dataset"]["args"]["sample_num"] = len(train_dataset)
    real_batch_size = (
        config["trainer"]["per_device_train_batch_size"]
        * config["trainer"]["gradient_accumulation_steps"]
        * config["n_gpus"]
    )
    step_per_epoch = int(len(train_dataset) / real_batch_size)
    # Set the training steps, mostly for fewshot settings
    if (
        config["train_dataset"]["args"]["sample_num"] <= 2048
        and config["train_dataset"]["args"]["sample_num"] != -1
    ):
        config["trainer"].update(
            {
                "dataloader_num_workers": config["n_gpus"] * 2,
                "seed": seed,
                "output_dir": config.save_dir.resolve().as_posix(),
                "logging_dir": config.log_dir.resolve().as_posix(),
                "max_steps": -1 if step_per_epoch >= 20 else 200,
                "warmup_steps": 20 if step_per_epoch < 20 else step_per_epoch,
                "eval_steps": 50 if step_per_epoch < 20 else 3 * step_per_epoch,
                "save_steps": 100 if step_per_epoch < 20 else 3 * step_per_epoch,
            }
        )
    else:
        config["trainer"].update(
            {
                "dataloader_num_workers": config["n_gpus"] * 2,
                "seed": seed,
                "output_dir": config.save_dir.resolve().as_posix(),
                "logging_dir": config.log_dir.resolve().as_posix(),
                "max_steps": -1,
                "warmup_steps": int(config["trainer"]["warmup_steps"] * step_per_epoch),
                "eval_steps": int(config["trainer"]["eval_steps"] * step_per_epoch),
                "save_steps": step_per_epoch,
            }
        )
    config["trainer"].update(
        {
            "load_best_model_at_end": False if config["do_test"] == 0 else True,
        }
    )
    training_args = TrainingArguments(**config["trainer"])
    extra_args = {"tokenizer": train_dataset.tokenizer}
    if "splinter" in config["arch"]["args"]["bert_version"].lower():
        extra_args["q_id"] = 104
    collator = config.init_obj("collator", module_data, **extra_args)
    trainer = config.init_obj(
        "trainer_type",
        module_trainer,
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=train_dataset,  # training dataset
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    if config["test_only"] != 1:
        if config.resume is None:
            trainer.train()
        else:
            trainer.train(config.resume.resolve().as_posix())
        trainer._save_checkpoint(trainer.model, None)
    if config["eval_dataset"]["args"].get("dataset", None) is None:
        full_eval_dataset = config.init_obj(
            "eval_dataset", module_data, tokenizer=tokenizer, dataset=config["dataset"]
        )
    else:
        full_eval_dataset = config.init_obj(
            "eval_dataset", module_data, tokenizer=tokenizer
        )
    result_f = open(
        os.path.join(config.save_dir.resolve().as_posix(), "results.json"), "w"
    )
    eval_output = trainer.predict(full_eval_dataset)
    logger.info(f"----Eval Results-----\n{eval_output.metrics}")
    result_f.write(f"----Eval Results-----\n{eval_output.metrics}\n")
    with open(
        os.path.join(config.save_dir.resolve().as_posix(), "eval_predictions.json"), "w"
    ) as f:
        json.dump(eval_output.predictions, f)
    if config["do_test"] != 0:
        test_output = trainer.predict(test_dataset)
        logger.info(f"-----Test Results-----\n{test_output.metrics}")
        result_f.write(f"-----Test Results-----\n{test_output.metrics}\n")
        with open(
            os.path.join(config.save_dir.resolve().as_posix(), "test_predictions.json"),
            "w",
        ) as f:
            json.dump(test_output.predictions, f)
    try:
        if config["test_on_train"] != 0:
            train_output = trainer.predict(train_dataset)
            logger.info(f"-----Train Results-----\n{train_output.metrics}")
            with open(
                os.path.join(
                    config.save_dir.resolve().as_posix(), "train_predictions.json"
                ),
                "w",
            ) as f:
                json.dump(train_output.predictions, f)
    except:
        pass
    result_f.close()


if __name__ == "__main__":
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
    args.add_argument(
        "--use_all", action="store_true", help="use all pretrained headers"
    )
    args.add_argument("--overwrite", action="store_true")

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(
            ["--lr", "--learning_rate"], type=float, target="trainer;learning_rate"
        ),
        CustomArgs(["--epochs"], type=int, target="trainer;num_train_epochs"),
        CustomArgs(["--eval_steps"], type=float, target="trainer;eval_steps"),
        CustomArgs(["--warmup_steps"], type=float, target="trainer;warmup_steps"),
        CustomArgs(
            ["--bs", "--batch_size"],
            type=int,
            target="trainer;per_device_train_batch_size",
        ),
        CustomArgs(["--acc"], type=int, target="trainer;gradient_accumulation_steps"),
        CustomArgs(["--seed", "--random_seed"], type=int, target="seed"),
        CustomArgs(
            ["--sample", "--train_sample_num"],
            type=float,
            target="train_dataset;args;sample_num",
        ),
        CustomArgs(["--neg_ratio"], type=float, target="train_dataset;args;neg_ratio"),
        CustomArgs(["--use_ours"], type=int, target="use_our_pretrained"),
        CustomArgs(["--preprocess_only"], type=int, target="preprocess_only"),
        CustomArgs(["--do_test"], type=int, target="do_test"),
        CustomArgs(["--test_only"], type=int, target="test_only"),
        CustomArgs(["--expr", "--expr_name"], type=str, target="name"),
        CustomArgs(["--dataset"], type=str, target="dataset"),
        CustomArgs(
            ["--pretrain", "--pretrain_model_type"],
            type=str,
            target="arch;args;bert_version",
        ),
        CustomArgs(
            ["--arch"],
            type=str,
            target="arch;type",
        ),
        CustomArgs(["--local_pretrained"], type=str, target="local_pretrained"),
    ]
    config = ConfigParser.from_args(args, options)
    args, _ = args.parse_known_args()
    main(config, args)
