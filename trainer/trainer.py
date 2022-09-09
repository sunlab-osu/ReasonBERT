import pdb
from typing import Any, Dict, List, Optional, Tuple, Union

import model.metric as metric
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_loader.data_loaders import MyDataLoader
from packaging import version
from torch.cuda.amp import autocast
from torch.utils.data.dataloader import DataLoader
from transformers.trainer import Trainer
from transformers.trainer_utils import PredictionOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


class BaseTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        return MyDataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=True,
            nominal_length=self.train_dataset.length,
        )

    def compute_loss(self, model, inputs):
        loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            q_loc=inputs["q_loc"],
            q_mask=inputs["q_mask"],
            target_start=inputs["target_start"],
            target_end=inputs["target_end"],
            target_mask=inputs["target_mask"],
            mlm_labels=inputs.get("mlm_target", None),
        )
        if isinstance(loss, dict):
            if self.args.n_gpu > 1:
                for k in loss:
                    if k != "loss":
                        loss[k] = loss[k].mean()
            try:
                for k in self.tmp_sub_loss:
                    if k != "loss":
                        self.tmp_sub_loss[k] += loss[k].detach()
                self.tmp_sub_loss_count += 1
            except:
                self.tmp_sub_loss = {
                    k: torch.tensor(0.0).to(self.args.device)
                    for k in loss
                    if k != "loss"
                }
                self.tmp_sub_loss_count = 0
            loss = loss["loss"]
        return loss

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch):
        if self.control.should_log:
            logs: Dict[str, float] = {}
            tr_loss_scalar = tr_loss.item()
            # reset tr_loss to zero
            tr_loss -= tr_loss

            for k, v in self.tmp_sub_loss.items():
                logs[k] = v.item() / self.tmp_sub_loss_count
            self.tmp_sub_loss = {k: v - v for k, v in self.tmp_sub_loss.items()}
            self.tmp_sub_loss_count = 0
            logs["loss"] = round(
                tr_loss_scalar
                / (self.state.global_step - self._globalstep_last_logged),
                4,
            )
            # backward compatibility for pytorch schedulers
            logs["learning_rate"] = (
                self.lr_scheduler.get_last_lr()[0]
                if version.parse(torch.__version__) >= version.parse("1.4")
                else self.lr_scheduler.get_lr()[0]
            )

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate()
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(
                self.args, self.state, self.control
            )


class HybridTrainer(BaseTrainer):
    def compute_loss(self, model, inputs):
        # pdb.set_trace()
        loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            q_loc=inputs["q_loc"],
            q_mask=inputs["q_mask"],
            q_cell_mask=inputs["q_cell_mask"],
            target_start=inputs["target_start"],
            target_end=inputs["target_end"],
            target_mask=inputs["target_mask"],
            target_row=inputs["target_row"],
            target_column=inputs["target_column"],
            mlm_labels=inputs.get("mlm_target", None),
        )
        if isinstance(loss, dict):
            if self.args.n_gpu > 1:
                for k in loss:
                    if k != "loss":
                        loss[k] = loss[k].mean()
            try:
                for k in self.tmp_sub_loss:
                    if k != "loss":
                        self.tmp_sub_loss[k] += loss[k].detach()
                self.tmp_sub_loss_count += 1
            except:
                self.tmp_sub_loss = {
                    k: torch.tensor(0.0).to(self.args.device)
                    for k in loss
                    if k != "loss"
                }
                self.tmp_sub_loss_count = 0
            loss = loss["loss"]
        return loss


class BaseTrainerForQA(Trainer):
    def get_test_dataloader(self, test_dataset):
        return self.get_eval_dataloader(test_dataset)

    def compute_loss(self, model, inputs):
        loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            q_loc=inputs["q_loc"],
            target_start_loc=inputs["target_start"],
            target_end_loc=inputs["target_end"],
            target_mask=inputs["target_mask"],
        )
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    loss, start_pred_logits, end_pred_logits = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"],
                        q_loc=inputs["q_loc"],
                        target_start_loc=inputs["target_start"],
                        target_end_loc=inputs["target_end"],
                        target_mask=inputs["target_mask"],
                        is_train=False,
                    )
                    start_pred_logits = start_pred_logits - start_pred_logits[:, :, 0:1]
                    start_p = start_pred_logits.exp()
                    end_pred_logits = end_pred_logits - end_pred_logits[:, :, 0:1]
                    end_p = end_pred_logits.exp()
                    start_p = start_p.unsqueeze(3)
                    end_p = end_p.unsqueeze(2)
                    pred_p = start_p + end_p
                    if "start_end_mask" in inputs:
                        pred_p = pred_p * inputs["start_end_mask"].unsqueeze(1)
                    pred_p[:, :, 0, :] = 0
                    pred_p = torch.triu(pred_p) - torch.triu(pred_p, diagonal=30)
                    max_p, pred_loc = torch.max(
                        pred_p.reshape(pred_p.shape[0], -1), dim=-1
                    )
                    pred_start = torch.floor(pred_loc.float() / pred_p.shape[-1]).int()
                    pred_end = torch.remainder(pred_loc, pred_p.shape[-1]).int()
            else:
                loss, start_pred_logits, end_pred_logits = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                    q_loc=inputs["q_loc"],
                    target_start_loc=inputs["target_start"],
                    target_end_loc=inputs["target_end"],
                    target_mask=inputs["target_mask"],
                    is_train=False,
                )
                start_pred_logits = start_pred_logits - start_pred_logits[:, :, 0:1]
                start_p = start_pred_logits.exp()
                end_pred_logits = end_pred_logits - end_pred_logits[:, :, 0:1]
                end_p = end_pred_logits.exp()
                start_p = start_p.unsqueeze(3)
                end_p = end_p.unsqueeze(2)
                pred_p = start_p + end_p
                if "start_end_mask" in inputs:
                    pred_p = pred_p * inputs["start_end_mask"].unsqueeze(1)
                pred_p[:, :, 0, :] = 0
                pred_p = torch.triu(pred_p) - torch.triu(pred_p, diagonal=30)
                max_p, pred_loc = torch.max(pred_p.reshape(pred_p.shape[0], -1), dim=-1)
                pred_start = torch.floor(pred_loc.float() / pred_p.shape[-1]).int()
                pred_end = torch.remainder(pred_loc, pred_p.shape[-1]).int()
        if self.args.n_gpu > 1:
            loss = loss.mean()
        return loss, (pred_start, pred_end, max_p)

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        model = self.model.to(self.args.device)
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        q_results = {}
        f1 = []
        exact_match = []
        total_loss = []

        for step, inputs in enumerate(dataloader):
            loss, preds = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            for i, pred in enumerate(zip(*preds)):
                answers = inputs["answers"][i]
                qid = inputs["qid"][i]
                if qid not in q_results:
                    q_results[qid] = {"answers": answers, "preds": {}}
                pred_answer = dataloader.dataset.tokenizer.decode(
                    inputs["input_ids"][i][pred[0].item() : pred[1].item() + 1]
                )
                if (
                    pred_answer not in q_results[qid]["preds"]
                    or q_results[qid]["preds"][pred_answer] < pred[2].item()
                ):
                    q_results[qid]["preds"][pred_answer] = pred[2].item()
            total_loss.append(loss.item())
            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )
        total_loss = np.mean(total_loss)
        predictions = {}
        for qid, result in q_results.items():
            pred = sorted(
                list(result["preds"].items()), reverse=True, key=lambda z: z[1]
            )[0]
            pred_answer = pred[0] + " [DOC] [PAR] [TLE] [SEP]"
            special_tok_loc = min(
                [
                    pred_answer.find("[DOC]"),
                    pred_answer.find("[PAR]"),
                    pred_answer.find("[TLE]"),
                    pred_answer.find("[SEP]"),
                ]
            )
            pred_answer = pred_answer[:special_tok_loc].strip()
            answers = result["answers"]
            result["pred_answer"] = pred_answer
            predictions[qid] = [pred_answer, pred[1]]
            f1.append(
                metric.metric_max_over_ground_truths(
                    metric.f1_score, pred_answer, answers
                )
            )
            exact_match.append(
                metric.metric_max_over_ground_truths(
                    metric.exact_match_score, pred_answer, answers
                )
            )
        f1 = 100.0 * np.mean(f1)
        exact_match = 100.0 * np.mean(exact_match)
        metrics = {"f1": f1, "exact_match": exact_match, "loss": total_loss}
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return PredictionOutput(
            predictions=predictions, metrics=metrics, label_ids=None
        )


class BaseTrainerForPS(Trainer):
    def get_test_dataloader(self, test_dataset):
        return self.get_eval_dataloader(test_dataset)

    def compute_loss(self, model, inputs):
        loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            q_loc=inputs["q_loc"],
            target=inputs["target"],
        )
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    loss, pred_logits = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"],
                        q_loc=inputs["q_loc"],
                        target=inputs["target"],
                        is_train=False,
                    )
            else:
                loss, pred_logits = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                    q_loc=inputs["q_loc"],
                    target=inputs["target"],
                    is_train=False,
                )
        if self.args.n_gpu > 1:
            loss = loss.mean()
        return loss, (pred_logits)

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        model = self.model.to(self.args.device)
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        q_results = {}
        top2_fact_recall = []
        top2_answer_recall = []
        top3_fact_recall = []
        top3_answer_recall = []
        top5_fact_recall = []
        top5_answer_recall = []
        total_loss = []

        for step, inputs in enumerate(dataloader):
            loss, preds = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            for i, pred in enumerate(preds.reshape(-1)):
                meta = inputs["meta"][i]
                qid = meta["qid"]
                if qid not in q_results:
                    q_results[qid] = {
                        "all_facts": meta["all_facts"],
                        "preds": [],
                        "metas": {},
                    }
                q_results[qid]["preds"].append([meta["cid"], pred.item()])
                q_results[qid]["metas"][meta["cid"]] = {
                    "is_fact": meta["is_fact"],
                    "is_answer": meta["answer_in_context"],
                    "fact_in_context": meta["fact_in_context"],
                }
            total_loss.append(loss.item())
            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )
        total_loss = np.mean(total_loss)
        predictions = {}
        for qid, result in q_results.items():
            preds = sorted(list(result["preds"]), reverse=True, key=lambda z: z[1])
            all_facts = {tuple(fact) for fact in result["all_facts"]}
            facts_found = set()
            predictions[qid] = preds[:10]
            found_answer = False
            for i, (cid, _) in enumerate(preds[:5]):
                meta = result["metas"][cid]
                if meta["is_answer"]:
                    found_answer = True
                facts_found |= {tuple(fact) for fact in meta["fact_in_context"]}
                if i == 1:
                    top2_fact_recall.append(
                        len(facts_found & all_facts) / len(all_facts)
                    )
                    top2_answer_recall.append(1 if found_answer else 0)
                elif i == 2:
                    top3_fact_recall.append(
                        len(facts_found & all_facts) / len(all_facts)
                    )
                    top3_answer_recall.append(1 if found_answer else 0)
                elif i == 4:
                    top5_fact_recall.append(
                        len(facts_found & all_facts) / len(all_facts)
                    )
                    top5_answer_recall.append(1 if found_answer else 0)
        top2_fact_recall = 100.0 * np.mean(top2_fact_recall)
        top2_answer_recall = 100.0 * np.mean(top2_answer_recall)
        top3_fact_recall = 100.0 * np.mean(top3_fact_recall)
        top3_answer_recall = 100.0 * np.mean(top3_answer_recall)
        top5_fact_recall = 100.0 * np.mean(top5_fact_recall)
        top5_answer_recall = 100.0 * np.mean(top5_answer_recall)
        metrics = {
            "top2_fact_recall": top2_fact_recall,
            "top2_answer_recall": top2_answer_recall,
            "top3_fact_recall": top3_fact_recall,
            "top3_answer_recall": top3_answer_recall,
            "top5_fact_recall": top5_fact_recall,
            "top5_answer_recall": top5_answer_recall,
            "loss": total_loss,
        }
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return PredictionOutput(
            predictions=predictions, metrics=metrics, label_ids=None
        )


class BaseTrainerForCS(Trainer):
    def get_test_dataloader(self, test_dataset):
        return self.get_eval_dataloader(test_dataset)

    def compute_loss(self, model, inputs):
        loss = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            q_loc=inputs["q_loc"],
            target_start_loc=inputs["target_start"],
            target_mask=inputs["target_mask"],
            target_start_onehot=inputs["target_start_onehot"]
            if "target_start_onehot" in inputs
            else None,
            target_row=inputs["target_row_onehot"],
            target_column=inputs["target_column_onehot"],
        )
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            if self.use_amp:
                with autocast():
                    result = model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        token_type_ids=inputs["token_type_ids"],
                        q_loc=inputs["q_loc"],
                        target_start_loc=inputs["target_start"],
                        target_mask=inputs["target_mask"],
                        target_start_onehot=inputs["target_start_onehot"]
                        if "target_start_onehot" in inputs
                        else None,
                        target_row=inputs["target_row_onehot"],
                        target_column=inputs["target_column_onehot"],
                        is_train=False,
                    )
            else:
                result = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                    q_loc=inputs["q_loc"],
                    target_start_loc=inputs["target_start"],
                    target_mask=inputs["target_mask"],
                    target_start_onehot=inputs["target_start_onehot"]
                    if "target_start_onehot" in inputs
                    else None,
                    target_row=inputs["target_row_onehot"],
                    target_column=inputs["target_column_onehot"],
                    is_train=False,
                )
            if len(result) == 2:
                loss, start_pred_logits = result
                start_pred_logits -= start_pred_logits[:, :, 0:1]
                start_p = start_pred_logits.exp()
                start_p[:, :, 0] = 0
                max_p, pred_loc = torch.max(
                    start_p.reshape(start_p.shape[0], -1), dim=-1
                )
            else:
                loss, row_pred_logits, column_pred_logits = result
                row_pred_logits -= row_pred_logits[:, :, 0:1]
                row_p = row_pred_logits.exp()
                row_p[:, :, 0] = 0
                max_row_p, pred_row_loc = torch.max(
                    row_p.reshape(row_p.shape[0], -1), dim=-1
                )
                column_pred_logits -= column_pred_logits[:, :, 0:1]
                column_p = column_pred_logits.exp()
                column_p[:, :, 0] = 0
                max_column_p, pred_column_loc = torch.max(
                    column_p.reshape(column_p.shape[0], -1), dim=-1
                )
                max_p = max_row_p * max_column_p
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if len(result) == 2:
            return loss, (pred_loc, max_p)
        else:
            return loss, (pred_row_loc, pred_column_loc - 1, max_p)

    def prediction_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> PredictionOutput:
        model = self.model.to(self.args.device)
        # multi-gpu eval
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        batch_size = dataloader.batch_size
        num_examples = self.num_examples(dataloader)
        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", num_examples)
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        self.callback_handler.eval_dataloader = dataloader

        q_results = {}
        exact_match = []
        top2_acc = []
        top3_acc = []
        top5_acc = []
        total_loss = []

        for step, inputs in enumerate(dataloader):
            loss, preds = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            for i, pred in enumerate(zip(*preds)):
                answers = [tuple(answer) for answer in inputs["answers"][i]]
                qid = inputs["qid"][i]
                if qid not in q_results:
                    q_results[qid] = {"answers": answers, "preds": {}}
                if len(pred) == 2:
                    cell_id = inputs["cell_ids"][i, pred[0]].item()
                    cell_id = (int(cell_id / 256), cell_id % 256)
                else:
                    cell_id = (pred[0].item(), pred[1].item())
                if (
                    cell_id not in q_results[qid]["preds"]
                    or q_results[qid]["preds"][cell_id] < pred[-1].item()
                ):
                    q_results[qid]["preds"][cell_id] = pred[-1].item()
            total_loss.append(loss.item())
            self.control = self.callback_handler.on_prediction_step(
                self.args, self.state, self.control
            )
        total_loss = np.mean(total_loss)
        predictions = {}
        for qid, result in q_results.items():
            preds = sorted(
                list(result["preds"].items()), reverse=True, key=lambda z: z[1]
            )
            pred_cells = [pred[0] for pred in preds]
            answers = result["answers"]
            predictions[qid] = preds[:10]
            exact_match.append(1 if pred_cells[0] in answers else 0)
            top2_acc.append(
                1 if any([pred_cell in answers for pred_cell in pred_cells[:2]]) else 0
            )
            top3_acc.append(
                1 if any([pred_cell in answers for pred_cell in pred_cells[:3]]) else 0
            )
            top5_acc.append(
                1 if any([pred_cell in answers for pred_cell in pred_cells[:5]]) else 0
            )
        exact_match = 100.0 * np.mean(exact_match)
        top2_acc = 100.0 * np.mean(top2_acc)
        top3_acc = 100.0 * np.mean(top3_acc)
        top5_acc = 100.0 * np.mean(top5_acc)
        metrics = {
            "exact_match": exact_match,
            "top2_acc": top2_acc,
            "top3_acc": top3_acc,
            "top5_acc": top5_acc,
            "loss": total_loss,
        }
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        return PredictionOutput(
            predictions=predictions, metrics=metrics, label_ids=None
        )
