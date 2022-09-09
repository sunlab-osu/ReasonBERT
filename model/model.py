import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torch_scatter import scatter
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    PreTrainedModel,
    TapasModel,
)
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform


def check_memory():
    print("GPU memory: %.1f" % (torch.cuda.memory_allocated() // 1024**2))


def cross_entropy_with_onehot(logits, labels, mask=None, softmax=True):
    if mask is not None:
        labels = labels * mask.unsqueeze(1)
    if softmax:
        return -(F.log_softmax(logits, dim=-1) * labels).sum() / labels.sum()
    else:
        return -(logits * labels).sum() / labels.sum()


class SentencePairModel(BaseModel):
    def __init__(
        self,
        bert_version="bert-base-uncased",
        vocab_size=None,
        gradient_checkpointing=False,
        use_all_target=False,
        use_transform=False,
    ):
        super().__init__()
        if ".json" in bert_version:
            config = AutoConfig.from_pretrained(bert_version)
            self.bert = AutoModelForMaskedLM.from_config(config)
        else:
            self.bert = AutoModelForMaskedLM.from_pretrained(
                bert_version, gradient_checkpointing=gradient_checkpointing
            )
        if vocab_size is not None:
            self.bert.resize_token_embeddings(vocab_size)
        self.hidden_size = self.bert.config.hidden_size
        self.W_start = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_end = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.use_transform = use_transform
        if self.use_transform:
            self.start_transform = BertPredictionHeadTransform(self.bert.config)
            self.end_transform = BertPredictionHeadTransform(self.bert.config)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.use_all_target = use_all_target
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        q_loc,
        q_mask,
        target_start,
        target_end,
        target_mask,
        is_train=True,
        mlm_labels=None,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=mlm_labels,
            return_dict=True,
            output_hidden_states=True,
        )
        bert_encoding = self.dropout(bert_output.hidden_states[-1])
        if self.use_transform:
            start_encoding = self.start_transform(bert_encoding)
            q_encoding = torch.gather(
                start_encoding, 1, q_loc.unsqueeze(2).expand(-1, -1, self.hidden_size)
            )
            target_mask = ((1.0 - target_mask) * -10000.0).unsqueeze(1)
            start_pred_logits = torch.matmul(
                self.W_start(q_encoding), start_encoding.transpose(1, 2)
            )
            start_pred_logits = start_pred_logits + target_mask
            end_encoding = self.end_transform(bert_encoding)
            q_encoding = torch.gather(
                end_encoding, 1, q_loc.unsqueeze(2).expand(-1, -1, self.hidden_size)
            )
            end_pred_logits = torch.matmul(
                self.W_end(q_encoding), end_encoding.transpose(1, 2)
            )
            end_pred_logits = end_pred_logits + target_mask
        else:
            q_encoding = torch.gather(
                bert_encoding, 1, q_loc.unsqueeze(2).expand(-1, -1, self.hidden_size)
            )
            target_mask = ((1.0 - target_mask) * -10000.0).unsqueeze(1)
            start_pred_logits = torch.matmul(
                self.W_start(q_encoding), bert_encoding.transpose(1, 2)
            )
            start_pred_logits = start_pred_logits + target_mask
            end_pred_logits = torch.matmul(
                self.W_end(q_encoding), bert_encoding.transpose(1, 2)
            )
            end_pred_logits = end_pred_logits + target_mask
        start_loss = 0
        end_loss = 0
        if self.use_all_target:
            start_loss = cross_entropy_with_onehot(
                start_pred_logits.reshape(-1, start_pred_logits.shape[-1]),
                target_start.reshape(-1, target_start.shape[-1]),
                mask=q_mask.reshape(-1),
            )
            end_loss = cross_entropy_with_onehot(
                end_pred_logits.reshape(-1, end_pred_logits.shape[-1]),
                target_end.reshape(-1, target_end.shape[-1]),
                mask=q_mask.reshape(-1),
            )
        else:
            target_start_loc = target_start.argmax(dim=-1) * q_mask - (1 - q_mask)
            start_loss = self.loss(
                start_pred_logits.reshape(-1, start_pred_logits.shape[-1]),
                target_start_loc.reshape(-1),
            )
            target_end_loc = target_end.argmax(dim=-1) * q_mask - (1 - q_mask)
            end_loss = self.loss(
                end_pred_logits.reshape(-1, end_pred_logits.shape[-1]),
                target_end_loc.reshape(-1),
            )
        loss = start_loss + end_loss
        if mlm_labels is not None:
            loss += bert_output.loss
            mlm_loss = bert_output.loss
        else:
            mlm_loss = 0 * start_loss
        if is_train:
            return {
                "loss": loss,
                "start_loss": start_loss,
                "end_loss": end_loss,
                "mlm_loss": mlm_loss,
            }
        else:
            return (loss, start_pred_logits, end_pred_logits)


class HybridPairModel(BaseModel):
    def __init__(
        self,
        bert_version="bert-base-uncased",
        vocab_size=None,
        gradient_checkpointing=False,
        use_all_target=False,
        use_column_row=False,
        agg="mean",
    ):
        super().__init__()
        if ".json" in bert_version:
            config = AutoConfig.from_pretrained(bert_version)
            self.bert = AutoModelForMaskedLM.from_config(config)
        else:
            self.bert = AutoModelForMaskedLM.from_pretrained(
                bert_version,
                gradient_checkpointing=gradient_checkpointing,
                revision="no_reset",
            )
        if vocab_size is not None:
            self.bert.resize_token_embeddings(vocab_size)
        self.hidden_size = self.bert.config.hidden_size
        self.W_start = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_end = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_row = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_column = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.use_all_target = use_all_target
        self.use_column_row = use_column_row
        self.agg = agg
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        q_loc,
        q_mask,
        q_cell_mask,
        target_start,
        target_end,
        target_mask,
        target_row,
        target_column,
        is_train=True,
        mlm_labels=None,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=mlm_labels,
            return_dict=True,
            output_hidden_states=True,
        )
        bert_encoding = self.dropout(bert_output.hidden_states[-1])
        q_encoding = torch.gather(
            bert_encoding, 1, q_loc.unsqueeze(2).expand(-1, -1, self.hidden_size)
        )
        target_mask = ((1.0 - target_mask) * -10000.0).unsqueeze(1)
        start_pred_logits = torch.matmul(
            self.W_start(q_encoding), bert_encoding.transpose(1, 2)
        )
        start_pred_logits = start_pred_logits + target_mask
        end_pred_logits = torch.matmul(
            self.W_end(q_encoding), bert_encoding.transpose(1, 2)
        )
        end_pred_logits = end_pred_logits + target_mask
        start_loss = 0
        end_loss = 0
        if self.use_all_target:
            start_loss = cross_entropy_with_onehot(
                start_pred_logits.reshape(-1, start_pred_logits.shape[-1]),
                target_start.reshape(-1, target_start.shape[-1]),
                mask=q_mask.reshape(-1),
            )
            end_loss = cross_entropy_with_onehot(
                end_pred_logits.reshape(-1, end_pred_logits.shape[-1]),
                target_end.reshape(-1, target_end.shape[-1]),
                mask=q_mask.reshape(-1),
            )
        else:
            target_start_loc = target_start.argmax(dim=-1) * q_mask - (1 - q_mask)
            start_loss = self.loss(
                start_pred_logits.reshape(-1, start_pred_logits.shape[-1]),
                target_start_loc.reshape(-1),
            )
            target_end_loc = target_end.argmax(dim=-1) * q_mask - (1 - q_mask)
            end_loss = self.loss(
                end_pred_logits.reshape(-1, end_pred_logits.shape[-1]),
                target_end_loc.reshape(-1),
            )
        loss = start_loss + end_loss

        # calculate cell selection loss
        if self.use_column_row:
            # column_ids = token_type_ids[:,:,1].unsqueeze(1)
            # row_ids = token_type_ids[:,:,2].unsqueeze(1)
            # target_cell_mask = (column_ids<1)*-10000.0
            # row_pred_logits = torch.matmul(self.W_row(q_encoding), bert_encoding.transpose(1,2))
            # row_pred_logits = row_pred_logits + target_cell_mask
            # row_pred_agg_logits = scatter(row_pred_logits, row_ids, dim=2, reduce='max', out=torch.full((row_pred_logits.shape[0],row_pred_logits.shape[1],256), -10000.0).to(row_pred_logits.device))
            # column_pred_logits = torch.matmul(self.W_column(q_encoding), bert_encoding.transpose(1,2))
            # column_pred_logits = column_pred_logits + target_cell_mask
            # column_pred_agg_logits = scatter(column_pred_logits, column_ids, dim=2, reduce='max', out=torch.full((column_pred_logits.shape[0],column_pred_logits.shape[1],256), -10000.0).to(column_pred_logits.device))
            column_ids = token_type_ids[:, :, 1].unsqueeze(1)
            row_ids = token_type_ids[:, :, 2].unsqueeze(1)
            target_cell_mask = (column_ids > 0).float()
            cell_ids = row_ids * 256 + column_ids
            cell_mask = scatter(
                target_cell_mask, cell_ids, dim=2, reduce="max", dim_size=256 * 256
            )
            # row_cell_count = scatter(target_cell_mask, row_ids, dim=2, reduce='sum', dim_size=256)
            row_cell_count = cell_mask.reshape(-1, 1, 256, 256).sum(dim=3)
            row_mask = (row_cell_count < 1) * -10000.0
            # column_cell_count = scatter(target_cell_mask, column_ids, dim=2, reduce='sum', dim_size=256)
            column_cell_count = cell_mask.reshape(-1, 1, 256, 256).sum(dim=2)
            column_mask = (column_cell_count < 1) * -10000.0

            row_pred_logits = torch.matmul(
                self.W_row(q_encoding), bert_encoding.transpose(1, 2)
            )
            # row_pred_logits = row_pred_logits + target_cell_mask
            # row_pred_logits = F.log_softmax(row_pred_logits, dim=-1)
            # row_pred_agg_logits = scatter(row_pred_logits, row_ids, dim=2, reduce='sum', out=torch.full((row_pred_logits.shape[0],row_pred_logits.shape[1],256), 0.0).to(row_pred_logits.device))
            row_pred_cell_agg_logits = scatter(
                row_pred_logits, cell_ids, dim=2, reduce="mean", dim_size=256 * 256
            )
            if self.agg == "mean":
                row_pred_agg_logits = row_pred_cell_agg_logits.reshape(
                    -1, row_pred_cell_agg_logits.shape[1], 256, 256
                ).sum(dim=3) / row_cell_count.clamp(min=1).reshape(-1, 1, 256)
            else:
                row_pred_agg_logits = row_pred_cell_agg_logits.reshape(
                    -1, row_pred_cell_agg_logits.shape[1], 256, 256
                ).max(dim=3)[0]
            row_pred_agg_logits += row_mask
            row_pred_agg_logits[:, :, 0] = row_pred_logits[:, :, 0]

            column_pred_logits = torch.matmul(
                self.W_column(q_encoding), bert_encoding.transpose(1, 2)
            )
            # column_pred_logits = column_pred_logits + target_cell_mask
            # column_pred_logits = F.log_softmax(column_pred_logits, dim=-1)
            # column_pred_agg_logits = scatter(column_pred_logits, column_ids, dim=2, reduce='sum', out=torch.full((column_pred_logits.shape[0],column_pred_logits.shape[1],256), 0.0).to(column_pred_logits.device))
            column_pred_cell_agg_logits = scatter(
                column_pred_logits, cell_ids, dim=2, reduce="mean", dim_size=256 * 256
            )
            if self.agg == "mean":
                column_pred_agg_logits = column_pred_cell_agg_logits.reshape(
                    -1, column_pred_cell_agg_logits.shape[1], 256, 256
                ).sum(dim=2) / column_cell_count.clamp(min=1).reshape(-1, 1, 256)
            else:
                column_pred_agg_logits = column_pred_cell_agg_logits.reshape(
                    -1, column_pred_cell_agg_logits.shape[1], 256, 256
                ).max(dim=2)[0]
            column_pred_agg_logits += column_mask
            column_pred_agg_logits[:, :, 0] = column_pred_logits[:, :, 0]
            if self.use_all_target:
                row_loss = cross_entropy_with_onehot(
                    row_pred_agg_logits.reshape(-1, 256),
                    target_row.reshape(-1, 256),
                    mask=q_cell_mask.reshape(-1),
                )
                column_loss = cross_entropy_with_onehot(
                    column_pred_agg_logits.reshape(-1, 256),
                    target_column.reshape(-1, 256),
                    mask=q_cell_mask.reshape(-1),
                )
            else:
                target_row_loc = target_row.argmax(dim=-1) * q_cell_mask - (
                    1 - q_cell_mask
                )
                row_loss = self.loss(
                    row_pred_agg_logits.reshape(-1, 256), target_row_loc.reshape(-1)
                )
                target_column_loc = target_column.argmax(dim=-1) * q_cell_mask - (
                    1 - q_cell_mask
                )
                column_loss = self.loss(
                    column_pred_agg_logits.reshape(-1, 256),
                    target_column_loc.reshape(-1),
                )
            loss += row_loss + column_loss
        else:
            row_loss = 0 * loss
            column_loss = 0 * loss
        if mlm_labels is not None:
            loss += bert_output.loss
            mlm_loss = bert_output.loss
        else:
            mlm_loss = 0 * loss
        if is_train:
            return {
                "loss": loss,
                "start_loss": start_loss,
                "end_loss": end_loss,
                "mlm_loss": mlm_loss,
                "row_loss": row_loss,
                "column_loss": column_loss,
            }
        else:
            return (loss, start_pred_logits, end_pred_logits)


class SentencePairModelForQA(BaseModel):
    def __init__(
        self,
        bert_version="bert-base-uncased",
        vocab_size=None,
        gradient_checkpointing=False,
        use_transform=False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            bert_version, gradient_checkpointing=gradient_checkpointing
        )
        if vocab_size is not None:
            self.bert.resize_token_embeddings(vocab_size)
        self.hidden_size = self.bert.config.hidden_size
        self.use_transform = use_transform
        self.W_start = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_end = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        if self.use_transform:
            self.start_transform = BertPredictionHeadTransform(self.bert.config)
            self.end_transform = BertPredictionHeadTransform(self.bert.config)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        q_loc,
        target_start_loc,
        target_end_loc,
        target_mask,
        is_train=True,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        bert_encoding = self.dropout(bert_output.last_hidden_state)
        if self.use_transform:
            start_encoding = self.dropout(self.start_transform(bert_encoding))
            q_encoding = torch.gather(
                start_encoding, 1, q_loc.unsqueeze(2).expand(-1, -1, self.hidden_size)
            )
            target_mask = ((1.0 - target_mask) * -10000.0).unsqueeze(1)
            start_pred_logits = torch.matmul(
                self.W_start(q_encoding), start_encoding.transpose(1, 2)
            )
            start_pred_logits = start_pred_logits + target_mask
            end_encoding = self.dropout(self.end_transform(bert_encoding))
            q_encoding = torch.gather(
                end_encoding, 1, q_loc.unsqueeze(2).expand(-1, -1, self.hidden_size)
            )
            end_pred_logits = torch.matmul(
                self.W_end(q_encoding), end_encoding.transpose(1, 2)
            )
            end_pred_logits = end_pred_logits + target_mask
        else:
            q_encoding = torch.gather(
                bert_encoding, 1, q_loc.unsqueeze(2).expand(-1, -1, self.hidden_size)
            )
            target_mask = ((1.0 - target_mask) * -10000.0).unsqueeze(1)
            start_pred_logits = torch.matmul(
                self.W_start(q_encoding), bert_encoding.transpose(1, 2)
            )
            start_pred_logits = start_pred_logits + target_mask
            end_pred_logits = torch.matmul(
                self.W_end(q_encoding), bert_encoding.transpose(1, 2)
            )
            end_pred_logits = end_pred_logits + target_mask
        start_loss = 0
        end_loss = 0
        start_loss = self.loss(
            start_pred_logits.reshape(-1, start_pred_logits.shape[-1]),
            target_start_loc.reshape(-1),
        )
        end_loss = self.loss(
            end_pred_logits.reshape(-1, end_pred_logits.shape[-1]),
            target_end_loc.reshape(-1),
        )
        loss = start_loss + end_loss
        if is_train:
            return loss
        else:
            return (loss, start_pred_logits, end_pred_logits)


class SentencePairModelForPS(BaseModel):
    def __init__(
        self,
        bert_version="bert-base-uncased",
        vocab_size=None,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            bert_version, gradient_checkpointing=gradient_checkpointing
        )
        if vocab_size is not None:
            self.bert.resize_token_embeddings(vocab_size)
        self.hidden_size = self.bert.config.hidden_size
        self.W_start = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.loss = nn.BCEWithLogitsLoss(
            weight=None,
            size_average=None,
            reduce=None,
            reduction="mean",
            pos_weight=None,
        )
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(
        self, input_ids, attention_mask, token_type_ids, q_loc, target, is_train=True
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        bert_encoding = self.dropout(bert_output.last_hidden_state)
        q_encoding = torch.gather(
            bert_encoding, 1, q_loc.unsqueeze(2).expand(-1, -1, self.hidden_size)
        )

        start_pred_logits = torch.matmul(
            self.W_start(q_encoding), bert_encoding.transpose(1, 2)
        )
        logits = start_pred_logits[:, :, 0]
        loss = self.loss(logits.reshape(-1), target.reshape(-1))
        if is_train:
            return loss
        else:
            return (loss, logits)


class TableModelForQA(BaseModel):
    def __init__(self, bert_version="google/tapas-base", vocab_size=None):
        super().__init__()
        if ".json" in bert_version:
            config = AutoConfig.from_pretrained(bert_version)
            self.bert = AutoModel.from_config(config)
        else:
            self.bert = AutoModel.from_pretrained(bert_version, revision="no_reset")
        if vocab_size is not None:
            self.bert.resize_token_embeddings(vocab_size)
        self.hidden_size = self.bert.config.hidden_size
        self.W_start = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.W_end = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        q_loc,
        target_start_loc,
        target_end_loc,
        target_mask,
        is_train=True,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        bert_encoding = self.dropout(bert_output.last_hidden_state)
        q_encoding = torch.gather(
            bert_encoding, 1, q_loc.unsqueeze(2).expand(-1, -1, self.hidden_size)
        )
        target_mask = ((1.0 - target_mask) * -10000.0).unsqueeze(1)

        start_pred_logits = torch.matmul(
            self.W_start(q_encoding), bert_encoding.transpose(1, 2)
        )
        start_pred_logits = start_pred_logits + target_mask
        end_pred_logits = torch.matmul(
            self.W_end(q_encoding), bert_encoding.transpose(1, 2)
        )
        end_pred_logits = end_pred_logits + target_mask
        start_loss = 0
        end_loss = 0
        start_loss = self.loss(
            start_pred_logits.reshape(-1, start_pred_logits.shape[-1]),
            target_start_loc.reshape(-1),
        )
        end_loss = self.loss(
            end_pred_logits.reshape(-1, end_pred_logits.shape[-1]),
            target_end_loc.reshape(-1),
        )
        loss = start_loss + end_loss
        if is_train:
            return loss
        else:
            return (loss, start_pred_logits, end_pred_logits)


class TableModelForCS(BaseModel):
    def __init__(
        self, bert_version="google/tapas-base", vocab_size=None, use_column_row=0
    ):
        super().__init__()
        if ".json" in bert_version:
            config = AutoConfig.from_pretrained(bert_version)
            self.bert = AutoModel.from_config(config)
        else:
            self.bert = AutoModel.from_pretrained(bert_version, revision="no_reset")
        if vocab_size is not None:
            self.bert.resize_token_embeddings(vocab_size)
        self.use_column_row = use_column_row
        self.hidden_size = self.bert.config.hidden_size
        if use_column_row:
            self.W_row = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
            self.W_column = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        else:
            self.W_start = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        q_loc,
        target_start_loc,
        target_mask,
        target_start_onehot=None,
        target_row=None,
        target_column=None,
        is_train=True,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        bert_encoding = self.dropout(bert_output.last_hidden_state)
        q_encoding = torch.gather(
            bert_encoding, 1, q_loc.unsqueeze(2).expand(-1, -1, self.hidden_size)
        )
        target_start_onehot *= target_mask
        target_mask = target_mask.unsqueeze(1)
        if self.use_column_row == 1:
            column_ids = token_type_ids[:, :, 1].unsqueeze(1)
            row_ids = token_type_ids[:, :, 2].unsqueeze(1)
            target_cell_mask = (column_ids > 0).float()
            cell_ids = row_ids * 256 + column_ids
            cell_mask = scatter(
                target_cell_mask, cell_ids, dim=2, reduce="max", dim_size=256 * 256
            )
            # row_cell_count = scatter(target_cell_mask, row_ids, dim=2, reduce='sum', dim_size=256)
            row_cell_count = cell_mask.reshape(-1, 1, 256, 256).sum(dim=3)
            row_mask = (row_cell_count < 1) * -10000.0
            # column_cell_count = scatter(target_cell_mask, column_ids, dim=2, reduce='sum', dim_size=256)
            column_cell_count = cell_mask.reshape(-1, 1, 256, 256).sum(dim=2)
            column_mask = (column_cell_count < 1) * -10000.0

            row_pred_logits = torch.matmul(
                self.W_row(q_encoding), bert_encoding.transpose(1, 2)
            )
            # row_pred_logits = row_pred_logits + target_cell_mask
            # row_pred_logits = F.log_softmax(row_pred_logits, dim=-1)
            # row_pred_agg_logits = scatter(row_pred_logits, row_ids, dim=2, reduce='sum', out=torch.full((row_pred_logits.shape[0],row_pred_logits.shape[1],256), 0.0).to(row_pred_logits.device))
            row_pred_cell_agg_logits = scatter(
                row_pred_logits, cell_ids, dim=2, reduce="mean", dim_size=256 * 256
            )
            row_pred_agg_logits = row_pred_cell_agg_logits.reshape(-1, 1, 256, 256).sum(
                dim=3
            ) / row_cell_count.clamp(min=1).reshape(-1, 1, 256)
            # row_pred_agg_logits = row_pred_cell_agg_logits.reshape(-1,1,256,256).max(dim=3)[0]
            row_pred_agg_logits += row_mask
            row_pred_agg_logits[:, :, 0] = row_pred_logits[:, :, 0]

            column_pred_logits = torch.matmul(
                self.W_column(q_encoding), bert_encoding.transpose(1, 2)
            )
            # column_pred_logits = column_pred_logits + target_cell_mask
            # column_pred_logits = F.log_softmax(column_pred_logits, dim=-1)
            # column_pred_agg_logits = scatter(column_pred_logits, column_ids, dim=2, reduce='sum', out=torch.full((column_pred_logits.shape[0],column_pred_logits.shape[1],256), 0.0).to(column_pred_logits.device))
            column_pred_cell_agg_logits = scatter(
                column_pred_logits, cell_ids, dim=2, reduce="mean", dim_size=256 * 256
            )
            column_pred_agg_logits = column_pred_cell_agg_logits.reshape(
                -1, 1, 256, 256
            ).sum(dim=2) / column_cell_count.clamp(min=1).reshape(-1, 1, 256)
            # column_pred_agg_logits = column_pred_cell_agg_logits.reshape(-1,1,256,256).max(dim=2)[0]
            column_pred_agg_logits += column_mask
            column_pred_agg_logits[:, :, 0] = column_pred_logits[:, :, 0]
            # row_loss = cross_entropy_with_onehot(row_pred_agg_logits.reshape(-1,256), target_row.reshape(-1,256), softmax=False)
            # column_loss = cross_entropy_with_onehot(column_pred_agg_logits.reshape(-1,256), target_column.reshape(-1,256), softmax=False)
            row_loss = cross_entropy_with_onehot(
                row_pred_agg_logits.reshape(-1, 256), target_row.reshape(-1, 256)
            )
            column_loss = cross_entropy_with_onehot(
                column_pred_agg_logits.reshape(-1, 256), target_column.reshape(-1, 256)
            )
            loss = (row_loss + column_loss) / 2.0
            if is_train:
                return loss
            else:
                return (loss, row_pred_agg_logits, column_pred_agg_logits)
        else:
            target_mask = (1.0 - target_mask) * -10000.0
            start_pred_logits = torch.matmul(
                self.W_start(q_encoding), bert_encoding.transpose(1, 2)
            )
            start_pred_logits = start_pred_logits + target_mask
            start_loss = 0
            if target_start_onehot is None:
                start_loss = self.loss(
                    start_pred_logits.reshape(-1, start_pred_logits.shape[-1]),
                    target_start_loc.reshape(-1),
                )
            else:
                start_loss = (
                    -(
                        F.log_softmax(start_pred_logits.squeeze(), dim=-1)
                        * target_start_onehot
                    ).sum()
                    / target_start_onehot.sum()
                )
            loss = start_loss
            if is_train:
                return loss
            else:
                return (loss, start_pred_logits)


class SimpleQA(BaseModel):
    def __init__(
        self,
        bert_version="bert-base-uncased",
        vocab_size=None,
        gradient_checkpointing=False,
        use_transform=False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(
            bert_version, gradient_checkpointing=gradient_checkpointing
        )
        if vocab_size is not None:
            self.bert.resize_token_embeddings(vocab_size)
        self.hidden_size = self.bert.config.hidden_size
        self.qa_outputs = nn.Linear(self.hidden_size, 2)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        q_loc,
        target_start_loc,
        target_end_loc,
        target_mask,
        is_train=True,
    ):
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        bert_encoding = self.dropout(bert_output.last_hidden_state)
        logits = self.qa_outputs(bert_encoding)
        target_mask = (1.0 - target_mask) * -10000.0
        start_pred_logits, end_pred_logits = logits.split(1, dim=-1)
        start_pred_logits = start_pred_logits.squeeze(-1) + target_mask
        end_pred_logits = end_pred_logits.squeeze(-1) + target_mask
        start_loss = 0
        end_loss = 0
        start_loss = self.loss(
            start_pred_logits.reshape(-1, start_pred_logits.shape[-1]),
            target_start_loc.reshape(-1),
        )
        end_loss = self.loss(
            end_pred_logits.reshape(-1, end_pred_logits.shape[-1]),
            target_end_loc.reshape(-1),
        )
        loss = start_loss + end_loss
        if is_train:
            return loss
        else:
            return (loss, start_pred_logits.unsqueeze(1), end_pred_logits.unsqueeze(1))
