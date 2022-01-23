from transformers import TapasTokenizer
from transformers import logging
from typing import Callable, Dict, Generator, List, Optional, Text, Tuple, Union
from transformers.tokenization_utils_base import (
    ENCODE_KWARGS_DOCSTRING,
    BatchEncoding,
    EncodedInput,
    PreTokenizedInput,
    TextInput,
)
from dataclasses import dataclass
# from transformers.file_utils import PaddingStrategy, TensorType, add_end_docstrings, is_pandas_available
from transformers.tokenization_utils import PreTrainedTokenizer, _is_control, _is_punctuation, _is_whitespace
logger = logging.get_logger(__name__)

from transformers.models.tapas.tokenization_tapas import *

class TableTokenizer(TapasTokenizer):
    def _pad(
        self,
        encoded_inputs: Union[Dict[str, EncodedInput], BatchEncoding],
        max_length: Optional[int] = None,
        padding_strategy: PaddingStrategy = PaddingStrategy.DO_NOT_PAD,
        pad_to_multiple_of: Optional[int] = None,
        return_attention_mask: Optional[bool] = None,
    ) -> dict:
        """
        Pad encoded inputs (on left/right and up to predefined length or max length in the batch)

        Args:
            encoded_inputs: Dictionary of tokenized inputs (`List[int]`) or batch of tokenized inputs (`List[List[int]]`).
            max_length: maximum length of the returned list and optionally padding length (see below).
                Will truncate by taking into account the special tokens.
            padding_strategy: PaddingStrategy to use for padding.

                - PaddingStrategy.LONGEST Pad to the longest sequence in the batch
                - PaddingStrategy.MAX_LENGTH: Pad to the max length (default)
                - PaddingStrategy.DO_NOT_PAD: Do not pad
                The tokenizer padding sides are defined in self.padding_side:

                    - 'left': pads on the left of the sequences
                    - 'right': pads on the right of the sequences
            pad_to_multiple_of: (optional) Integer if set will pad the sequence to a multiple of the provided value.
                This is especially useful to enable the use of Tensor Core on NVIDIA hardware with compute capability
                >= 7.5 (Volta).
            return_attention_mask: (optional) Set to False to avoid returning attention mask (default: set to model specifics)
        """
        # Load from model defaults
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if padding_strategy == PaddingStrategy.LONGEST:
            max_length = len(encoded_inputs["input_ids"])

        if max_length is not None and pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
            max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

        needs_to_be_padded = (
            padding_strategy != PaddingStrategy.DO_NOT_PAD and len(encoded_inputs["input_ids"]) != max_length
        )

        if needs_to_be_padded:
            difference = max_length - len(encoded_inputs["input_ids"])
            if self.padding_side == "right":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = (
                        encoded_inputs["token_type_ids"] + [[self.pad_token_type_id] * 7] * difference
                    )
                if "labels" in encoded_inputs:
                    encoded_inputs["labels"] = encoded_inputs["labels"] + [0] * difference
                if "target_mask" in encoded_inputs:
                    encoded_inputs["target_mask"] = encoded_inputs["target_mask"] + [0] * difference
                if "numeric_values" in encoded_inputs:
                    encoded_inputs["numeric_values"] = encoded_inputs["numeric_values"] + [float("nan")] * difference
                if "numeric_values_scale" in encoded_inputs:
                    encoded_inputs["numeric_values_scale"] = (
                        encoded_inputs["numeric_values_scale"] + [1.0] * difference
                    )
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"] + [1] * difference
                encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            elif self.padding_side == "left":
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [0] * difference + [1] * len(encoded_inputs["input_ids"])
                if "token_type_ids" in encoded_inputs:
                    encoded_inputs["token_type_ids"] = [[self.pad_token_type_id] * 7] * difference + encoded_inputs[
                        "token_type_ids"
                    ]
                if "labels" in encoded_inputs:
                    encoded_inputs["labels"] = [0] * difference + encoded_inputs["labels"]
                if "numeric_values" in encoded_inputs:
                    encoded_inputs["numeric_values"] = [float("nan")] * difference + encoded_inputs["numeric_values"]
                if "numeric_values_scale" in encoded_inputs:
                    encoded_inputs["numeric_values_scale"] = [1.0] * difference + encoded_inputs[
                        "numeric_values_scale"
                    ]
                if "special_tokens_mask" in encoded_inputs:
                    encoded_inputs["special_tokens_mask"] = [1] * difference + encoded_inputs["special_tokens_mask"]
                encoded_inputs["input_ids"] = [self.pad_token_id] * difference + encoded_inputs["input_ids"]
            else:
                raise ValueError("Invalid padding strategy:" + str(self.padding_side))
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        return encoded_inputs
    def _find_tokens(self, text, segment):
        """Return start index of segment in text or None."""
#         logger.info("text: %s %s", text, segment)
        for index in range(1 + len(text) - len(segment)):
            match=True
            for seg_index, seg_token in enumerate(segment):
                if text[index + seg_index] != seg_token:
                    match=False
                    break
            if match:
                return index
        return None

    def _find_answer_ids_from_answer_texts(
        self,
        column_ids,
        row_ids,
        table_ids,
        table_offset,
        answer_texts,
    ):
        """Maps question with answer texts to the first matching token indexes."""
        answer_ids = [0] * len(column_ids)
        for answer_text in answer_texts:
            if answer_text is None:
                answer_ids[0]  = 1
            else:
                index = self._find_tokens(table_ids, answer_text)
                # Maps answer coordinates to indexes this can fail if tokens / rows have
                # been pruned.
                if index is not None:
                    index += table_offset
                    for index in range(index, index+len(answer_text)):
                        answer_ids[index] = 1
                else:
                    answer_ids[0] = 1
        return answer_ids
    
    def _get_all_answer_ids_from_coordinates(
        self,
        column_ids,
        row_ids,
        answers_list,
    ):
        """Maps lists of answer coordinates to token indexes."""
        answer_ids = [0] * len(column_ids)
        for answers in answers_list:
            row_index, column_index = answers
            if column_index == -1:
                answer_ids[0] = 1
            else:
                for index in range(len(column_ids)):
                    if column_ids[index] - 1 == column_id and row_ids[index] == row_id:
                        answer_ids[index] = 1

        return answer_ids

    def get_answer_ids(self, column_ids, row_ids, table_ids, table_offset, answer_texts_question, answer_coordinates_question):
        if self.update_answer_coordinates:
            return self._find_answer_ids_from_answer_texts(
                column_ids,
                row_ids,
                table_ids,
                table_offset,
                answer_texts=[self.convert_tokens_to_ids(self.tokenize(at)) if at!='' else None for at in answer_texts_question],
            )
        return self._get_all_answer_ids_from_coordinates(column_ids, row_ids, answer_coordinates_question)
    
    def _encode_plus(
        self,
        table: "pd.DataFrame",
        query: Union[
            TextInput,
            PreTokenizedInput,
            EncodedInput,
        ],
        answer_coordinates: Optional[List[Tuple]] = None,
        answer_text: Optional[List[TextInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        **kwargs
    ):
        if query is None:
            query = ""
            logger.warning(
                "TAPAS is a question answering model but you have not passed a query. Please be aware that the "
                "model will probably not behave correctly."
            )

        table_tokens = self._tokenize_table(table)
        query_tokens = self.tokenize(query)

        return self.prepare_for_model(
            table,
            query,
            tokenized_table=table_tokens,
            query_tokens=query_tokens,
            answer_coordinates=answer_coordinates,
            answer_text=answer_text,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding=padding,
            max_length=max_length,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors=return_tensors,
            prepend_batch_axis=False,
            return_attention_mask=return_attention_mask,
            return_token_type_ids=return_token_type_ids,
            return_special_tokens_mask=return_special_tokens_mask,
            return_length=return_length,
            verbose=verbose,
        )
    
    def prepare_for_model(
        self,
        raw_table: "pd.DataFrame",
        raw_query: Union[
            TextInput,
            PreTokenizedInput,
            EncodedInput,
        ],
        tokenized_table: Optional[TokenizedTable] = None,
        query_tokens: Optional[TokenizedTable] = None,
        answer_coordinates: Optional[List[Tuple]] = None,
        answer_text: Optional[List[TextInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TapasTruncationStrategy] = False,
        max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_token_type_ids: Optional[bool] = True,
        return_attention_mask: Optional[bool] = True,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        prepend_batch_axis: bool = False,
        **kwargs
    ) -> BatchEncoding:
        """
        Prepares a sequence of input id so that it can be used by the model. It adds special tokens, truncates
        sequences if overflowing while taking into account the special tokens.

        Args:
            raw_table (:obj:`pd.DataFrame`):
                The original table before any transformation (like tokenization) was applied to it.
            raw_query (:obj:`TextInput` or :obj:`PreTokenizedInput` or :obj:`EncodedInput`):
                The original query before any transformation (like tokenization) was applied to it.
            tokenized_table (:obj:`TokenizedTable`):
                The table after tokenization.
            query_tokens (:obj:`List[str]`):
                The query after tokenization.
            answer_coordinates (:obj:`List[Tuple]` or :obj:`List[List[Tuple]]`, `optional`):
                Answer coordinates of each table-question pair in the batch. The answer_coordinates must be a single
                list of one or more tuples. Each tuple must be a (row_index, column_index) pair. The first data row
                (not the column header row) has index 0. The first column has index 0.
            answer_text (:obj:`List[str]` or :obj:`List[List[str]]`, `optional`):
                Answer text of each table-question pair in the batch. The answer_text must be a single list of one or
                more strings. Each string must be the answer text of a corresponding answer coordinate.
        """
        if isinstance(padding, bool):
            if padding and (max_length is not None or pad_to_multiple_of is not None):
                padding = PaddingStrategy.MAX_LENGTH
            else:
                padding = PaddingStrategy.DO_NOT_PAD
        elif not isinstance(padding, PaddingStrategy):
            padding = PaddingStrategy(padding)

        if isinstance(truncation, bool):
            if truncation:
                truncation = TapasTruncationStrategy.DROP_ROWS_TO_FIT
            else:
                truncation = TapasTruncationStrategy.DO_NOT_TRUNCATE
        elif not isinstance(truncation, TapasTruncationStrategy):
            truncation = TapasTruncationStrategy(truncation)

        encoded_inputs = {}

        is_part_of_batch = False
        prev_answer_coordinates, prev_answer_text = None, None
        if "prev_answer_coordinates" in kwargs and "prev_answer_text" in kwargs:
            is_part_of_batch = True
            prev_answer_coordinates = kwargs["prev_answer_coordinates"]
            prev_answer_text = kwargs["prev_answer_text"]

        num_rows = self._get_num_rows(raw_table, truncation != TapasTruncationStrategy.DO_NOT_TRUNCATE)
        num_columns = self._get_num_columns(raw_table)
        _, _, num_tokens = self._get_table_boundaries(tokenized_table)

        if truncation != TapasTruncationStrategy.DO_NOT_TRUNCATE:
            num_rows, num_tokens = self._get_truncated_table_rows(
                query_tokens, tokenized_table, num_rows, num_columns, max_length, truncation_strategy=truncation
            )
        table_data = list(self._get_table_values(tokenized_table, num_columns, num_rows, num_tokens))

        query_ids = self.convert_tokens_to_ids(query_tokens)
        table_ids = list(zip(*table_data))[0] if len(table_data) > 0 else list(zip(*table_data))
        table_ids = self.convert_tokens_to_ids(list(table_ids))

        if "return_overflowing_tokens" in kwargs and kwargs["return_overflowing_tokens"]:
            raise ValueError("TAPAS does not return overflowing tokens as it works on tables.")
        
        if add_special_tokens:
            input_ids = self.build_inputs_with_special_tokens(query_ids, table_ids)
            table_offset = len(query_ids)+2
        else:
            input_ids = query_ids + table_ids
            table_offset = len(query_ids)
        
        target_mask = [1] + [0] * (len(query_ids) + 1) + [1] * len(table_ids)

        if max_length is not None and len(input_ids) > max_length:
            raise ValueError(
                "Could not encode the query and table header given the maximum length. Encoding the query and table"
                f"header results in a length of {len(input_ids)} which is higher than the max_length of {max_length}"
            )

        encoded_inputs["input_ids"] = input_ids
        encoded_inputs["target_mask"] = target_mask
        encoded_inputs["q_loc"] = [i for i,token_id in enumerate(input_ids) if token_id==self._convert_token_to_id_with_added_voc('[QUESTION]')]
        

        segment_ids = self.create_segment_token_type_ids_from_sequences(query_ids, table_data)
        column_ids = self.create_column_token_type_ids_from_sequences(query_ids, table_data)
        row_ids = self.create_row_token_type_ids_from_sequences(query_ids, table_data)
        if not is_part_of_batch or (prev_answer_coordinates is None and prev_answer_text is None):
            # simply set the prev_labels to zeros
            prev_labels = [0] * len(row_ids)
        else:
            prev_labels = self.get_answer_ids(
                column_ids, row_ids, table_ids, table_offset, prev_answer_text, prev_answer_coordinates
            )

        # FIRST: parse both the table and question in terms of numeric values

        raw_table = add_numeric_table_values(raw_table)
        raw_query = add_numeric_values_to_question(raw_query)

        # SECOND: add numeric-related features (and not parse them in these functions):

        column_ranks, inv_column_ranks = self._get_numeric_column_ranks(column_ids, row_ids, raw_table)
        numeric_relations = self._get_numeric_relations(raw_query, column_ids, row_ids, raw_table)

        # Load from model defaults
        if return_token_type_ids is None:
            return_token_type_ids = "token_type_ids" in self.model_input_names
        if return_attention_mask is None:
            return_attention_mask = "attention_mask" in self.model_input_names

        if return_attention_mask:
            attention_mask = self.create_attention_mask_from_sequences(query_ids, table_data)
            encoded_inputs["attention_mask"] = attention_mask

        if answer_coordinates is not None and answer_text is not None:
            labels = self.get_answer_ids(column_ids, row_ids, table_ids, table_offset, answer_text, answer_coordinates)
            numeric_values = self._get_numeric_values(raw_table, column_ids, row_ids)
            numeric_values_scale = self._get_numeric_values_scale(raw_table, column_ids, row_ids)

            encoded_inputs["labels"] = labels
            encoded_inputs["numeric_values"] = numeric_values
            encoded_inputs["numeric_values_scale"] = numeric_values_scale

        if return_token_type_ids:
            token_type_ids = [
                segment_ids,
                column_ids,
                row_ids,
                prev_labels,
                column_ranks,
                inv_column_ranks,
                numeric_relations,
            ]

            token_type_ids = [list(ids) for ids in list(zip(*token_type_ids))]
            encoded_inputs["token_type_ids"] = token_type_ids

        if return_special_tokens_mask:
            if add_special_tokens:
                encoded_inputs["special_tokens_mask"] = self.get_special_tokens_mask(query_ids, table_ids)
            else:
                encoded_inputs["special_tokens_mask"] = [0] * len(input_ids)

        # Check lengths
        if max_length is None and len(encoded_inputs["input_ids"]) > self.model_max_length and verbose:
            if not self.deprecation_warnings.get("sequence-length-is-longer-than-the-specified-maximum", False):
                logger.warning(
                    "Token indices sequence length is longer than the specified maximum sequence length "
                    "for this model ({} > {}). Running this sequence through the model will result in "
                    "indexing errors".format(len(encoded_inputs["input_ids"]), self.model_max_length)
                )
            self.deprecation_warnings["sequence-length-is-longer-than-the-specified-maximum"] = True

        # Padding
        if padding != PaddingStrategy.DO_NOT_PAD or return_attention_mask:
            encoded_inputs = self.pad(
                encoded_inputs,
                max_length=max_length,
                padding=padding.value,
                pad_to_multiple_of=pad_to_multiple_of,
                return_attention_mask=return_attention_mask,
            )

        if return_length:
            encoded_inputs["length"] = len(encoded_inputs["input_ids"])

        batch_outputs = BatchEncoding(
            encoded_inputs, tensor_type=return_tensors, prepend_batch_axis=prepend_batch_axis
        )

        return batch_outputs