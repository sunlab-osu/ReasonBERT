from torch.nn.functional import embedding
from transformers.utils import logging
import webdataset as wds
import random
import math
import numpy as np
from transformers import AutoTokenizer, TapasTokenizer, BertTokenizerFast
from data_loader.table_tokenizer import TableTokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data._utils.collate import default_collate
from webdataset.autodecode import Continue
import os
import json
import random
import torch
import pandas as pd
import re
import pickle
from tqdm import tqdm
import copy
import data_loader.table_utils as table_utils

from multiprocessing import Pool
from functools import partial
import itertools
import collections

import pdb

logger = logging.get_logger(__name__)

def get_tokenized_loc(tokenized_input, start, end):
    if start<0 or end<0:
        return None, None
    tokenized_start = tokenized_input.char_to_token(start)
    while tokenized_start is None and start<end:
        start += 1
        tokenized_start = tokenized_input.char_to_token(start)
    tokenized_end = tokenized_input.char_to_token(end)
    while tokenized_end is None and end>start:
        end -= 1
        tokenized_end = tokenized_input.char_to_token(end)
    return tokenized_start, tokenized_end


class SentencePairProcessor:
    def __init__(self, tokenizer, max_seq_length, token_type=False, mlm_probability=0.15, nonanswerable=True, sample_single=False) -> None:
        self.tokenizer = tokenizer
        self.q_id = tokenizer.convert_tokens_to_ids("[QUESTION]")
        self.sep_id = tokenizer.sep_token_id
        self.cls_id = tokenizer.cls_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.max_query_length = max_seq_length[0]
        self.max_evidence_length = max_seq_length[1]
        self.max_input_length = max_seq_length[2]
        self.token_type = token_type
        self.vocab_size = len(tokenizer)
        self.mlm_probability = mlm_probability
        self.nonanswerable = nonanswerable
        self.sample_single = sample_single
        super().__init__()
    def mlm(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        From transformers
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def __call__(self, sample):
        tokenized_query = self.tokenizer(sample['s1_text'], add_special_tokens=False)
        query_ids = tokenized_query['input_ids']
        s1_all_links = {}
        query_entity_mask = np.zeros_like(query_ids)
        for entity, link in sample['s1_all_links'].items():
            s1_all_links[entity] = []
            for start, end in link:
                start, end = get_tokenized_loc(tokenized_query, start, end-1)
                if start is not None and end is not None:
                    query_entity_mask[start:end+1] = 1
                    s1_all_links[entity].append([start,end+1])
        available_pairs = sample['pairs']
        if len(available_pairs)>1:
            sampled_pairs = random.sample(available_pairs, 1 if self.sample_single else 2)
        else:
            sampled_pairs = available_pairs
        num_sample = len(sampled_pairs)
        list_evidence_ids = []
        list_evidence_entity_mask = []
        query_entity_locs = {}
        evidence_entity_locs = {}
        used_pair = []
        all_entity_in_evidences = set()
        max_evidence_length = self.max_evidence_length if len(sampled_pairs)==2 else self.max_evidence_length*2
        for i, sampled_pair in enumerate(sampled_pairs):
            masked_e_id = 0 if random.random()>0.5 else 1
            pair = sampled_pair['pair']
            s1_pair_locs = [s1_all_links.get(pair[0], []), s1_all_links.get(pair[1], [])]
            used_pair.append(pair)
            masked_e = pair[masked_e_id]
            s2 = random.choice(sampled_pair['s2s'])

            offset = 0
            tokenized_s2 = self.tokenizer(s2['text'], add_special_tokens=False)
            s2['ids'] = tokenized_s2['input_ids']
            s2_char_start, s2_char_end =  s2['s_loc']
            s2_start, s2_end = get_tokenized_loc(tokenized_s2,s2_char_start, s2_char_end-1)
            if len(s2['ids'])>max_evidence_length:
                s2_length = s2_end - s2_start + 1
                if s2_length>max_evidence_length:
                    for start, end in s2['pair_locs'][masked_e_id]:
                        if start>=s2_char_start and start<s2_char_end:
                            start, end = get_tokenized_loc(tokenized_s2, start, end-1)
                            if start is None or end is None:
                                continue
                            span_length = end - start + 1
                            if span_length < max_evidence_length:
                                offset = max([0, start-math.ceil((max_evidence_length-span_length)/2)])
                                break
                else:
                    offset = max([0, s2_start-math.ceil((max_evidence_length-s2_length)/2)])
            s2_ids = s2['ids'][offset:offset+max_evidence_length]
            s2_entity_mask = np.zeros_like(s2_ids)
            s2_all_links = {}
            for entity, link in s2['all_links'].items():
                s2_all_links[entity] = []
                for start, end in link:
                    start, end = get_tokenized_loc(tokenized_s2, start, end-1)
                    if start is not None and end is not None and start>=offset and end<offset+max_evidence_length:
                        s2_entity_mask[start-offset:end-offset+1] = 1
                        s2_all_links[entity].append([start-offset,end-offset+1])
            all_entity_in_evidences |= s2_all_links.keys()
            list_evidence_ids.append(s2_ids)
            list_evidence_entity_mask.append(s2_entity_mask)
            s2_pair_locs = [s2_all_links.get(pair[0], []), s2_all_links.get(pair[1], [])]
            for e,locs in zip(pair,s2_pair_locs):
                if e not in evidence_entity_locs:
                    evidence_entity_locs[e] = [[] for _ in range(num_sample)]
                for loc in locs:
                    if loc[0] is not None and loc[1] is not None:
                        evidence_entity_locs[e][i].append(loc)
            if masked_e not in query_entity_locs:
                query_entity_locs[masked_e] = s1_pair_locs[masked_e_id]
        # mask query with [QUESTION]
        all_query_entities = set(s1_all_links.keys())
        query_entities_not_shown_in_evidences = all_query_entities - all_entity_in_evidences
        if len(query_entities_not_shown_in_evidences)!=0 and self.nonanswerable:
            non_answerable_entity = random.choice(list(query_entities_not_shown_in_evidences))
            non_answerable_entity_locs = s1_all_links[non_answerable_entity]
        else:
            non_answerable_entity = None
            non_answerable_entity_locs = []


        query_entity_locs = sorted(
            [[e, loc] for e,locs in query_entity_locs.items() for loc in locs]\
            + ([[non_answerable_entity, loc] for loc in non_answerable_entity_locs] if non_answerable_entity is not None else []),\
            key=lambda x:x[1][0])
        masked_query_ids = []
        query_mlm_mask = []
        per_mask_target = []
        per_mask_loc = []
        last = 0
        used_e = set()
        query_entity_mask = query_entity_mask.tolist()
        for e, loc in query_entity_locs:
            if len(masked_query_ids) + loc[0] - last + 1 > self.max_query_length:
                if last == 0:
                    last = loc[0]-math.ceil(self.max_query_length/2)
                else:
                    masked_query_ids += query_ids[last:last+self.max_query_length-len(masked_query_ids)]
                    query_mlm_mask += query_entity_mask[last:last+self.max_query_length-len(query_mlm_mask)]
                    break
            masked_query_ids += query_ids[last:loc[0]]
            query_mlm_mask += query_entity_mask[last:loc[0]]
            if e not in used_e:
                if e == non_answerable_entity:
                    per_mask_target.append(None)
                    per_mask_loc.append(len(masked_query_ids))
                    used_e.add(e)
                else:
                    per_mask_target.append([[] for _ in range(num_sample)])
                    for i, locs_in_evidence in enumerate(evidence_entity_locs[e]):
                        per_mask_target[-1][i] = locs_in_evidence
                    per_mask_loc.append(len(masked_query_ids))
                    used_e.add(e)
            masked_query_ids.append(self.q_id)
            query_mlm_mask.append(1)
            last = loc[1]
        if len(masked_query_ids)<self.max_query_length:
            masked_query_ids += query_ids[last:last+self.max_query_length-len(masked_query_ids)]
            query_mlm_mask += query_entity_mask[last:last+self.max_query_length-len(query_mlm_mask)]
        # formulate the input
        # use concat
        # [CLS] query [SEP] evidence_0 [SEP] evidence_1 [SEP]
        final_mlm_mask = torch.zeros(self.max_input_length)
        final_input_ids = torch.full((self.max_input_length,), self.pad_id)
        final_input_type_ids = torch.zeros(self.max_input_length, dtype=int)
        final_attention_mask = torch.zeros(self.max_input_length)
        final_q_loc = torch.zeros(3, dtype=int)
        final_q_mask = torch.zeros(3, dtype=int)
        final_target_start = torch.zeros((3, self.max_input_length))
        final_target_end = torch.zeros((3, self.max_input_length))
        final_target_mask = torch.zeros(self.max_input_length)
        final_target_mask[0] = 1 #[CLS] is the target for nonanswrable
        final_input_ids[0] = self.cls_id
        final_mlm_mask[0] = 1
        final_input_ids[1:1+len(masked_query_ids)] = torch.as_tensor(masked_query_ids)
        final_mlm_mask[1:1+len(masked_query_ids)] = torch.as_tensor(query_mlm_mask)
        final_input_ids[len(masked_query_ids)+1] = self.sep_id
        final_mlm_mask[len(masked_query_ids)+1] = 1
        p = len(masked_query_ids)+2
        for evidence_ids, evidence_entity_mask in zip(list_evidence_ids, list_evidence_entity_mask):
            final_input_ids[p:p+len(evidence_ids)] = torch.as_tensor(evidence_ids)
            final_mlm_mask[p:p+len(evidence_ids)] = torch.as_tensor(evidence_entity_mask)
            final_target_mask[p:p+len(evidence_ids)] = 1
            final_input_ids[p+len(evidence_ids)] = self.sep_id
            final_mlm_mask[p+len(evidence_ids)] = 1
            if self.token_type:
                final_input_type_ids[p:p+len(evidence_ids)+1] = 1
            p += len(evidence_ids)+1
        final_attention_mask[:p] = 1
        final_mlm_mask[p+1:] = 1
        for i, (loc, target) in enumerate(zip(per_mask_loc[:3], per_mask_target[:3])):
            if target is None: #nonanswerable
                final_q_mask[i] = 1
                final_q_loc[i] = loc+1 # extra [CLS]
                #nonanswerble points to [CLS]
                final_target_start[i,0] = 1
                final_target_end[i,0] = 1
            else:
                if all([len(target_locs)==0 for target_locs in target]):
                    continue
                final_q_mask[i] = 1
                final_q_loc[i] = loc+1 # extra [CLS]
                offset = len(masked_query_ids)+2
                for j, target_locs in enumerate(target):
                    for loc in target_locs:
                        final_target_start[i,loc[0]+offset] = 1
                        final_target_end[i,loc[1]-1+offset] = 1
                    offset += len(list_evidence_ids[j])+1
        final_input_ids = torch.as_tensor(final_input_ids)
        final_mlm_mask = torch.as_tensor(final_mlm_mask)
        if self.mlm_probability!=-1:
            final_input_ids, final_mlm_target = self.mlm(final_input_ids, final_mlm_mask)
            return {
                'input_ids': final_input_ids,
                'token_type_ids': final_input_type_ids,
                'attention_mask': final_attention_mask,
                'q_loc': final_q_loc,
                'q_mask': final_q_mask,
                'target_start': final_target_start,
                'target_end': final_target_end,
                'target_mask': final_target_mask,
                'mlm_target': final_mlm_target
            }
        else:
            return {
                'input_ids': final_input_ids,
                'token_type_ids': final_input_type_ids,
                'attention_mask': final_attention_mask,
                'q_loc': final_q_loc,
                'q_mask': final_q_mask,
                'target_start': final_target_start,
                'target_end': final_target_end,
                'target_mask': final_target_mask,
            }
        
def tmp_func(x):
    return x[0]
    
class node_selector:
    def __init__(self, index, num_cores=8):
        self.index = index
        self.num_cores = num_cores
    
    def __call__(self, urls_):
        """Split urls correctly per accelerator.
        :param urls_:
        :return: slice of urls_
        """
        urls_this = urls_[self.index::self.num_cores]
        return urls_this

class SentencePairDataset(wds.WebDataset):
    def __init__(
        self, tokenizer='bert-base-uncased', urls='', 
        shuffle_cache_size=500, batch_size=20, 
        max_seq_length=[100, 200, 512],
        n_gpus=1, length=None, mlm_probability=-1, nonanswerable=True, sample_single=False,
        index=0, num_cores=1
        ):
        super().__init__(urls)
        self.node_selection = node_selector(index, num_cores)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[QUESTION]"]})
        processor = SentencePairProcessor(self.tokenizer,
                                    max_seq_length,
                                    token_type=tokenizer.startswith('bert'),
                                    mlm_probability=mlm_probability,
                                    nonanswerable=nonanswerable,sample_single=sample_single)
        self.shuffle(shuffle_cache_size)
        self.decode()
        self.to_tuple("json")
        self.map_tuple(processor)
        self.batched(batch_size*n_gpus)
        self.map(tmp_func)

class HybridPairProcessor(SentencePairProcessor):
    def __init__(self, tokenizer, max_seq_length, token_type=False, mlm_probability=0.15) -> None:
        super().__init__(tokenizer, max_seq_length, token_type, mlm_probability)
        self.max_column_num = 5
        self.max_cell_length = 20
        
    def __call__(self, sample):
        tokenized_query = self.tokenizer(sample['s1_text'], add_special_tokens=False)
        query_ids = tokenized_query['input_ids']
        s1_all_links = {}
        query_entity_mask = np.zeros_like(query_ids)
        for entity, link in sample['s1_all_links'].items():
            s1_all_links[entity] = []
            for start, end in link:
                start, end = get_tokenized_loc(tokenized_query, start, end-1)
                if start is not None and end is not None:
                    query_entity_mask[start:end+1] = 1
                    s1_all_links[entity].append([start,end+1])
        available_sentence_pairs = sample['sentence_pairs']
        num_sample = 0
        available_table_pairs = sample['table_pairs']
        if available_table_pairs:
            sampled_table_pair = random.choice(available_table_pairs)
            sampled_table_match_pair = random.choice(sampled_table_pair['pairs'])
            table_pair = tuple(sampled_table_match_pair['pair'])
            num_sample += 1
        else:
            sampled_table_pair = None
        if available_sentence_pairs:
            unused_sentence_pairs = [sentence_pair for sentence_pair in available_sentence_pairs if tuple(sentence_pair['pair'])!=table_pair]
            if unused_sentence_pairs:
                sampled_sentence_pair = random.choice(unused_sentence_pairs)
                num_sample += 1
            else:
                sampled_sentence_pair = None
        else:
            sampled_sentence_pair = None
        list_evidence_ids = []
        list_evidence_column_ids = []
        list_evidence_row_ids = []
        list_evidence_column_ranks = []
        list_evidence_column_inv_ranks = []
        list_evidence_entity_mask = []
        query_entity_locs = {}
        evidence_entity_locs = {}
        evidence_entity_cell_index = {}
        used_pair = []
        all_entity_in_evidences = set()
        max_evidence_length = self.max_evidence_length if (num_sample==2) else self.max_evidence_length*2
        if sampled_table_pair is not None:
            table = sampled_table_pair['text']
            table_all_links = sampled_table_pair['all_links']
            table_index = sampled_table_pair['index']
            table_value_ranks = sampled_table_pair['value_ranks']
            table_value_inv_ranks = sampled_table_pair['value_inv_ranks']
            masked_e_id = 0 if random.random()>0.5 else 1
            pair = sampled_table_match_pair['pair']
            s1_pair_locs = []
            for e, locs in zip(pair, sampled_table_match_pair['s1_pair_locs']):
                if e in s1_all_links:
                    s1_pair_locs.append(s1_all_links[e])
                else:
                    tokenized_locs = []
                    for start, end in locs:
                        start, end = get_tokenized_loc(tokenized_query, start, end-1)
                        if start is not None and end is not None:
                            query_entity_mask[start:end+1] = 1
                            tokenized_locs.append([start,end+1])
                    s1_pair_locs.append(tokenized_locs)
            table_pair_loc = random.choice(list(sampled_table_match_pair['table_pair_locs'].items()))
            target_row = int(table_pair_loc[0])
            table_pair_loc = copy.deepcopy([random.choice(table_pair_loc[1][0]), random.choice(table_pair_loc[1][1])])
            target_column = set(([index[1] for index,_ in table_pair_loc]+[0]))
            if len(table[0])>self.max_column_num:
                columns_to_use = random.sample(set(range(len(table[0])))-target_column, self.max_column_num-len(target_column))+list(target_column)
            else:
                columns_to_use = list(range(len(table[0])))
            columns_to_use.sort()
            available_rows = [i for i in range(1,len(table)) if i!=target_row]
            random.shuffle(available_rows)
            table_ids = {}
            table_column_ids = {}
            table_row_ids = {}
            table_column_ranks = {}
            table_column_inv_ranks = {}
            total_table_length = 0
            if target_row!=0:
                # add header row
                row_ids = []
                row_column_ids = []
                row_row_ids = []
                row_column_ranks = []
                row_column_inv_ranks = []
                for j in columns_to_use:
                    cell = table[0][j]
                    cell_ids = self.tokenizer(' '+cell, add_special_tokens=False)['input_ids'][:self.max_cell_length]
                    cell_length = len(cell_ids)
                    total_table_length += cell_length
                    row_ids += cell_ids
                    row_column_ids += [(table_index[0][j][1] if table_index[0][j][1]<128 else table_index[0][j][1]%128)+1]*cell_length # column 0 is for text only part
                    row_row_ids += [(table_index[0][j][0] if table_index[0][j][0]<128 else table_index[0][j][0]%128)]*cell_length
                    row_column_ranks += [table_value_ranks[0][j]]*cell_length
                    row_column_inv_ranks += [table_value_inv_ranks[0][j]]*cell_length
                table_ids[0] = row_ids
                table_column_ids[0] = row_column_ids
                table_row_ids[0] = row_row_ids
                table_column_ranks[0] = row_column_ranks
                table_column_inv_ranks[0] = row_column_inv_ranks
            # add target row
            row_ids = []
            row_column_ids = []
            row_row_ids = []
            row_column_ranks = []
            row_column_inv_ranks = []
            for j in columns_to_use:
                cell = table[target_row][j]
                tokenized_cell = self.tokenizer(' '+cell, add_special_tokens=False)
                cell_ids = tokenized_cell['input_ids']
                cell_offset = 0
                if j == table_pair_loc[0][0][1]:
                    start, end = table_pair_loc[0][1]
                    start, end = get_tokenized_loc(tokenized_cell, start+1, end)
                    if start is not None and end is not None:
                        end += 1
                        if end > self.max_cell_length:
                            cell_offset = start
                            cell_ids = cell_ids[start:start+self.max_cell_length]
                            table_pair_loc[0][1] = [len(row_ids),len(row_ids)+min([end-start,self.max_cell_length])]
                        else:
                            table_pair_loc[0][1] = [len(row_ids)+start, len(row_ids)+end]
                        evidence_entity_cell_index[pair[0]] = [
                            (table_index[target_row][j][0] if table_index[target_row][j][0]<128 else table_index[target_row][j][0]%128),
                            (table_index[target_row][j][1] if table_index[target_row][j][1]<128 else table_index[target_row][j][1]%128)+1
                            ]
                    else:
                        table_pair_loc[0][1] = [None, None]
                if j == table_pair_loc[1][0][1]:
                    start, end = table_pair_loc[1][1]
                    start, end = get_tokenized_loc(tokenized_cell, start+1, end) # extra prefix whitespace
                    if start is not None and end is not None:
                        start -= cell_offset
                        end -= cell_offset
                    if start is not None and end is not None and start >= 0 and (j!=table_pair_loc[0][0][1] or end<self.max_cell_length):
                        end += 1
                        if end > self.max_cell_length:
                            cell_ids = cell_ids[start:start+self.max_cell_length]
                            table_pair_loc[1][1] = [len(row_ids),len(row_ids)+min([end-start,self.max_cell_length])]
                        else:
                            table_pair_loc[1][1] = [len(row_ids)+start, len(row_ids)+end]
                        evidence_entity_cell_index[pair[1]] = [
                            (table_index[target_row][j][0] if table_index[target_row][j][0]<128 else table_index[target_row][j][0]%128),
                            (table_index[target_row][j][1] if table_index[target_row][j][1]<128 else table_index[target_row][j][1]%128)+1
                            ]
                    else:
                        table_pair_loc[1][1] = [None, None]
                cell_ids = cell_ids[:self.max_cell_length]
                cell_length = len(cell_ids)
                total_table_length += cell_length
                row_ids += cell_ids
                row_column_ids += [(table_index[target_row][j][1] if table_index[target_row][j][1]<128 else table_index[target_row][j][1]%128)+1]*cell_length # column 0 is for text only part
                row_row_ids += [(table_index[target_row][j][0] if table_index[target_row][j][0]<128 else table_index[target_row][j][0]%128)]*cell_length
                row_column_ranks += [table_value_ranks[target_row][j]]*cell_length
                row_column_inv_ranks += [table_value_inv_ranks[target_row][j]]*cell_length
            table_ids[target_row] = row_ids
            table_column_ids[target_row] = row_column_ids
            table_row_ids[target_row] = row_row_ids
            table_column_ranks[target_row] = row_column_ranks
            table_column_inv_ranks[target_row] = row_column_inv_ranks
            for i in available_rows:
                if total_table_length>max_evidence_length:
                    break
                row_ids = []
                row_column_ids = []
                row_row_ids = []
                row_column_ranks = []
                row_column_inv_ranks = []
                for j in columns_to_use:
                    cell = table[i][j]
                    cell_ids = self.tokenizer(' '+cell, add_special_tokens=False)['input_ids'][:self.max_cell_length]
                    cell_length = len(cell_ids)
                    total_table_length += cell_length
                    row_ids += cell_ids
                    row_column_ids += [(table_index[i][j][1] if table_index[i][j][1]<128 else table_index[i][j][1]%128)+1]*cell_length # column 0 is for text only part
                    row_row_ids += [(table_index[i][j][0] if table_index[i][j][0]<128 else table_index[i][j][0]%128)]*cell_length
                    row_column_ranks += [table_value_ranks[i][j]]*cell_length
                    row_column_inv_ranks += [table_value_inv_ranks[i][j]]*cell_length
                if total_table_length+len(row_ids)>max_evidence_length:
                    break
                table_ids[i] = row_ids
                table_column_ids[i] = row_column_ids
                table_row_ids[i] = row_row_ids
                table_column_ranks[i] = row_column_ranks
                table_column_inv_ranks[i] = row_column_inv_ranks
            used_pair.append(pair)
            masked_e = pair[masked_e_id]
            flatten_table_ids = []
            flatten_table_column_ids = []
            flatten_table_row_ids = []
            flatten_table_column_ranks = []
            flatten_table_column_inv_ranks = []
            for i in range(len(table)):
                if i in table_ids:
                    if i==target_row:
                        if table_pair_loc[0][1][0] is not None:
                            table_pair_loc[0][1] = [len(flatten_table_ids)+table_pair_loc[0][1][0],len(flatten_table_ids)+table_pair_loc[0][1][1]]
                        if table_pair_loc[1][1][0] is not None:
                            table_pair_loc[1][1] = [len(flatten_table_ids)+table_pair_loc[1][1][0],len(flatten_table_ids)+table_pair_loc[1][1][1]]
                    flatten_table_ids += table_ids[i]
                    flatten_table_column_ids += table_column_ids[i]
                    flatten_table_row_ids += table_row_ids[i]
                    flatten_table_column_ranks += table_column_ranks[i]
                    flatten_table_column_inv_ranks += table_column_inv_ranks[i]
            table_entity_mask = np.ones_like(flatten_table_ids)
            if target_row!=0:
                table_entity_mask[:len(table_ids[0])] = 0
            all_entity_in_evidences |= table_all_links.keys()
            list_evidence_ids.append(flatten_table_ids)
            list_evidence_column_ids.append(flatten_table_column_ids)
            list_evidence_row_ids.append(flatten_table_row_ids)
            list_evidence_column_ranks.append(flatten_table_column_ranks)
            list_evidence_column_inv_ranks.append(flatten_table_column_inv_ranks)
            list_evidence_entity_mask.append(table_entity_mask)
            for e,(_,loc) in zip(pair,table_pair_loc):
                if e not in evidence_entity_locs:
                    evidence_entity_locs[e] = [[] for _ in range(num_sample)]
                if loc[0] is not None and loc[1] is not None:
                    evidence_entity_locs[e][0].append(loc)
            if masked_e not in query_entity_locs:
                query_entity_locs[masked_e] = s1_pair_locs[masked_e_id]
        if sampled_sentence_pair is not None:
            masked_e_id = 0 if random.random()>0.5 else 1
            pair = sampled_sentence_pair['pair']
            s1_pair_locs = [s1_all_links.get(pair[0], []), s1_all_links.get(pair[1], [])]
            used_pair.append(pair)
            masked_e = pair[masked_e_id]
            s2 = random.choice(sampled_sentence_pair['s2s'])
            offset = 0
            tokenized_s2 = self.tokenizer(s2['text'], add_special_tokens=False)
            s2['ids'] = tokenized_s2['input_ids']
            s2_char_start, s2_char_end =  s2['s_loc']
            s2_start, s2_end = get_tokenized_loc(tokenized_s2,s2_char_start, s2_char_end-1)
            if len(s2['ids'])>max_evidence_length:
                s2_length = s2_end - s2_start + 1
                if s2_length>max_evidence_length:
                    for start, end in s2['pair_locs'][masked_e_id]:
                        if start>=s2_char_start and start<s2_char_end:
                            start, end = get_tokenized_loc(tokenized_s2, start, end-1)
                            if start is None or end is None:
                                continue
                            span_length = end - start + 1
                            if span_length < max_evidence_length:
                                offset = max([0, start-math.ceil((max_evidence_length-span_length)/2)])
                                break
                else:
                    offset = max([0, s2_start-math.ceil((max_evidence_length-s2_length)/2)])
            s2_ids = s2['ids'][offset:offset+max_evidence_length]
            s2_entity_mask = np.zeros_like(s2_ids)
            s2_all_links = {}
            for entity, link in s2['all_links'].items():
                s2_all_links[entity] = []
                for start, end in link:
                    start, end = get_tokenized_loc(tokenized_s2, start, end-1)
                    if start is not None and end is not None and start>=offset and end<offset+max_evidence_length:
                        s2_entity_mask[start-offset:end-offset+1] = 1
                        s2_all_links[entity].append([start-offset,end-offset+1])
            all_entity_in_evidences |= s2_all_links.keys()
            list_evidence_ids.append(s2_ids)
            e_in_table = pair[0] if pair[0] in evidence_entity_cell_index else pair[1]
            sent_cell_index = evidence_entity_cell_index.get(e_in_table, [0,0])
            list_evidence_column_ids.append([sent_cell_index[1]]*len(s2_ids))
            list_evidence_row_ids.append([sent_cell_index[0]]*len(s2_ids))

            list_evidence_column_ranks.append([0]*len(s2_ids))
            list_evidence_column_inv_ranks.append([0]*len(s2_ids))
            list_evidence_entity_mask.append(s2_entity_mask)
            s2_pair_locs = [s2_all_links.get(pair[0], []), s2_all_links.get(pair[1], [])]
            for e,locs in zip(pair,s2_pair_locs):
                if e not in evidence_entity_locs:
                    evidence_entity_locs[e] = [[] for _ in range(num_sample)]
                for loc in locs:
                    if loc[0] is not None and loc[1] is not None:
                        evidence_entity_locs[e][num_sample-1].append(loc)
            if masked_e not in query_entity_locs:
                query_entity_locs[masked_e] = s1_pair_locs[masked_e_id]
        # mask query with [QUESTION]
        all_query_entities = set(s1_all_links.keys())
        query_entities_not_shown_in_evidences = all_query_entities - all_entity_in_evidences
        if len(query_entities_not_shown_in_evidences)!=0:
            non_answerable_entity = random.choice(list(query_entities_not_shown_in_evidences))
            non_answerable_entity_locs = s1_all_links[non_answerable_entity]
        else:
            non_answerable_entity = None
            non_answerable_entity_locs = []

        query_entity_locs = sorted(
            [[e, loc] for e,locs in query_entity_locs.items() for loc in locs]\
            + ([[non_answerable_entity, loc] for loc in non_answerable_entity_locs] if non_answerable_entity is not None else []),\
            key=lambda x:x[1][0])
        masked_query_ids = []
        query_mlm_mask = []
        per_mask_target = []
        per_mask_loc = []
        last = 0
        used_e = set()
        query_entity_mask = query_entity_mask.tolist()
        for e, loc in query_entity_locs:
            if len(masked_query_ids) + loc[0] - last + 1 > self.max_query_length:
                if last == 0:
                    last = loc[0]-math.ceil(self.max_query_length/2)
                else:
                    masked_query_ids += query_ids[last:last+self.max_query_length-len(masked_query_ids)]
                    query_mlm_mask += query_entity_mask[last:last+self.max_query_length-len(query_mlm_mask)]
                    break
            masked_query_ids += query_ids[last:loc[0]]
            query_mlm_mask += query_entity_mask[last:loc[0]]
            if e not in used_e:
                if e == non_answerable_entity:
                    per_mask_target.append(None)
                    per_mask_loc.append(len(masked_query_ids))
                    used_e.add(e)
                else:
                    per_mask_target.append([[] for _ in range(num_sample)])
                    for i, locs_in_evidence in enumerate(evidence_entity_locs[e]):
                        per_mask_target[-1][i] = locs_in_evidence
                    per_mask_loc.append(len(masked_query_ids))
                    used_e.add(e)
            masked_query_ids.append(self.q_id)
            query_mlm_mask.append(1)
            last = loc[1]
        if len(masked_query_ids)<self.max_query_length:
            masked_query_ids += query_ids[last:last+self.max_query_length-len(masked_query_ids)]
            query_mlm_mask += query_entity_mask[last:last+self.max_query_length-len(query_mlm_mask)]
        # formulate the input
        # use concat
        # [CLS] query [SEP] evidence_0 [SEP] evidence_1 [SEP]
        final_mlm_mask = torch.zeros(self.max_input_length)
        final_input_ids = torch.full((self.max_input_length,), self.pad_id)
        final_input_column_ids = torch.zeros(self.max_input_length, dtype=int)
        final_input_row_ids = torch.zeros(self.max_input_length, dtype=int)
        final_input_column_ranks = torch.zeros(self.max_input_length, dtype=int)
        final_input_column_inv_ranks = torch.zeros(self.max_input_length, dtype=int)
        final_input_segment_ids = torch.zeros(self.max_input_length, dtype=int)
        final_attention_mask = torch.zeros(self.max_input_length)
        final_q_loc = torch.zeros(3, dtype=int)
        final_q_mask = torch.zeros(3, dtype=int)
        final_q_cell_mask = torch.zeros(3, dtype=int)
        final_target_start = torch.zeros((3, self.max_input_length))
        final_target_end = torch.zeros((3, self.max_input_length))
        final_target_row = torch.zeros((3, 256))
        final_target_column = torch.zeros((3, 256))
        final_target_mask = torch.zeros(self.max_input_length)
        final_target_mask[0] = 1 #[CLS] is the target for nonanswrable
        final_input_ids[0] = self.cls_id
        final_mlm_mask[0] = 1
        final_input_ids[1:1+len(masked_query_ids)] = torch.as_tensor(masked_query_ids)
        final_mlm_mask[1:1+len(masked_query_ids)] = torch.as_tensor(query_mlm_mask)
        final_input_ids[len(masked_query_ids)+1] = self.sep_id
        final_mlm_mask[len(masked_query_ids)+1] = 1
        p = len(masked_query_ids)+2
        for evidence_ids, evidence_column_ids, evidence_row_ids, evidence_column_ranks, evidence_column_inv_ranks, evidence_entity_mask in zip(list_evidence_ids, list_evidence_column_ids, list_evidence_row_ids, list_evidence_column_ranks, list_evidence_column_inv_ranks, list_evidence_entity_mask):
            if p+len(evidence_ids)>self.max_input_length:
                print(sample['s1_text'].encode(), sampled_table_pair['tid'], p, len(masked_query_ids))
            final_input_ids[p:p+len(evidence_ids)] = torch.as_tensor(evidence_ids)
            final_mlm_mask[p:p+len(evidence_ids)] = torch.as_tensor(evidence_entity_mask)
            final_target_mask[p:p+len(evidence_ids)] = 1
            final_input_ids[p+len(evidence_ids)] = self.sep_id
            final_mlm_mask[p+len(evidence_ids)] = 1
            if self.token_type:
                final_input_column_ids[p:p+len(evidence_ids)] = torch.as_tensor(evidence_column_ids)
                final_input_row_ids[p:p+len(evidence_ids)] = torch.as_tensor(evidence_row_ids)
                final_input_column_ranks[p:p+len(evidence_ids)] = torch.as_tensor(evidence_column_ranks)
                final_input_column_inv_ranks[p:p+len(evidence_ids)] = torch.as_tensor(evidence_column_inv_ranks)
                final_input_segment_ids[p:p+len(evidence_ids)+1] = 1
            p += len(evidence_ids)+1
        final_attention_mask[:p] = 1
        final_mlm_mask[p+1:] = 1
        for i, (loc, target) in enumerate(zip(per_mask_loc[:3], per_mask_target[:3])):
            if target is None: #nonanswerable
                final_q_mask[i] = 1
                final_q_loc[i] = loc+1 # extra [CLS]
                #nonanswerble points to [CLS]/[0,0]
                final_target_start[i,0] = 1
                final_target_end[i,0] = 1
                final_q_cell_mask[i] = 1
                final_target_row[i,0] = 1
                final_target_column[i,0] = 1
            else:
                if all([len(target_locs)==0 for target_locs in target]):
                    continue
                final_q_mask[i] = 1
                final_q_loc[i] = loc+1 # extra [CLS]
                offset = len(masked_query_ids)+2
                for j, target_locs in enumerate(target):
                    for loc in target_locs:
                        final_target_start[i,loc[0]+offset] = 1
                        final_target_end[i,loc[1]-1+offset] = 1
                        if final_input_column_ids[loc[0]+offset]>0:
                            final_q_cell_mask[i] = 1
                            final_target_row[i,final_input_row_ids[loc[0]+offset]] = 1
                            final_target_column[i,final_input_column_ids[loc[0]+offset]] = 1
                    offset += len(list_evidence_ids[j])+1
        final_input_type_ids = torch.stack([
            final_input_segment_ids,
            final_input_column_ids,
            final_input_row_ids,
            torch.zeros(self.max_input_length, dtype=int), #prev_labels,
            final_input_column_ranks.clamp(0,128),
            final_input_column_inv_ranks.clamp(0,128),
            torch.zeros(self.max_input_length, dtype=int) #numeric_relations
        ], dim=1)
        final_input_ids = torch.as_tensor(final_input_ids)
        final_mlm_mask = torch.as_tensor(final_mlm_mask)
        if self.mlm_probability!=-1:
            final_input_ids, final_mlm_target = self.mlm(final_input_ids, final_mlm_mask)
            return {
                'input_ids': final_input_ids,
                'token_type_ids': final_input_type_ids,
                'attention_mask': final_attention_mask,
                'q_loc': final_q_loc,
                'q_mask': final_q_mask,
                'q_cell_mask': final_q_cell_mask,
                'target_start': final_target_start,
                'target_end': final_target_end,
                'target_row': final_target_row,
                'target_column': final_target_column,
                'target_mask': final_target_mask,
                'mlm_target': final_mlm_target
            }
        else:
            return {
                'input_ids': final_input_ids,
                'token_type_ids': final_input_type_ids,
                'attention_mask': final_attention_mask,
                'q_loc': final_q_loc,
                'q_mask': final_q_mask,
                'q_cell_mask': final_q_cell_mask,
                'target_start': final_target_start,
                'target_end': final_target_end,
                'target_row': final_target_row,
                'target_column': final_target_column,
                'target_mask': final_target_mask,
            }

class HybridPairDataset(wds.WebDataset):
    def __init__(
        self, tokenizer='bert-base-uncased', urls='', 
        shuffle_cache_size=500, batch_size=20, 
        max_seq_length=[100, 200, 512],
        n_gpus=1, length=None, mlm_probability=-1,
        index=0, num_cores=1, token_type=False
        ):
        super().__init__(urls)
        self.node_selection = node_selector(index, num_cores)
        if 'tapas' in tokenizer:
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens({"additional_special_tokens": ["[QUESTION]"]})
        processor = HybridPairProcessor(self.tokenizer,
                                    max_seq_length,
                                    token_type=token_type,
                                    mlm_probability=mlm_probability)
        self.shuffle(shuffle_cache_size)
        self.decode()
        self.to_tuple("json")
        self.map_tuple(processor)
        self.batched(batch_size*n_gpus)
        self.map(tmp_func)

class MyDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, num_workers, pin_memory, nominal_length=-1):
        self.nominal_length = nominal_length
        super().__init__(dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=default_collate, pin_memory=pin_memory)
    def __len__(self) -> int:
        if self.nominal_length==-1:
            return super().__len__()
        else:
            return self.nominal_length

def MRQA_preprocess(x, tokenizer, is_train=True):
    if isinstance(x, str):
        x = json.loads(x)
    sep_id = tokenizer.vocab['[SEP]']
    context = x['context']
    tokenized_context = tokenizer(context.strip(), add_special_tokens=False)
    tokenized_context_ids = [token_id if token_id!=sep_id else tokenizer.sep_token_id for token_id in tokenized_context['input_ids']]
    uid = x.get('id', 'nan')
    qas = []
    for y in x['qas']:
        question = y['question']
        answers = y['answers']
        detected_answers = y['detected_answers']
        offset = 0
        chunked_examples = []
        tokenized_question = tokenizer(question.strip(), add_special_tokens=False)
        max_context_length = 500 - len(tokenized_question['input_ids'])
        target_mask = []
        if 'target_mask' in y:
            for start, end in y['target_mask']:
                start, end = get_tokenized_loc(tokenized_context, start, end-1)
                if start is not None and end is not None:
                    target_mask.append([start, end+1])
        while offset < len(tokenized_context_ids):
            context_chunk_ids = tokenized_context_ids[offset:offset+max_context_length]
            answer_in_current_chunk = None
            for answer in detected_answers:
                for span in answer['char_spans']:
                    span = [tokenized_context.char_to_token(span[0]), tokenized_context.char_to_token(span[1])]
                    if span[0] is not None and span[1] is not None and span[0]>=offset and span[1]<offset+max_context_length:
                        answer_in_current_chunk = {'text':answer['text'], 'span':[span[0]-offset, span[1]-offset]}
                        break
            if answer_in_current_chunk is not None:
                chunked_examples.append({
                    'context_chunk_ids': context_chunk_ids,
                    'answer': answer_in_current_chunk['text'],
                    'answer_span': answer_in_current_chunk['span'],
                    'target_mask': [[start-offset, end-offset] for start, end in target_mask]
                })
            else:
                chunked_examples.append({
                    'context_chunk_ids': context_chunk_ids,
                    'answer': None,
                    'answer_span': None,
                    'target_mask': [[start-offset, end-offset] for start, end in target_mask]
                })
            if offset+max_context_length >= len(tokenized_context_ids):
                break
            offset += 128
        qas.append({
            'qid': y['qid'],
            'question_ids': tokenized_question['input_ids'],
            'question': question,
            'chunked_examples': chunked_examples,
            'num_chunks': len(chunked_examples),
            'answers': [tokenizer.decode(tokenizer(answer, add_special_tokens=False)['input_ids']) for answer in answers]
        })
    return qas

class MRQADataset(Dataset):
    def __init__(self, datadir, dataset, split, tokenizer='bert-base-uncased', sample_num = -1, overwrite = False, neg_ratio=-1, skip_first_line=False) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['[QUESTION]']+(['[SEP]'] if self.tokenizer.sep_token!='[SEP]' else [])+['[DOC]', '[PAR]', '[TLE]']+[f'[{p}={i}]' for i in range(50) for p in ['P','Tab','List']]})
        preprocessed_file_name = os.path.join(datadir, split, f'processed_{dataset}_{tokenizer.split("/")[-1]}.jsonl')
        logger.info(f"***** loading {split} data for {dataset}")
        if os.path.exists(preprocessed_file_name) and not overwrite:
            data = []
            with open(preprocessed_file_name, 'r') as f_in:
                for line in f_in:
                    data.append(json.loads(line))
            logger.info(f"load {len(data)} preprocessed examples from {preprocessed_file_name}")
        else:
            logger.info(f"cannot find preprocessed file or choose to overwrite, create new one in {preprocessed_file_name}")
            with open(os.path.join(datadir, split, f'{dataset}.jsonl'), 'r') as f_in,\
                open(preprocessed_file_name, 'w') as f_out:
                    if skip_first_line:
                        next(f_in)
                    raw = []
                    for line in tqdm(f_in):
                        raw.append(line)
                    with Pool(8) as pool:
                        data = pool.map(partial(MRQA_preprocess, tokenizer=self.tokenizer, is_train=split=="train"), raw)
                    for qas in data:
                        for sample in qas:
                            f_out.write('%s\n'%json.dumps(sample))
            data = list(itertools.chain(*data))
            logger.info(f"processed {len(data)} examples and saved in {preprocessed_file_name}")
        if sample_num < 1 and sample_num > 0:
            sample_num = int(len(data)*sample_num)
        else:
            sample_num = int(sample_num)
        if sample_num != -1 and sample_num<len(data):
            data = random.sample(data, sample_num)
            logger.info(f"random sample {sample_num} examples")
        else:
            data = data
            logger.info(f"use the full dataset")
        pos_data = []
        neg_data = []
        for sample in data:
            for chunk in sample['chunked_examples']:
                if chunk['answer_span'] is not None:
                    pos_data.append({
                        'qid': sample['qid'],
                        'question_ids': sample['question_ids'],
                        'context_chunk_ids': chunk['context_chunk_ids'],
                        'target_mask': chunk.get('target_mask', []),
                        'answer_span': chunk['answer_span'],
                        'answers': sample['answers']
                    })
                else:
                    neg_data.append({
                        'qid': sample['qid'],
                        'question_ids': sample['question_ids'],
                        'context_chunk_ids': chunk['context_chunk_ids'],
                        'target_mask': chunk.get('target_mask', []),
                        'answer_span': chunk['answer_span'],
                        'answers': sample['answers']
                    })
        
        logger.info(f"{len(pos_data)+len(neg_data)} after chunking the context")
        if neg_ratio!=-1 and sample_num==-1:
            logger.info(f"resample based on negative ratio: {neg_ratio}")
            pos_num = len(pos_data)
            neg_num = int(pos_num*neg_ratio)
            if neg_num<len(neg_data):
                self.data = pos_data + random.sample(neg_data, neg_num)
            else:
                self.data = pos_data + neg_data
            logger.info(f"{len(self.data)} examples after resample")
        else:
            self.data = pos_data + neg_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

class MRQA_collate:
    def __init__(self, tokenizer, use_token_type=True, q_id = None):
        if q_id is None:
            self.q_id = tokenizer.convert_tokens_to_ids("[QUESTION]")
        else:
            self.q_id = q_id
        self.sep_id = tokenizer.sep_token_id
        self.cls_id = tokenizer.cls_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.use_token_type = use_token_type
    def __call__(self, batch_samples):
        batch_size = len(batch_samples)
        batched_input_ids = np.full((batch_size, 512), self.pad_id)
        batched_input_ids[:, 0] = self.cls_id
        batched_input_type_ids = np.full((batch_size, 512), 0)
        batched_attention_mask = np.zeros((batch_size, 512))
        batched_q_loc = np.zeros((batch_size, 1), dtype=int)
        batched_target_start = np.zeros((batch_size, 1), dtype=int)
        batched_target_end = np.zeros((batch_size, 1), dtype=int)
        batched_target_mask = np.zeros((batch_size, 512))
        batched_answers = []
        batched_qid = []
        for batch_id, sample in enumerate(batch_samples):
            question_ids = sample['question_ids']
            context_ids = sample['context_chunk_ids']
            answer_span = sample['answer_span']
            batched_answers.append(sample['answers'])
            batched_qid.append(sample['qid'])
            batched_input_ids[batch_id, 1:1+len(question_ids)] = question_ids
            batched_input_ids[batch_id, 1+len(question_ids)] = self.q_id
            batched_input_ids[batch_id, 2+len(question_ids)] = self.sep_id

            if len(context_ids)>511-(3+len(question_ids)):
                context_ids = context_ids[:511-(3+len(question_ids))]
                if answer_span is not None and answer_span[1]>=511-(3+len(question_ids)):
                    answer_span = None
            batched_input_ids[batch_id, 3+len(question_ids):3+len(question_ids)+len(context_ids)] = context_ids
            batched_input_ids[batch_id, 3+len(question_ids)+len(context_ids)] = self.sep_id
            if self.use_token_type:
                batched_input_type_ids[batch_id, 3+len(question_ids):4+len(question_ids)+len(context_ids)] = 1
            batched_attention_mask[batch_id, :4+len(question_ids)+len(context_ids)] = 1
            batched_q_loc[batch_id, 0] = 1+len(question_ids)
            if answer_span is not None:
                batched_target_start[batch_id, 0] = answer_span[0]+3+len(question_ids)
                batched_target_end[batch_id, 0] = answer_span[1]+3+len(question_ids)
            else:
                batched_target_start[batch_id, 0] = 0
                batched_target_end[batch_id, 0] = 0
            batched_target_mask[batch_id, 0] = 1
            target_mask = sample.get('target_mask', [])
            if target_mask:
                for start, end in target_mask:
                    batched_target_mask[batch_id, 3+len(question_ids)+start:3+len(question_ids)+end] = 1
            else:
                batched_target_mask[batch_id, 3+len(question_ids):3+len(question_ids)+len(context_ids)] = 1
        return {
            'input_ids': torch.as_tensor(batched_input_ids),
            'token_type_ids': torch.as_tensor(batched_input_type_ids),
            'attention_mask': torch.as_tensor(batched_attention_mask),
            'q_loc': torch.as_tensor(batched_q_loc),
            'target_start': torch.as_tensor(batched_target_start),
            'target_end': torch.as_tensor(batched_target_end),
            'target_mask': torch.as_tensor(batched_target_mask),
            'answers': batched_answers,
            'qid': batched_qid
        }

def HotpotQA_preprocess(x, tokenizer, is_train=True, max_context=200, stride=100):
    if isinstance(x, str):
        x = json.loads(x)
    sep_id = tokenizer.vocab['[SEP]']
    supporting_facts = {}
    all_facts = []
    for title, sent_idx in x.get("supporting_facts", []):
        all_facts.append([title, sent_idx])
        if title not in supporting_facts:
            supporting_facts[title] = [sent_idx]
        else:
            supporting_facts[title].append(sent_idx)
    context_ids = []
    evidence_in_context = []
    sent_in_context = []
    chunked_examples = []
    question_ids = tokenizer(x['question'], add_special_tokens=False)['input_ids']
    answer = tokenizer.decode(tokenizer(x['answer'], add_special_tokens=False)['input_ids'])
    for title, sentences in x['context']:
        for sent_idx, sent in enumerate(sentences):
            tokenized_sent = tokenizer(' '+sent.strip(), add_special_tokens=False)
            sent_ids = tokenized_sent['input_ids']
            while len(context_ids)>0 and len(context_ids)+len(sent_ids)>max_context:
                answer_loc = [-1,-1]
                chunked_context_ids = context_ids[:max_context]
                chunked_context = tokenizer.decode(chunked_context_ids)
                start = chunked_context.lower().find(answer.lower())
                if start!=-1:
                    tokenized_chunked_context = tokenizer(chunked_context, add_special_tokens=False)
                    chunked_context_ids = tokenized_chunked_context['input_ids']
                    end = start+len(answer)
                    start, end = get_tokenized_loc(tokenized_chunked_context, start, end-1)
                    answer_loc = [start, end]
                chunked_examples.append({
                    'question': x["question"],
                    "question_ids": question_ids,
                    "answer": answer,
                    "answer_loc": answer_loc,
                    "context_title": title,
                    "context": chunked_context,
                    "context_ids": chunked_context_ids,
                    "fact_in_context": [[title, s[0]] for s in sent_in_context],
                    "is_fact": True if evidence_in_context else False,
                    "answer_in_context": True if answer_loc[0]!=-1 else False,
                    'qid': x['_id'],
                    'cid': len(chunked_examples),
                    'level': x['level'],
                    'type': x['type'],
                    'all_facts': all_facts
                })
                context_ids = context_ids[stride:]
                new_evidence_in_context = []
                for start, end in evidence_in_context:
                    start -= stride
                    end -= stride
                    if end>0:
                        new_evidence_in_context.append([start if start>=0 else 0,end])
                evidence_in_context = new_evidence_in_context
                new_sent_in_context = []
                for idx, start, end in sent_in_context:
                    start -= stride
                    end -= stride
                    if end>0:
                        new_sent_in_context.append([idx, start if start>=0 else 0,end])
                sent_in_context = new_sent_in_context
            if sent_idx in supporting_facts.get(title, []):
                evidence_in_context.append([len(context_ids), len(context_ids)+len(sent_ids)])
            sent_in_context.append([sent_idx, len(context_ids), len(context_ids)+len(sent_ids)])
            context_ids += sent_ids
        while context_ids:
            answer_loc = [-1,-1]
            chunked_context_ids = context_ids[:max_context]
            chunked_context = tokenizer.decode(chunked_context_ids)
            start = chunked_context.lower().find(answer.lower())
            if start!=-1:
                tokenized_chunked_context = tokenizer(chunked_context, add_special_tokens=False)
                chunked_context_ids = tokenized_chunked_context['input_ids']
                end = start+len(answer)
                start, end = get_tokenized_loc(tokenized_chunked_context, start, end-1)
                if start is not None and end is not None:
                    answer_loc = [start, end]
            chunked_examples.append({
                'question': x["question"],
                "question_ids": question_ids,
                "answer": answer,
                "answer_loc": answer_loc,
                "context_title": title,
                "context": chunked_context,
                "context_ids": chunked_context_ids,
                "fact_in_context": [[title, s[0]] for s in sent_in_context],
                "is_fact": True if evidence_in_context else False,
                "answer_in_context": True if answer_loc[0]!=-1 else False,
                'qid': x['_id'],
                'cid': len(chunked_examples),
                'level': x['level'],
                'type': x['type'],
                'all_facts': all_facts
            })
            if len(context_ids) <= max_context:
                break
            context_ids = context_ids[stride:]
            new_evidence_in_context = []
            for start, end in evidence_in_context:
                start -= stride
                end -= stride
                if end>0:
                    new_evidence_in_context.append([start if start>=0 else 0,end])
            evidence_in_context = new_evidence_in_context
            new_sent_in_context = []
            for idx, start, end in sent_in_context:
                start -= stride
                end -= stride
                if end>0:
                    new_sent_in_context.append([idx, start if start>=0 else 0,end])
            sent_in_context = new_sent_in_context
        context_ids = []
        evidence_in_context = []
        sent_in_context = []
    return chunked_examples

class HotpotQADatasetForPS(Dataset):
    def __init__(self, datadir, dataset, split, tokenizer='bert-base-uncased', sample_num = -1, overwrite = False, neg_ratio=-1) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['[QUESTION]']+(['[SEP]'] if self.tokenizer.sep_token!='[SEP]' else [])+[f'[{p}={i}]' for i in range(50) for p in ['P','Tab','List']]})
        preprocessed_file_name = os.path.join(datadir, split, f'processed_{dataset}_{tokenizer.split("/")[-1]}_PS.jsonl')
        logger.info(f"***** loading {split} data for {dataset}")
        if os.path.exists(preprocessed_file_name) and not overwrite:
            data = []
            with open(preprocessed_file_name, 'r') as f_in:
                for line in f_in:
                    data.append(json.loads(line))
            logger.info(f"load {len(data)} preprocessed examples from {preprocessed_file_name}")
        else:
            logger.info(f"cannot find preprocessed file or choose to overwrite, create new one in {preprocessed_file_name}")
            with open(os.path.join(datadir, split, f'{dataset}.json'), 'r') as f_in,\
                open(preprocessed_file_name, 'w') as f_out:
                    raw = json.load(f_in)
                    with Pool(8) as pool:
                        data = pool.map(partial(HotpotQA_preprocess, tokenizer=self.tokenizer, is_train=split=="train"), raw)
                    for chunked_examples in data:
                        f_out.write('%s\n'%json.dumps(chunked_examples))
            logger.info(f"processed {len(data)} examples and saved in {preprocessed_file_name}")
        if sample_num < 1 and sample_num > 0:
            sample_num = int(len(data)*sample_num)
        else:
            sample_num = int(sample_num)
        if sample_num != -1 and sample_num<len(data):
            data = random.sample(data, sample_num)
            logger.info(f"random sample {sample_num} examples")
        else:
            data = data
            logger.info(f"use the full dataset")
        pos_data = []
        neg_data = []
        for chunked_examples in data:
            for chunk in chunked_examples:
                if chunk['is_fact']:
                    pos_data.append(chunk)
                else:
                    neg_data.append(chunk)
        
        logger.info(f"{len(pos_data)+len(neg_data)} after chunking the context")
        if neg_ratio!=-1:
            logger.info(f"resample based on negative ratio: {neg_ratio}")
            pos_num = len(pos_data)
            neg_num = int(pos_num*neg_ratio)
            self.data = pos_data + random.sample(neg_data, neg_num)
            logger.info(f"{len(self.data)} examples after resample")
        else:
            self.data = pos_data + neg_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

class HotposQAForPS_collate:
    def __init__(self, tokenizer, use_token_type=True):
        self.q_id = tokenizer.convert_tokens_to_ids("[QUESTION]")
        self.sep_id = tokenizer.sep_token_id
        self.cls_id = tokenizer.cls_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
        self.use_token_type = use_token_type
    def __call__(self, batch_samples):
        batch_size = len(batch_samples)
        batched_input_ids = np.full((batch_size, 400), self.pad_id)
        batched_input_ids[:, 0] = self.cls_id
        batched_input_type_ids = np.full((batch_size, 400), 0)
        batched_attention_mask = np.zeros((batch_size, 400))
        batched_q_loc = np.zeros((batch_size, 1), dtype=int)
        batched_target = np.zeros((batch_size,), dtype=float)
        batched_meta = []
        for batch_id, sample in enumerate(batch_samples):
            question_ids = sample['question_ids']
            context_ids = sample['context_ids']
            batched_input_ids[batch_id, 1:1+len(question_ids)] = question_ids
            batched_input_ids[batch_id, 1+len(question_ids)] = self.q_id
            batched_input_ids[batch_id, 2+len(question_ids)] = self.sep_id
            batched_input_ids[batch_id, 3+len(question_ids):3+len(question_ids)+len(context_ids)] = context_ids
            batched_input_ids[batch_id, 3+len(question_ids)+len(context_ids)] = self.sep_id
            if self.use_token_type:
                batched_input_type_ids[batch_id, 3+len(question_ids):4+len(question_ids)+len(context_ids)] = 1
            batched_attention_mask[batch_id, :4+len(question_ids)+len(context_ids)] = 1
            batched_q_loc[batch_id, 0] = 1+len(question_ids)
            batched_target[batch_id] = 1 if sample['is_fact'] else 0
            batched_meta.append({
                'is_fact': sample['is_fact'],
                'answer_in_context': sample['answer_in_context'],
                'fact_in_context': sample['fact_in_context'],
                'all_facts': sample['all_facts'],
                'qid': sample['qid'],
                'cid': sample['cid']
            })
        return {
            'input_ids': torch.as_tensor(batched_input_ids),
            'token_type_ids': torch.as_tensor(batched_input_type_ids),
            'attention_mask': torch.as_tensor(batched_attention_mask),
            'q_loc': torch.as_tensor(batched_q_loc),
            'target': torch.as_tensor(batched_target),
            'meta': batched_meta,
        }

class HotpotQADatasetForQA(Dataset):
    def __init__(self, datadir, dataset, split, tokenizer='bert-base-uncased', sample_num = -1, overwrite = False, neg_ratio=-1, prediction_prefix="") -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['[QUESTION]']+(['[SEP]'] if self.tokenizer.sep_token!='[SEP]' else [])+[f'[{p}={i}]' for i in range(50) for p in ['P','Tab','List']]})
        preprocessed_file_name = os.path.join(datadir, split, f'processed_{dataset}_{tokenizer.split("/")[-1]}_PS.jsonl')
        logger.info(f"***** loading {split} data for {dataset}")
        if os.path.exists(preprocessed_file_name) and not overwrite:
            data = []
            with open(preprocessed_file_name, 'r') as f_in:
                for line in f_in:
                    data.append(json.loads(line))
            logger.info(f"load {len(data)} preprocessed examples from {preprocessed_file_name}")
        else:
            logger.info(f"cannot find preprocessed file or choose to overwrite, create new one in {preprocessed_file_name}")
            with open(os.path.join(datadir, split, f'{dataset}.json'), 'r') as f_in,\
                open(preprocessed_file_name, 'w') as f_out:
                    raw = json.load(f_in)
                    with Pool(8) as pool:
                        data = pool.map(partial(HotpotQA_preprocess, tokenizer=self.tokenizer, is_train=split=="train"), raw)
                    for chunked_examples in data:
                        f_out.write('%s\n'%json.dumps(chunked_examples))
            logger.info(f"processed {len(data)} examples and saved in {preprocessed_file_name}")
        if prediction_prefix=="oracle":
            use_oracle = True
        else:
            use_oracle = False
        prediction_path = os.path.join(datadir, split, f'{dataset}_PS_{prediction_prefix}_predictions.json')
        if os.path.exists(prediction_path) and not use_oracle:
            logger.info(f"load predictions from {prediction_path}")
            with open(prediction_path, 'r') as f:
                predictions = json.load(f)
                processed_predictions = {}
                for qid in predictions:
                    for idx, (cid, score) in enumerate(predictions[qid][:3]):
                        processed_predictions[(qid,cid)] = [idx, score]
        else:
            processed_predictions = None
        if sample_num < 1 and sample_num > 0:
            sample_num = int(len(data)*sample_num)
        else:
            sample_num = int(sample_num)
        if sample_num != -1 and sample_num<len(data):
            data = random.sample(data, sample_num)
            logger.info(f"random sample {sample_num} examples")
        else:
            data = data
            logger.info(f"use the full dataset")
        pos_data = []
        neg_data = []
        for chunked_examples in data:
            fact_examples = {}
            for chunk in chunked_examples:
                add = False
                if processed_predictions is not None:
                    if (chunk['qid'], chunk['cid']) in processed_predictions:
                        add = True
                        rank, score = processed_predictions[(chunk['qid'], chunk['cid'])]
                        chunk.update({'rank': rank, 'score': score})
                if (split=='train' or processed_predictions is None or use_oracle) and chunk['is_fact']:
                    add = True     
                if add:
                    if chunk['context_title'] not in fact_examples:
                        fact_examples[chunk['context_title']] = []
                    fact_examples[chunk['context_title']].append(chunk)
            fact_examples = list(fact_examples.items())
            if len(fact_examples)==1:
                title1,fact_chunks_1 = fact_examples[0]
                for chunk1 in fact_chunks_1:
                    concatenated_context_chunk_ids = self.tokenizer.convert_tokens_to_ids(['yes', 'no'])+chunk1['context_ids']
                    if chunk1['answer'] == 'yes':
                        answer_span = [0,0]
                    elif chunk1['answer'] == 'no':
                        answer_span = [1,1]
                    elif chunk1['answer_in_context']:
                        answer_span = [chunk1['answer_loc'][0]+2, chunk1['answer_loc'][1]+2]
                    else:
                        answer_span = None
                    if answer_span is not None:
                        pos_data.append({
                            'qid': chunk1['qid'],
                            'question_ids': chunk1['question_ids'],
                            'context_chunk_ids': concatenated_context_chunk_ids,
                            'answer_span': answer_span,
                            'answers': [chunk1['answer']]
                        })
                    else:
                        neg_data.append({
                            'qid': chunk1['qid'],
                            'question_ids': chunk1['question_ids'],
                            'context_chunk_ids': concatenated_context_chunk_ids,
                            'answer_span': answer_span,
                            'answers': [chunk1['answer']]
                        })
            else:
                for i, (title1,fact_chunks_1) in enumerate(fact_examples):
                    for chunk1 in fact_chunks_1:
                        for title2,fact_chunks_2 in fact_examples[i+1:]:
                            for chunk2 in fact_chunks_2:
                                concatenated_context_chunk_ids = self.tokenizer.convert_tokens_to_ids(['yes', 'no'])\
                                    +chunk1['context_ids']+[self.tokenizer.sep_token_id]+chunk2['context_ids']
                                
                                if chunk1['answer'] == 'yes':
                                    answer_span = [0,0]
                                elif chunk1['answer'] == 'no':
                                    answer_span = [1,1]
                                elif chunk1['answer_in_context']:
                                    answer_span = [chunk1['answer_loc'][0]+2, chunk1['answer_loc'][1]+2]
                                elif chunk2['answer_in_context']:
                                    answer_span = [chunk2['answer_loc'][0]+len(chunk1['context_ids'])+3, chunk2['answer_loc'][1]+len(chunk1['context_ids'])+3]
                                else:
                                    answer_span = None
                                if answer_span is not None:
                                    pos_data.append({
                                        'qid': chunk1['qid'],
                                        'question_ids': chunk1['question_ids'],
                                        'context_chunk_ids': concatenated_context_chunk_ids,
                                        'answer_span': answer_span,
                                        'answers': [chunk1['answer']]
                                    })
                                else:
                                    neg_data.append({
                                        'qid': chunk1['qid'],
                                        'question_ids': chunk1['question_ids'],
                                        'context_chunk_ids': concatenated_context_chunk_ids,
                                        'answer_span': answer_span,
                                        'answers': [chunk1['answer']]
                                    })

        
        logger.info(f"{len(pos_data)+len(neg_data)} after chunking the context")
        if neg_ratio!=-1:
            logger.info(f"resample based on negative ratio: {neg_ratio}")
            pos_num = len(pos_data)
            neg_num = int(pos_num*neg_ratio)
            self.data = pos_data + random.sample(neg_data, neg_num)
            logger.info(f"{len(self.data)} examples after resample")
        else:
            self.data = pos_data + neg_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

class TAPAS_processor:
    def __init__(self, tokenizer, max_cell_length, max_input_length) -> None:
        self.tokenizer = tokenizer
        self.max_cell_length = max_cell_length
        self.max_input_length = max_input_length
        self.q_id = tokenizer.convert_tokens_to_ids("[QUESTION]")
        self.sep_id = tokenizer.sep_token_id
        self.cls_id = tokenizer.cls_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
    def __call__(self, table, query, answer, select_cell=False, query_values=[]):
        answer_cell_index = answer['index']
        answer_cell_span = answer['span']
        tokenized_query = self.tokenizer(query, add_special_tokens=False)
        query_ids = tokenized_query['input_ids']
        table_data = table['data']
        table_values = table['values']
        table_index = table['index']
        table_value_ranks = table['value_ranks']
        table_value_inv_ranks = table['value_inv_ranks']
        table_ids = []
        table_column_ids = []
        table_row_ids = []
        table_column_ranks = []
        table_column_inv_ranks = []
        table_value_relations = {}
        target_start = -1
        target_end = -1
        total_table_length = 0
        max_table_length = self.max_input_length-len(query_ids)-3
        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                if not cell.strip():
                    continue
                tokenized_cell = self.tokenizer(' '+cell if cell!='[EMPTY]' else cell, add_special_tokens=False)
                cell_ids = tokenized_cell['input_ids'][:self.max_cell_length]
                cell_length = len(cell_ids)
                total_table_length += cell_length
                if total_table_length>max_table_length:
                    break
                table_ids += cell_ids

                table_column_ids += [j+1]*cell_length # column 0 is for text only part
                table_row_ids += [i]*cell_length
                
                table_column_ranks += [table_value_ranks[i][j]]*cell_length
                table_column_inv_ranks += [table_value_inv_ranks[i][j]]*cell_length
                def cmp_date(date0, date1):
                    for a,b in zip(date0, date1):
                        if a is None and b is None:
                            continue
                        elif (a is None and b is not None) or (b is None and a is not None):
                            return None
                        elif a<b:
                            return 1
                        elif a>b:
                            return 2
                    return 0
                if table_values[i][j] and query_values:
                    for t_value in table_values[i][j]:
                        for q_value in query_values:
                            if t_value[0] == q_value[0]:
                                start, end = t_value[1]
                                if t_value[0]=='number':
                                    relation = 0 if q_value[2]==t_value[2] else (1 if q_value[2]<t_value[2] else 2)
                                else:
                                    relation = cmp_date(q_value[2],t_value[2])
                                    if relation is None:
                                        continue
                                start, end = get_tokenized_loc(tokenized_cell, start+1, end) # cell is prefix with extra ' '
                                if start is not None and end is not None and end<self.max_cell_length:
                                    for v_loc in range(start, end+1):
                                        v_loc = total_table_length-cell_length+v_loc
                                        if v_loc not in table_value_relations:
                                            table_value_relations[v_loc] = {relation}
                                        else:
                                            table_value_relations[v_loc].add(relation)
                if i==answer_cell_index[0] and j==answer_cell_index[1]:
                    if select_cell:
                        target_start = total_table_length-cell_length
                        target_end = total_table_length
                    else:
                        start, end = get_tokenized_loc(tokenized_cell, answer_cell_span[0]+1, answer_cell_span[1])
                        if start is not None and end is not None and end<self.max_cell_length:
                            target_start = total_table_length-cell_length+start
                            target_end = total_table_length-cell_length+end
            if total_table_length>max_table_length:
                break
        # formulate the input
        # use concat
        # [CLS] query [SEP] evidence_0 [SEP] evidence_1 [SEP]
        final_input_ids = torch.full((self.max_input_length,), self.pad_id)
        final_input_column_ids = torch.zeros(self.max_input_length, dtype=int)
        final_input_row_ids = torch.zeros(self.max_input_length, dtype=int)
        final_input_column_ranks = torch.zeros(self.max_input_length, dtype=int)
        final_input_column_inv_ranks = torch.zeros(self.max_input_length, dtype=int)
        final_input_segment_ids = torch.zeros(self.max_input_length, dtype=int)
        final_input_value_relations = torch.zeros(self.max_input_length, dtype=int)
        final_attention_mask = torch.zeros(self.max_input_length)
        final_q_loc = torch.zeros(1, dtype=int)
        final_target_start = torch.zeros(1, dtype=int)
        final_target_end = torch.zeros(1, dtype=int)
        final_target_mask = torch.zeros(self.max_input_length)

        final_target_mask[0] = 1 #[CLS] is the target for nonanswrable
        final_input_ids[0] = self.cls_id
        final_input_ids[1:1+len(query_ids)] = torch.as_tensor(query_ids)
        final_input_ids[len(query_ids)+1] = self.q_id
        final_q_loc[0] = len(query_ids)+1
        final_input_ids[len(query_ids)+2] = self.sep_id
        p = len(query_ids)+3

        for v_loc, rs in table_value_relations.items():
            final_input_value_relations[p+v_loc] = sum([2**r for r in rs])
        final_input_ids[p:p+len(table_ids)] = torch.as_tensor(table_ids)
        final_target_mask[p:p+len(table_ids)] = 1
        final_input_column_ids[p:p+len(table_ids)] = torch.as_tensor(table_column_ids)
        final_input_row_ids[p:p+len(table_ids)] = torch.as_tensor(table_row_ids)
        final_input_column_ranks[p:p+len(table_ids)] = torch.as_tensor(table_column_ranks)
        final_input_column_inv_ranks[p:p+len(table_ids)] = torch.as_tensor(table_column_inv_ranks)
        final_input_segment_ids[p:p+len(table_ids)] = 1
        final_target_start[0] = 0 if target_start==-1 else p+target_start
        final_target_end[0] = 0 if target_end==-1 else p+target_end
        p += len(table_ids)+1

        final_attention_mask[:p] = 1
        
        final_input_type_ids = torch.stack([
            final_input_segment_ids,
            final_input_column_ids,
            final_input_row_ids,
            torch.zeros(self.max_input_length, dtype=int), #prev_labels,
            final_input_column_ranks.clamp(0,128),
            final_input_column_inv_ranks.clamp(0,128),
            final_input_value_relations #numeric_relations
        ], dim=1)
        return {
            'input_ids': final_input_ids,
            'token_type_ids': final_input_type_ids,
            'attention_mask': final_attention_mask,
            'q_loc': final_q_loc,
            'target_start': final_target_start,
            'target_end': final_target_end,
            'target_mask': final_target_mask,
        }

class TAPASQA_collate(object):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
    def __call__(self, batch_samples):
        # pdb.set_trace()
        processed_batch_samples = default_collate([
            {k:v for k,v in sample.items() if k in\
                {'input_ids', 'token_type_ids', 'attention_mask', 'q_loc', 'target_start', 'target_end', 'target_mask', 'start_end_mask'}}\
                 for sample in batch_samples])
        processed_batch_samples['answers'] = [sample['all_answers'] for sample in batch_samples]
        processed_batch_samples['qid'] = [sample['qid'] for sample in batch_samples]
        return processed_batch_samples

class TAPASDatasetForQA(Dataset):
    def __init__(self, datadir, dataset, split, tokenizer='roberta', sample_num = -1, neg_ratio=-1):
        super().__init__()
        if 'tapas' in tokenizer.lower():
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['[QUESTION]', '[EMPTY]', '<Li>']+(['[SEP]'] if self.tokenizer.sep_token!='[SEP]' else [])+[f'[{p}={i}]' for i in range(50) for p in ['P','Tab','List']]})
        self.processor = TAPAS_processor(self.tokenizer, 50, 512)
        # self.tokenizer_v0 = TableTokenizer.from_pretrained('google/tapas-base', update_answer_coordinates=True, additional_special_tokens=['[EMPTY]'])
        # self.tokenizer_v0.add_special_tokens({'additional_special_tokens':['[EMPTY]', '[QUESTION]', '<Li>']+(['[SEP]'] if self.tokenizer.sep_token!='[SEP]' else [])+[f'[{p}={i}]' for i in range(50) for p in ['P','Tab','List']]})
        with open(os.path.join(datadir, split, f'{dataset}.json'), 'r') as f_in:
            data = json.load(f_in)
            valid_ids = {item['qid'] for item in data}
            logger.info(f"load {len(valid_ids)} examples, {len(data)} after chunking")
            if sample_num < 1 and sample_num > 0:
                sample_num = int(len(valid_ids)*sample_num)
            else:
                sample_num = int(sample_num)
            if sample_num != -1 and sample_num<len(valid_ids):
                sampled_ids = set(random.sample(valid_ids, sample_num))
                data = [item for item in data if item['qid'] in sampled_ids]
                logger.info(f"random sample {sample_num} examples")
            else:
                data = data
                logger.info(f"use the full dataset")
            if neg_ratio!=-1 and split=='train':
                logger.info(f"resample based on negative ratio: {neg_ratio}")
                pos_data = []
                neg_data = []
                for item in data:
                    if item['answer']['text']=='':
                        neg_data.append(item)
                    else:
                        pos_data.append(item)
                pos_num = len(pos_data)
                neg_num = int(pos_num*neg_ratio)
                self.data = pos_data + random.sample(neg_data, neg_num)
                logger.info(f"{len(self.data)} examples after resample")
            else:
                self.data = data
        self.cache_data = None
        # if split!='train':
        #     cache_file_name = os.path.join(datadir, split, f'processed_{dataset}.pkl')
        #     if os.path.exists(cache_file_name):
        #         with open(cache_file_name, 'rb') as f:
        #             self.cache_data = pickle.load(f)
        #     else:
        #         cache_data = []
        #         logger.info("cache processing data for evaluation")
        #         for idx in tqdm(range(len(self.data))):
        #             processed_item = self[idx]
        #             cache_data.append(processed_item)
        #         with open(cache_file_name, 'wb') as f:
        #             pickle.dump(cache_data, f)
        #         self.cache_data = cache_data

    def __getitem__(self, idx):
        if self.cache_data is not None:
            return self.cache_data[idx]
        item = self.data[idx]
        # table = pd.DataFrame(item['table']['data'][1:], columns=item['table']['data'][0]).astype(str)
        # encoding_v0 = self.tokenizer_v0(table=table,
        #                     queries=f'[Tab={min([49, item["table"]["idx"]])}] '+item['question']+f' [QUESTION]',
        #                     answer_coordinates=[item['answer']['index']],
        #                     answer_text=[item['answer']['text']],
        #                     truncation=True,
        #                     padding="max_length",
        #                     return_tensors="pt"                        
        # )
        # def get_first_answer_span(labels):
        #     start, end = -1, -1
        #     for i, is_answer in enumerate(labels):
        #         if start==-1 and is_answer==1:
        #             start = i
        #         if start!=-1 and is_answer==1:
        #             end = i
        #         if end!=-1 and is_answer==0:
        #             break
        #     return (start, end)
        # target_start, target_end = get_first_answer_span(encoding_v0['labels'])
        encoding = self.processor(table=item['table'],
                            query=f'[Tab={min([49, item["table"]["idx"]])}] '+item['question'],
                            answer=item['answer'],
                            query_values=item['question_values']
        )
        # if (encoding['input_ids']!=encoding_v0['input_ids']).sum().item()!=0:
        #     print(self.tokenizer.decode(encoding['input_ids']).encode())
        #     print(self.tokenizer_v0.decode(encoding_v0['input_ids']).encode())
        #     pdb.set_trace()
        # if (encoding['token_type_ids'][:,:6]!=encoding_v0['token_type_ids'][:,:6]).sum().item()!=0:
        #     pdb.set_trace()
        # if target_start!=encoding['target_start'][0].item() or target_end!=encoding['target_end'][0].item():
        #     pdb.set_trace()
        column_ids = encoding['token_type_ids'][:,1]
        row_ids = encoding['token_type_ids'][:,2]
        seq_length = column_ids.shape[0]
        start_end_mask = torch.zeros((seq_length, seq_length))
        cell_id_map = {}
        for i in range(seq_length):
            if column_ids[i]==0:
                continue
            cell_id = (row_ids[i].item(),column_ids[i].item())
            if cell_id in cell_id_map:
                cell_id_map[cell_id][1] += 1
            else:
                cell_id_map[cell_id] = [i, i+1]
        for _, (cell_start, cell_end) in cell_id_map.items():
            start_end_mask[cell_start:cell_end, cell_start:cell_end] = 1
        start_end_mask[0,:] = 1
        start_end_mask[:,0] = 1
        encoding['qid'] = item['qid']
        encoding['all_answers'] = [self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))) for answer in item['all_answers']]
        encoding['start_end_mask'] = start_end_mask
        return encoding
    
    def __len__(self):
        return len(self.data)

class TAPASDatasetForQA_v0(Dataset):
    def __init__(self, datadir, dataset, split, tokenizer='google/tapas-base', sample_num = -1, neg_ratio=-1):
        super().__init__()
        self.tokenizer = TableTokenizer.from_pretrained('google/tapas-base', update_answer_coordinates=True, additional_special_tokens=['[EMPTY]'])
        self.tokenizer.add_special_tokens({'additional_special_tokens':['[EMPTY]', '[QUESTION]']+(['[SEP]'] if self.tokenizer.sep_token!='[SEP]' else [])+[f'[{p}={i}]' for i in range(50) for p in ['P','Tab','List']]})
        # self.tokenizer_v0 = BertTokenizerFast.from_pretrained("./output/models/TAPAS")
        # self.tokenizer_v0.add_special_tokens({'additional_special_tokens':['[QUESTION]']+(['[SEP]'] if self.tokenizer.sep_token!='[SEP]' else [])+[f'[{p}={i}]' for i in range(50) for p in ['P','Tab','List']]})
        # self.processor = TAPAS_processor(self.tokenizer_v0, 50, 512)
        with open(os.path.join(datadir, split, f'{dataset}.json'), 'r') as f_in:
            data = json.load(f_in)
            logger.info(f"load {len(data)} examples")
            if neg_ratio!=-1 and split=='train':
                logger.info(f"resample based on negative ratio: {neg_ratio}")
                pos_data = []
                neg_data = []
                for item in data:
                    if item['answer']['text']=='':
                        neg_data.append(item)
                    else:
                        pos_data.append(item)
                pos_num = len(pos_data)
                neg_num = int(pos_num*neg_ratio)
                self.data = pos_data + random.sample(neg_data, neg_num)
                logger.info(f"{len(self.data)} examples after resample")
            else:
                self.data = data
        self.cache_data = None
        if split!='train':
            cache_file_name = os.path.join(datadir, split, f'processed_{dataset}.pkl')
            if os.path.exists(cache_file_name):
                with open(cache_file_name, 'rb') as f:
                    self.cache_data = pickle.load(f)
            else:
                cache_data = []
                logger.info("cache processing data for evaluation")
                for idx in tqdm(range(len(self.data))):
                    processed_item = self[idx]
                    cache_data.append(processed_item)
                with open(cache_file_name, 'wb') as f:
                    pickle.dump(cache_data, f)
                self.cache_data = cache_data

    def __getitem__(self, idx):
        if self.cache_data is not None:
            return self.cache_data[idx]
        item = self.data[idx]
        table = pd.DataFrame(item['table']['data'][1:], columns=item['table']['data'][0]).astype(str)
        encoding = self.tokenizer(table=table,
                            queries=item['question']+f' [QUESTION] [Tab={min([49, item["table"]["idx"]])}]',
                            answer_coordinates=[item['answer']['index']],
                            answer_text=[item['answer']['text']],
                            truncation=True,
                            padding="max_length",
                            return_tensors="pt"                        
        )
        # encoding_v0 = self.processor(table=item['table'],
        #                     query=item['question']+f' [Tab={min([49, item["table"]["idx"]])}] ',
        #                     answer=item['answer'],
        # )
        def get_first_answer_span(labels):
            start, end = -1, -1
            for i, is_answer in enumerate(labels):
                if start==-1 and is_answer==1:
                    start = i
                if start!=-1 and is_answer==1:
                    end = i
                if end!=-1 and is_answer==0:
                    break
            return (start, end)
        target_start, target_end = get_first_answer_span(encoding['labels'])
        column_ids = encoding['token_type_ids'][:,1]
        row_ids = encoding['token_type_ids'][:,2]
        seq_length = column_ids.shape[0]
        start_end_mask = torch.zeros((seq_length, seq_length))
        cell_id_map = {}
        for i in range(seq_length):
            if column_ids[i]==0:
                continue
            cell_id = (row_ids[i].item(),column_ids[i].item())
            if cell_id in cell_id_map:
                cell_id_map[cell_id][1] += 1
            else:
                cell_id_map[cell_id] = [i, i+1]
        for _, (cell_start, cell_end) in cell_id_map.items():
            start_end_mask[cell_start:cell_end, cell_start:cell_end] = 1
        start_end_mask[0,:] = 1
        start_end_mask[:,0] = 1
        encoding['target_start'] = torch.as_tensor([target_start])
        encoding['target_end'] = torch.as_tensor([target_end])
        encoding['qid'] = item['qid']
        encoding['all_answers'] = [self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(answer))) for answer in item['all_answers']]
        encoding['start_end_mask'] = start_end_mask
        return encoding
    
    def __len__(self):
        return len(self.data)

class TAPASCS_processor:
    def __init__(self, tokenizer, max_query_length, max_cell_length, max_linked_context_length, max_input_length) -> None:
        self.tokenizer = tokenizer
        self.max_query_length = max_query_length
        self.max_cell_length = max_cell_length
        self.max_linked_context_length = max_linked_context_length
        self.max_input_length = max_input_length
        self.q_id = tokenizer.convert_tokens_to_ids("[QUESTION]")
        self.sep_id = tokenizer.sep_token_id
        self.cls_id = tokenizer.cls_token_id
        self.mask_id = tokenizer.mask_token_id
        self.pad_id = tokenizer.pad_token_id
    def __call__(self, table, query, linked_cells=[], query_values=[]):
        tokenized_query = self.tokenizer(query, add_special_tokens=False)
        query_ids = tokenized_query['input_ids'][:self.max_query_length]
        table_data = table['data']
        table_values = table['values']
        table_index = table['index']
        table_value_ranks = table['value_ranks']
        table_value_inv_ranks = table['value_inv_ranks']

        linked_cells = {tuple(cell) for cell in linked_cells}
        table_ids = []
        table_column_ids = []
        table_row_ids = []
        table_column_ranks = []
        table_column_inv_ranks = []
        table_prev_labels = []
        table_value_relations = {}
        total_table_length = 0
        max_table_length = self.max_input_length-len(query_ids)-3
        for i, row in enumerate(table_data):
            for j, cell in enumerate(row):
                if not cell.strip():
                    continue
                tokenized_cell = self.tokenizer(' '+cell if cell!='[EMPTY]' else cell, add_special_tokens=False)
                if (table_index[i][j][0], table_index[i][j][1]) in linked_cells:
                    cell_ids = tokenized_cell['input_ids'][:(self.max_cell_length+self.max_linked_context_length)]
                    table_prev_label = 1
                else:
                    cell_ids = tokenized_cell['input_ids'][:self.max_cell_length]
                    table_prev_label = 0
                cell_length = len(cell_ids)

                
                total_table_length += cell_length
                if total_table_length>max_table_length:
                    break
                table_ids += cell_ids
                table_prev_labels += [table_prev_label]*cell_length

                table_column_ids += [table_index[i][j][1]+1]*cell_length # column 0 is for text only part
                table_row_ids += [table_index[i][j][0]]*cell_length
                
                table_column_ranks += [table_value_ranks[i][j]]*cell_length
                table_column_inv_ranks += [table_value_inv_ranks[i][j]]*cell_length
                def cmp_date(date0, date1):
                    for a,b in zip(date0, date1):
                        if a is None and b is None:
                            continue
                        elif (a is None and b is not None) or (b is None and a is not None):
                            return None
                        elif a<b:
                            return 1
                        elif a>b:
                            return 2
                    return 0
                if table_values[i][j] and query_values:
                    for t_value in table_values[i][j]:
                        for q_value in query_values:
                            if t_value[0] == q_value[0]:
                                start, end = t_value[1]
                                if t_value[0]=='number':
                                    relation = 0 if q_value[2]==t_value[2] else (1 if q_value[2]<t_value[2] else 2)
                                else:
                                    relation = cmp_date(q_value[2],t_value[2])
                                    if relation is None:
                                        continue
                                start, end = get_tokenized_loc(tokenized_cell, start+1, end) # cell is prefix with extra ' '
                                if start is not None and end is not None and end<cell_length:
                                    for v_loc in range(start, end+1):
                                        v_loc = total_table_length-cell_length+v_loc
                                        if v_loc not in table_value_relations:
                                            table_value_relations[v_loc] = {relation}
                                        else:
                                            table_value_relations[v_loc].add(relation)
            if total_table_length>max_table_length:
                break
        # formulate the input
        # use concat
        # [CLS] query [SEP] evidence_0 [SEP] evidence_1 [SEP]
        final_input_ids = torch.full((self.max_input_length,), self.pad_id)
        final_input_column_ids = torch.zeros(self.max_input_length, dtype=int)
        final_input_row_ids = torch.zeros(self.max_input_length, dtype=int)
        final_input_column_ranks = torch.zeros(self.max_input_length, dtype=int)
        final_input_column_inv_ranks = torch.zeros(self.max_input_length, dtype=int)
        final_input_segment_ids = torch.zeros(self.max_input_length, dtype=int)
        final_input_value_relations = torch.zeros(self.max_input_length, dtype=int)
        final_input_prev_labels = torch.zeros(self.max_input_length, dtype=int)
        final_attention_mask = torch.zeros(self.max_input_length)
        final_q_loc = torch.zeros(1, dtype=int)
        final_target_mask = torch.zeros(self.max_input_length)

        final_target_mask[0] = 1 #[CLS] is the target for nonanswrable
        final_input_ids[0] = self.cls_id
        final_input_ids[1:1+len(query_ids)] = torch.as_tensor(query_ids)
        final_input_ids[len(query_ids)+1] = self.q_id
        final_q_loc[0] = len(query_ids)+1
        final_input_ids[len(query_ids)+2] = self.sep_id
        p = len(query_ids)+3

        for v_loc, rs in table_value_relations.items():
            final_input_value_relations[p+v_loc] = sum([2**r for r in rs])
        final_input_ids[p:p+len(table_ids)] = torch.as_tensor(table_ids)
        final_target_mask[p:p+len(table_ids)] = 1
        final_input_column_ids[p:p+len(table_ids)] = torch.as_tensor(table_column_ids)
        final_input_row_ids[p:p+len(table_ids)] = torch.as_tensor(table_row_ids)
        final_input_column_ranks[p:p+len(table_ids)] = torch.as_tensor(table_column_ranks)
        final_input_column_inv_ranks[p:p+len(table_ids)] = torch.as_tensor(table_column_inv_ranks)
        final_input_prev_labels[p:p+len(table_ids)] = torch.as_tensor(table_prev_labels)
        final_input_segment_ids[p:p+len(table_ids)] = 1
        p += len(table_ids)+1

        final_attention_mask[:p] = 1
        
        final_input_type_ids = torch.stack([
            final_input_segment_ids,
            final_input_column_ids,
            final_input_row_ids,
            final_input_prev_labels, #linked cells,
            final_input_column_ranks.clamp(0,128),
            final_input_column_inv_ranks.clamp(0,128),
            final_input_value_relations #numeric_relations
        ], dim=1)
        return {
            'input_ids': final_input_ids,
            'token_type_ids': final_input_type_ids,
            'attention_mask': final_attention_mask,
            'q_loc': final_q_loc,
            'target_mask': final_target_mask,
        }

class TAPASCS_collate:
    def __call__(self, batch_samples, tokenizer=None):
        processed_batch_samples = default_collate([
            {k:v for k,v in sample.items() if k in\
                {'input_ids', 'token_type_ids', 'attention_mask', 'q_loc', 'target_start',\
                    'target_start_onehot', 'target_mask', 'cell_ids', 'target_row_onehot', 'target_column_onehot'}}\
                 for sample in batch_samples])
        processed_batch_samples['answers'] = [sample['all_answers'] for sample in batch_samples]
        processed_batch_samples['qid'] = [sample['qid'] for sample in batch_samples]
        return processed_batch_samples

class TAPASDatasetForCS(TAPASDatasetForQA):
    def __init__(self, datadir, dataset, split, tokenizer='roberta', sample_num = -1, neg_ratio=-1):
        super().__init__(datadir, dataset, split, tokenizer, sample_num, neg_ratio)
        self.processor = TAPASCS_processor(self.tokenizer, 256, 20, 100, 512)
    def __getitem__(self, idx):
        if self.cache_data is not None:
            return self.cache_data[idx]
        item = self.data[idx]
        encoding = self.processor(table=item['table'],
                            query=f'[Tab={min([49, item["table"]["idx"]])}] '+item['question'],
                            linked_cells=item['linked_cells'],
                            query_values=item['question_values']
        )
        all_answers = {tuple(answer) for answer in item['all_answers']}
        encoding['all_answers'] = all_answers
        answer_index = item['answer']['index']
        column_ids = encoding['token_type_ids'][:,1]
        row_ids = encoding['token_type_ids'][:,2]
        cell_ids = torch.zeros_like(column_ids)
        seq_length = column_ids.shape[0]
        target_mask = torch.zeros_like(encoding['target_mask'])
        target_mask[0] = 1
        cell_id_map = {}
        target_start = -1
        target_start_onehot = torch.zeros_like(encoding['target_mask'])
        target_row_onehot = torch.zeros(256)
        target_column_onehot = torch.zeros(256)
        for i in range(seq_length):
            cell_id = [row_ids[i].item(),column_ids[i].item()-1]
            if row_ids[i]==0:
                continue
            if target_start==-1 and cell_id[0]==answer_index[0] and cell_id[1]==answer_index[1]:
                target_start = i
            cell_id = tuple(cell_id)
            cell_ids[i] = cell_id[0]*256+cell_id[1]
            if cell_id in all_answers:
                target_start_onehot[i] = 1
                target_row_onehot[cell_id[0]] = 1
                target_column_onehot[cell_id[1]+1] = 1
            if cell_id not in cell_id_map:
                target_mask[i] = 1
                cell_id_map[cell_id] = [i, i+1]
        if answer_index[0]==-1:
            target_start = 0
            target_start_onehot[0] = 1
            target_row_onehot[0] = 1
            target_column_onehot[0] = 1
        encoding['cell_ids'] = cell_ids
        encoding['target_mask'] = target_mask
        encoding['target_start'] = torch.as_tensor([target_start])
        encoding['target_start_onehot'] = target_start_onehot
        encoding['target_row_onehot'] = target_row_onehot
        encoding['target_column_onehot'] = target_column_onehot
        encoding['qid'] = item['qid']
        
        return encoding

if __name__=="__main__":
    dataset = HybridPairDataset(tokenizer='roberta-base',\
        urls="/workspace/data/table_pairs_for_pretrain_no_tokenization/{000000..000003}.tar",mlm_probability=0.15)
    x = dataset[0]

        


            


