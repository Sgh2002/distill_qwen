#!/usr/bin/python
#****************************************************************#
# ScriptName: utils.py
# Author: @alibaba-inc.com
# Create Date: 2023-10-31 11:35
# Modify Author: @alibaba-inc.com
# Modify Date: 2024-04-07 14:47
# Function: 
#***************************************************************#

import torch
from datasets import concatenate_datasets
from datasets import Dataset


def group_by(d, col, join): 
    """from: https://github.com/huggingface/datasets/issues/3644"""
    # Get the indices of each group
    groups = {key: [] for key in d.unique(col)} 
    def create_groups_indices(key, i): 
        groups[key].append(i) 
    d.map(create_groups_indices, with_indices=True, input_columns=col) 
    # Get one dataset object per group
    groups = {key: d.select(indices) for key, indices in groups.items()} 
    # Apply join function
    groups = {
        key: dataset_group.map(join, batched=True, batch_size=len(dataset_group), remove_columns=d.column_names)
        for key, dataset_group in groups.items()
    } 
    # Return concatenation of all the joined groups
    return concatenate_datasets(groups.values())


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            torch.set_printoptions(profile="full")
            print(message, flush=True)
    else:
        torch.set_printoptions(profile="full")
        print(message, flush=True)

def preprocess_raw_groupby(tokenizer, examples, is_training, generate_cls, max_input_length):
    input_ids = []
    labels = []
    attention_masks = []
    cls_position = []

    for text, label, query in zip(examples['input'], examples['label'], examples['query']):
        text_id = tokenizer(text, padding=False).input_ids
        label_id = tokenizer(label, padding=False).input_ids
        post_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是:").input_ids
        #post_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是").input_ids
        text_id = text_id[:max_input_length - len(post_prompt_id) - len(label_id)] + post_prompt_id
        text_id = text_id[:max_input_length - len(label_id)]

        cls_index = len(text_id) - 1

        if generate_cls:
            input_id = text_id + label_id
        else:
            input_id = text_id 
        attention_mask = [1] * len(input_id)

        label_id = [-100] * len(text_id[:max_input_length - len(label_id)]) + label_id

        input_id = input_id + [tokenizer.pad_token_id] * (max_input_length - len(input_id))
        #input_id = input_id + [tokenizer.eos_token_id] * (max_input_length - len(input_id))
        label_id = label_id + [-100] * (max_input_length - len(label_id))
        attention_mask = attention_mask + [0] * (max_input_length - len(attention_mask))


        input_ids.append(input_id)
        labels.append(label_id)
        attention_masks.append(attention_mask)
        cls_position.append(cls_index)

    model_inputs = dict(
        query=examples['query'],
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_masks,
        cls_position=cls_position,
    )
    def join(batch):
    # take the batch of all the examples of a group, and return a batch with one aggregated example
    # (we could aggregate examples into several rows instead of one, if you want)
        #return {"input_ids": [batch["input_ids"]], "attention_mask": [batch["attention_mask"]], "cls_position": [batch["cls_position"]], "query": [batch["query"]]} 
        return {"input_ids": [batch["input_ids"]]} 

    '''
    print(model_inputs["input_ids"][:3])
    print('***********************')
    d = Dataset.from_dict(model_inputs)
    new_d = group_by(d, "query", join)
    #print(new_d[:5])
    print('---------------************')
    print(new_d["input_ids"][:3])
    print('---------------')
    #print(model_inputs.to_polars().groupby('query').len())
    print(new_d)
    new_d = new_d.map(lambda x: {"input_ids": [i for row in x["input_ids"] for i in row]}, batched=True)
    print('==========')
    print(new_d["input_ids"][:3])
    '''
    return model_inputs

