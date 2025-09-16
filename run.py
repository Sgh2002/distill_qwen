# Copyright 2023 The Distilling-step-by-step authors


# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import argparse
import torch

from datasets import Dataset
from functools import partial
from datasets import DatasetDict, concatenate_datasets
from datasets.distributed import split_dataset_by_node
from transformers import AutoTokenizer
from transformers import BloomTokenizerFast

from data_utils import CQADatasetLoader, SVAMPDatasetLoader, ESNLIDatasetLoader, ANLI1DatasetLoader, ASDivDatasetLoader, QTCDatasetLoader, QTC2DatasetLoader, QTCNewDatasetLoader
from metrics import compute_text_acc, compute_equation_acc, compute_metrics_text, compute_metrics_equation, compute_metrics_text_aux, compute_metrics_equation_aux, compute_metrics_cls_acc
from train_utils import train_and_evaluate
from utils import preprocess_raw_groupby

from urllib import parse
import string

torch.set_printoptions(threshold=10000)
#from datasets import set_caching_enabled
#set_caching_enabled(False)

query_d = {}

def preprocess_raw(tokenizer, examples, is_training, generate_cls):
    input_ids = []
    labels = []
    attention_masks = []
    cls_position = []

    q2d_input_ids = []
    q2d_labels = []
    q2d_attention_masks = []

    # for text, label, query in zip(examples['input'], examples['label'], examples['query']):
    for url, title, summary, body, label, query, elapsedDay in zip(examples['url'], examples['title'], examples['summary'], examples['body'], examples['label'], examples['query'], examples['elapsedDays']):
        url = url.rstrip(string.punctuation)
        hostname = parse.urlparse(url).hostname
        title = title + hostname
        # text = 'query:%s\n标题:%s\n摘要:%s\n正文:%s' % (query, title, summary, body)
        # 加入时间信号
        text = 'query:%s\n标题:%s\n距离发文时间过去:%s月\n摘要:%s\n正文:%s' % (query, title, elapsedDay, summary, body)
        # print(f"query:{query}\nurl:{url}\n距离发文时间过去:{elapsedDay}月")
        text_id = tokenizer(text, padding=False, add_special_tokens=False).input_ids
        label_id = tokenizer(label, padding=False, add_special_tokens=False).input_ids
        # print('text_id:', text_id)
        # # post_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是:", padding=False).input_ids
        # prefix_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n", padding=False, add_special_tokens=False).input_ids
        # post_prompt_id = tokenizer("答案:", padding=False).input_ids
        # # text_id = prefix_prompt_id + text_id[:args.max_input_length - len(prefix_prompt_id) - len(post_prompt_id) - len(label_id)] + post_prompt_id
        # # # text_id = text_id[:args.max_input_length - len(post_prompt_id) - len(label_id)] + post_prompt_id
        # # # text_id = text_id[:args.max_input_length - len(label_id)]
        # # no prompt 
        # text_id = text_id[:args.max_input_length - len(label_id)]
        
        # add qwen3的template，加上/no_think 关闭思考模式
        prefix_prompt_id = tokenizer("<|im_start|>user\n", padding=False, add_special_tokens=False).input_ids
        post_prompt_id = tokenizer("/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", padding=False, add_special_tokens=False).input_ids
        text_id = prefix_prompt_id + text_id[:args.max_input_length - len(prefix_prompt_id) - len(post_prompt_id) - len(label_id)] + post_prompt_id
        # print('label_id', label_id)
        # print(len(text_id))
        # print('prefix_prompt_id', prefix_prompt_id)

        cls_index = len(text_id) - 1

        if generate_cls:
            input_id = text_id + label_id
        else:
            input_id = text_id
        attention_mask = [1] * len(input_id)

        label_id = [-100] * len(text_id[:args.max_input_length - len(label_id)]) + label_id

        input_id = input_id + [tokenizer.pad_token_id] * (args.max_input_length - len(input_id))
        #input_id = input_id + [tokenizer.eos_token_id] * (args.max_input_length - len(input_id))
        label_id = label_id + [-100] * (args.max_input_length - len(label_id))
        attention_mask = attention_mask + [0] * (args.max_input_length - len(attention_mask))


        input_ids.append(input_id)
        labels.append(label_id)
        attention_masks.append(attention_mask)
        cls_position.append(cls_index)

    model_inputs = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_masks,
        cls_position=cls_position,
    )
    return model_inputs

def preprocess_raw_new(tokenizer, examples, is_training, generate_cls):
    input_ids = []
    labels = []
    attention_masks = []
    cls_position = []

    q2d_input_ids = []
    q2d_labels = []
    q2d_attention_masks = []

    # for text, label, query in zip(examples['input'], examples['label'], examples['query']):
    for url, title, summary, body, label, query in zip(examples['url'], examples['title'], examples['summary'], examples['body'], examples['label'], examples['query']):
        url = url.rstrip(string.punctuation)
        hostname = parse.urlparse(url).hostname
        title = title + hostname
        text = '标题:%s\n摘要:%s\n正文:%s' % (title, summary, body)
        text_q = '\nquery:%s\n' % query

        text_id = tokenizer(text, padding=False).input_ids
        text_q_id = tokenizer(text_q, padding=False).input_ids
        label_id = tokenizer(label, padding=False).input_ids
        post_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是:", padding=False).input_ids
        #post_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是").input_ids
        #prefix_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级: bad:不相关\nfair:部分相关,或者内容质量相对较差\ngood:相关,且内容质量较好\nperfect:相关,且权威,或者内容质量很好,或者是百科结果\nawesome:相关,且是官网结果\n").input_ids
        #post_prompt_id = tokenizer("答案:").input_ids
        #text_id = prefix_prompt_id + text_id[:args.max_input_length - len(prefix_prompt_id) - len(post_prompt_id) - len(label_id)] + post_prompt_id
        text_id = text_id[:args.max_input_length - len(post_prompt_id) - len(label_id) - len(text_q_id)] + text_q_id + post_prompt_id
        text_id = text_id[:args.max_input_length - len(label_id)]

        cls_index = len(text_id) - 1

        if generate_cls:
            input_id = text_id + label_id
        else:
            input_id = text_id 
        attention_mask = [1] * len(input_id)

        label_id = [-100] * len(text_id[:args.max_input_length - len(label_id)]) + label_id

        input_id = input_id + [tokenizer.pad_token_id] * (args.max_input_length - len(input_id))
        #input_id = input_id + [tokenizer.eos_token_id] * (args.max_input_length - len(input_id))
        label_id = label_id + [-100] * (args.max_input_length - len(label_id))
        attention_mask = attention_mask + [0] * (args.max_input_length - len(attention_mask))


        input_ids.append(input_id)
        labels.append(label_id)
        attention_masks.append(attention_mask)
        cls_position.append(cls_index)

    model_inputs = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_masks,
        cls_position=cls_position,
    )
    return model_inputs

def preprocess_raw_aggfeat(tokenizer, examples, is_training, generate_cls):
    input_ids = []
    labels = []
    attention_masks = []
    cls_position = []

    q2d_input_ids = []
    q2d_labels = []
    q2d_attention_masks = []

    # for text, label, query in zip(examples['input'], examples['label'], examples['query']):
    for url, title, summary, body, label, query, aggfeat in zip(examples['url'], examples['title'], examples['summary'], examples['body'], examples['label'], examples['query'], examples['aggfeat']):
        url = url.rstrip(string.punctuation)
        hostname = parse.urlparse(url).hostname
        title = title + hostname
        if aggfeat:
            MainBodyLength, MLAuthorityScore, PQQualityScore, SiteValueScore, ElapsedDays, TimeSensitiveScore, pv = aggfeat
        else:
            MainBodyLength, MLAuthorityScore, PQQualityScore, SiteValueScore, ElapsedDays, TimeSensitiveScore, pv = '', '', '', '', '', '', ''
        text = 'query:%s\n页面质量分:%s,页面权威分:%s,网站质量:%s,内容发布天数:%s,时效性:%s,pv:%s,正文长度:%s\n标题:%s\n摘要:%s\n正文:%s' % (query, PQQualityScore, MLAuthorityScore, SiteValueScore, ElapsedDays, TimeSensitiveScore, pv, MainBodyLength, title, summary, body)
        text_id = tokenizer(text, padding=False).input_ids
        label_id = tokenizer(label, padding=False).input_ids
        post_prompt_id = tokenizer("请根据网页的标题,正文,摘要及页面质量、权威、时效性、正文长度、pv等信息判断网页对query的满足等级, 选项: A.不满足\nB.部分满足,或者内容质量相对较差\nC.满足,且内容质量较好\nD.满足,且权威,或者内容质量很好,或者是百科结果\nE.满足,且是官网或者寻址结果\n答案是:", padding=False).input_ids
        #post_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是").input_ids
        #prefix_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级: bad:不相关\nfair:部分相关,或者内容质量相对较差\ngood:相关,且内容质量较好\nperfect:相关,且权威,或者内容质量很好,或者是百科结果\nawesome:相关,且是官网结果\n").input_ids
        #post_prompt_id = tokenizer("答案:").input_ids
        #text_id = prefix_prompt_id + text_id[:args.max_input_length - len(prefix_prompt_id) - len(post_prompt_id) - len(label_id)] + post_prompt_id
        text_id = text_id[:args.max_input_length - len(post_prompt_id) - len(label_id)] + post_prompt_id
        text_id = text_id[:args.max_input_length - len(label_id)]

        cls_index = len(text_id) - 1

        if generate_cls:
            input_id = text_id + label_id
        else:
            input_id = text_id 
        attention_mask = [1] * len(input_id)

        label_id = [-100] * len(text_id[:args.max_input_length - len(label_id)]) + label_id

        input_id = input_id + [tokenizer.pad_token_id] * (args.max_input_length - len(input_id))
        #input_id = input_id + [tokenizer.eos_token_id] * (args.max_input_length - len(input_id))
        label_id = label_id + [-100] * (args.max_input_length - len(label_id))
        attention_mask = attention_mask + [0] * (args.max_input_length - len(attention_mask))


        input_ids.append(input_id)
        labels.append(label_id)
        attention_masks.append(attention_mask)
        cls_position.append(cls_index)

    model_inputs = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_masks,
        cls_position=cls_position,
    )
    return model_inputs

def preprocess_raw_cot(tokenizer, examples, is_training, generate_cls):
    input_ids = []
    labels = []
    attention_masks = []
    cls_position = []

    rationale_input_ids = []
    rationale_labels = []
    rationale_attention_masks = []
    rationale_cls_position = []
    pre_answer_position = []

    # for text, label, query in zip(examples['input'], examples['label'], examples['query']):
    for url, title, summary, body, label, query, rationale in zip(examples['url'], examples['title'], examples['summary'], examples['body'], examples['label'], examples['query'], examples['rationale']):
        url = url.rstrip(string.punctuation)
        hostname = parse.urlparse(url).hostname
        title = title + hostname
        text = 'query:%s\n标题:%s\n摘要:%s\n正文:%s\n' % (query, title, summary, body)
        text_id = tokenizer(text, padding=False).input_ids
        label_id = tokenizer(label, padding=False).input_ids
        # post_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是:", padding=False).input_ids
        post_prompt_id = tokenizer("请判断query和网页的标题并给出推理过程,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是:", padding=False).input_ids

        #post_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是").input_ids
        #prefix_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级: bad:不相关\nfair:部分相关,或者内容质量相对较差\ngood:相关,且内容质量较好\nperfect:相关,且权威,或者内容质量很好,或者是百科结果\nawesome:相关,且是官网结果\n").input_ids
        #post_prompt_id = tokenizer("答案:").input_ids
        #text_id = prefix_prompt_id + text_id[:args.max_input_length - len(prefix_prompt_id) - len(post_prompt_id) - len(label_id)] + post_prompt_id
        text_promp_id = text_id[:args.max_input_length - len(post_prompt_id) - len(label_id)] + post_prompt_id
        # text_id = text_id[:args.max_input_length - len(label_id)]

        cls_index = len(text_promp_id) - 1

        if generate_cls:
            input_id = text_promp_id + label_id
        else:
            input_id = text_promp_id 
        attention_mask = [1] * len(input_id)

        label_id = [-100] * len(text_promp_id[:args.max_input_length - len(label_id)]) + label_id

        input_id = input_id + [tokenizer.pad_token_id] * (args.max_input_length - len(input_id))
        #input_id = input_id + [tokenizer.eos_token_id] * (args.max_input_length - len(input_id))
        label_id = label_id + [-100] * (args.max_input_length - len(label_id))
        attention_mask = attention_mask + [0] * (args.max_input_length - len(attention_mask))


        input_ids.append(input_id)
        labels.append(label_id)
        attention_masks.append(attention_mask)
        cls_position.append(cls_index)

        # pre_label_id = tokenizer("相关性等级：",padding=False).input_ids
        # rationale = rationale.split("相关性等级：")[-1]
        rationale_id = tokenizer(rationale, padding=False).input_ids
        rationale_label_id = rationale_id
        ration_prompt_id = tokenizer("请判断query和网页的标题并给出推理过程,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是:", padding=False).input_ids
        rationale_prefix_prompt_id = text_id[:args.max_input_length - len(ration_prompt_id) - len(rationale_label_id)] + ration_prompt_id
        rationale_text_id = rationale_prefix_prompt_id + rationale_label_id
        # rationale_text_id = rationale_text_id[:args.max_input_length] 
        pre_answer_index = len(rationale_prefix_prompt_id) - 1



        rationale_label_id = [-100] * (len(rationale_text_id)-len(rationale_label_id)) + rationale_label_id
        rationale_attention_mask = [1] * len(rationale_text_id)

        rationale_input_id = rationale_text_id + [tokenizer.pad_token_id] * (args.max_input_length - len(rationale_text_id))
        rationale_label_id = rationale_label_id[:args.max_input_length] 
        rationale_label_id = rationale_label_id + [-100] * (args.max_input_length - len(rationale_label_id))
        rationale_attention_mask = rationale_attention_mask + [0]* (args.max_input_length - len(rationale_attention_mask))

        rationale_input_ids.append(rationale_input_id)
        rationale_attention_masks.append(rationale_attention_mask)
        rationale_labels.append(rationale_label_id)
        pre_answer_position.append(pre_answer_index)


    model_inputs = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_masks,
        cls_position=cls_position,
        expl_input_ids=rationale_input_ids,
        expl_labels=rationale_labels,
        expl_attention_masks=rationale_attention_masks,
        pre_answer_position=pre_answer_position
    )
    return model_inputs

def preprocess_raw_cot_new(tokenizer, examples, is_training, generate_cls):
    input_ids = []
    labels = []
    attention_masks = []
    cls_position = []

    rationale_input_ids = []
    rationale_labels = []
    rationale_attention_masks = []
    rationale_cls_position = []
    pre_answer_position = []

    # for text, label, query in zip(examples['input'], examples['label'], examples['query']):
    for url, title, summary, body, label, query, rationale in zip(examples['url'], examples['title'], examples['summary'], examples['body'], examples['label'], examples['query'], examples['rationale']):
        url = url.rstrip(string.punctuation)
        hostname = parse.urlparse(url).hostname
        title = title + hostname
        text = 'query:%s\n标题:%s\n摘要:%s\n正文:%s\n' % (query, title, summary, body)
        text_id = tokenizer(text, padding=False).input_ids
        label_id = tokenizer(label, padding=False).input_ids
        # post_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是:", padding=False).input_ids
        post_prompt_id = tokenizer("请判断query和网页的标题并给出推理过程,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是:", padding=False).input_ids
        rationale_prefix_prompt = \
        f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你是一家搜索引擎公司的数据标注专家，非常擅长判断搜索query和搜索结果的相关性。你需要对一批需要相关性标注的query,搜索结果的title、content进行相关性标注，并给出相关性标注的推理过程。
给你的的输入有用户搜索query和搜索结果。其中用户搜索query可能出现错别字、语序不当、拼写错误等问题，你要尝试消除这些问题的影响，正确把握用户意图和主次需求；搜索结果包括标题和正文两部分。相关性分为五个等级：
A：不相关，即标题和正文内容均与用户query完全无关或几乎不相关，不能满足用户需求或回答query问题；
B：部分相关，即内容与用户query弱相关或仅有小部分相关，对用户有帮助但不能完全满足用户主要需求，或有关联但内容质量相对较差，相关内容不丰富或包含太多无关内容；
C：相关，即内容与用户query相关，正文能够满足用户query主要需求，但内容不够丰富，且不是来自权威站点；
D：相关，即内容与用户query强相关，能够满足用户query的主要需求和次要需求，内容丰富且质量很好，或者是百科结果或权威结果；
E：相关且官方，即内容与用户query强相关，能够完全满足用户query的主要需求和次要需求，且来自官网等绝对权威网站或者寻址结果。
你需要判断相关性等级并给出推理过程。陈述推理过程时，要将判断流程程拆分成多个子流程，一步一步得出判断结果。判断流程包括：分析query意图、分析搜索结果相关性、分析满足query需求情况、分析内容质量和丰富度、分析权威性、总结。语言尽可能精炼。

需要判断的query和搜索结果：
query:{query}
title:{title}
summary: {summary}
content:{body[:600]}

请严格按照以下格式输出：
相关性等级：
推理过程：<|im_end|>
<|im_start|>assistant
相关性等级："""

        #post_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是").input_ids
        #prefix_prompt_id = tokenizer("请判断query和网页的标题,正文以及摘要的相关性等级: bad:不相关\nfair:部分相关,或者内容质量相对较差\ngood:相关,且内容质量较好\nperfect:相关,且权威,或者内容质量很好,或者是百科结果\nawesome:相关,且是官网结果\n").input_ids
        #post_prompt_id = tokenizer("答案:").input_ids
        #text_id = prefix_prompt_id + text_id[:args.max_input_length - len(prefix_prompt_id) - len(post_prompt_id) - len(label_id)] + post_prompt_id
        text_promp_id = text_id[:args.max_input_length - len(post_prompt_id) - len(label_id)] + post_prompt_id
        # text_id = text_id[:args.max_input_length - len(label_id)]

        rationale_prefix_prompt_id = tokenizer(rationale_prefix_prompt, padding=False).input_ids
        text_promp_id = rationale_prefix_prompt_id[:args.max_input_length - len(label_id)]


        cls_index = len(text_promp_id) - 1

        if generate_cls:
            input_id = text_promp_id + label_id
        else:
            input_id = text_promp_id 
        attention_mask = [1] * len(input_id)

        label_id = [-100] * len(text_promp_id[:args.max_input_length - len(label_id)]) + label_id

        input_id = input_id + [tokenizer.pad_token_id] * (args.max_input_length - len(input_id))
        #input_id = input_id + [tokenizer.eos_token_id] * (args.max_input_length - len(input_id))
        label_id = label_id + [-100] * (args.max_input_length - len(label_id))
        attention_mask = attention_mask + [0] * (args.max_input_length - len(attention_mask))


        input_ids.append(input_id)
        labels.append(label_id)
        attention_masks.append(attention_mask)
        cls_position.append(cls_index)

        pre_label_id = tokenizer("相关性等级：",padding=False).input_ids
        rationale = rationale.split("相关性等级：")[-1]
        rationale_id = tokenizer(rationale, padding=False).input_ids
        rationale_label_id = rationale_id
        # ration_prompt_id = tokenizer("请判断query和网页的标题并给出推理过程,正文以及摘要的相关性等级, 选项: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n答案是:", padding=False).input_ids
        # rationale_prefix_prompt_id = text_id[:args.max_input_length - len(ration_prompt_id) - len(rationale_label_id)] + ration_prompt_id
        rationale_prefix_prompt_id = rationale_prefix_prompt_id[:args.max_input_length - len(rationale_label_id)]
        rationale_text_id = rationale_prefix_prompt_id + rationale_label_id
        # rationale_text_id = rationale_text_id[:args.max_input_length] 
        pre_answer_index = len(rationale_prefix_prompt_id) - 1



        rationale_label_id = [-100] * (len(rationale_text_id)-len(rationale_label_id)) + rationale_label_id
        rationale_attention_mask = [1] * len(rationale_text_id)

        rationale_input_id = rationale_text_id + [tokenizer.pad_token_id] * (args.max_input_length - len(rationale_text_id))
        rationale_label_id = rationale_label_id[:args.max_input_length] 
        rationale_label_id = rationale_label_id + [-100] * (args.max_input_length - len(rationale_label_id))
        rationale_attention_mask = rationale_attention_mask + [0]* (args.max_input_length - len(rationale_attention_mask))

        rationale_input_ids.append(rationale_input_id)
        rationale_attention_masks.append(rationale_attention_mask)
        rationale_labels.append(rationale_label_id)
        pre_answer_position.append(pre_answer_index)


    model_inputs = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_masks,
        cls_position=cls_position,
        expl_input_ids=rationale_input_ids,
        expl_labels=rationale_labels,
        expl_attention_masks=rationale_attention_masks,
        pre_answer_position=pre_answer_position
    )
    return model_inputs

def preprocess_chatml(tokenizer, examples, max_input_length):

    system_message = "You are a helpful assistant."
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    input_ids = []
    labels = []
    attention_masks = []
    cls_position = []

    #post_prompt = "\n请判断query和网页的标题,正文以及摘要的相关性等级: A.不相关\nB.部分相关,或者内容质量相对较差\nC.相关,且内容质量较好\nD.相关,且权威,或者内容质量很好,或者是百科结果\nE.相关,且是官网或者寻址结果\n"
    post_prompt = "请判断query和网页的标题,正文以及摘要的相关性等级: bad:不相关\nfair:部分相关,或者内容质量相对较差\ngood:相关,且内容质量较好\nperfect:相关,且权威,或者内容质量很好,或者是百科结果\nawesome:相关,且是官网结果\n"
    for text, label, rationale in zip(examples['input'], examples['label'], examples['rationale']):
        input_id = []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system

        _input_id = tokenizer("<|im_start|>user").input_ids + nl_tokens + \
                tokenizer(text[:1000] + post_prompt).input_ids + [im_end] + nl_tokens
        input_id += _input_id

        _input_id = tokenizer("<|im_start|>assistant").input_ids + nl_tokens + \
                tokenizer("答案是:").input_ids + [im_end] + nl_tokens
        input_id += _input_id

        input_id = input_id[:max_input_length]

        attention_mask = [1] * len(input_id)
        cls_index = len(input_id) - 1

        input_id = input_id + [tokenizer.pad_token_id] * (max_input_length - len(input_id))

        label_id = tokenizer(label, padding=False).input_ids    
        label_id = label_id[:max_input_length]
        label_id = [-100] * len(input_id[:max_input_length - 1]) + label_id
        label_id = label_id + [-100] * (max_input_length - len(label_id))

        attention_mask = attention_mask[:max_input_length]
        attention_mask = attention_mask + [0] * (args.max_input_length - len(attention_mask))

        input_ids.append(input_id)
        labels.append(label_id)
        cls_position.append(cls_index)
        attention_masks.append(attention_mask)

    #input_ids = torch.tensor(input_ids, dtype=torch.int)
    #labels = torch.tensor(labels, dtype=torch.int)
    #cls_position = torch.tensor(cls_position, dtype=torch.int)

    model_inputs = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_masks,
        cls_position=cls_position,
    )
    return model_inputs

def run(args):
    is_streaming = args.is_streaming 
    #### Prepare datasets
    if args.dataset == 'cqa':
        dataset_loader = CQADatasetLoader()
    elif args.dataset == 'svamp':
        dataset_loader = SVAMPDatasetLoader()
    elif args.dataset == 'esnli':
        dataset_loader = ESNLIDatasetLoader()
    elif args.dataset == 'qtc_predict':
        dataset_loader = QTCNewDatasetLoader(data_version=args.dataset, is_streaming=is_streaming)
    elif "qtc" in args.dataset or "cot" in args.dataset:
        dataset_loader = QTCDatasetLoader(data_version=args.dataset,  test_data=args.test_data, is_streaming=is_streaming)
    elif args.dataset == 'qtc6':
        dataset_loader = QTCDatasetLoader(data_version="qtc_v6", is_streaming=is_streaming)
    elif args.dataset == 'qtc8':
        dataset_loader = QTCDatasetLoader(data_version="qtc_v8", is_streaming=is_streaming)
    elif args.dataset == 'anli1':
        dataset_loader = ANLI1DatasetLoader()
    elif args.dataset == 'asdiv':  # NOTE: for augmenting SVAMP only
        dataset_loader = SVAMPDatasetLoader()
        dataset_loader_svamp = SVAMPDatasetLoader()
        dataset_loader_asdiv = ASDivDatasetLoader()
    else:
        raise ValueError

    if args.dataset == 'asdiv':
        datasets_svamp = dataset_loader_svamp.load_from_json()
        datasets_asdiv = dataset_loader_asdiv.load_from_json()
        datasets = DatasetDict({
            'train': concatenate_datasets([datasets_svamp['train'], datasets_asdiv['train']]),
            'test': datasets_svamp['test']
        })
    else:
        datasets = dataset_loader.load_from_json()

    
    if args.llm is None:
        pass
    elif args.llm == 'palm':
        if args.dataset == 'asdiv':
            # training set = SVAMP training + ASDiv training
            train_llm_rationales_svamp, train_llm_labels_svamp = dataset_loader_svamp.load_llm_preds(split='train')
            train_llm_rationales_asdiv, train_llm_labels_asdiv = dataset_loader_asdiv.load_llm_preds(split='train')
            train_llm_rationales = train_llm_rationales_svamp + train_llm_rationales_asdiv
            train_llm_labels = train_llm_labels_svamp + train_llm_labels_asdiv
            # test set = SVAMP test
            test_llm_rationales, test_llm_labels = dataset_loader_svamp.load_llm_preds(split='test')
        else:
            train_llm_rationales, train_llm_labels = dataset_loader.load_llm_preds(split='train')
            test_llm_rationales, test_llm_labels = dataset_loader.load_llm_preds(split='test')
    elif args.llm == 'gpt':
        train_llm_rationales, train_llm_labels = dataset_loader.load_gpt_preds(split='train')
        test_llm_rationales, test_llm_labels = dataset_loader.load_gpt_preds(split='test')
    else:
        pass 

    if args.subsample < 1.0:
        datasets['train'] = datasets['train'].train_test_split(test_size=1.0-args.subsample, seed=args.run)['train']

    if dataset_loader.has_valid:
        if args.llm is None:
            pass
        elif args.llm == 'palm':
            valid_llm_rationales, valid_llm_labels = dataset_loader.load_llm_preds(split='valid')
        elif args.llm == 'gpt':
            valid_llm_rationales, valid_llm_labels = dataset_loader.load_gpt_preds(split='valid')
        else:
            raise ValueError

        datasets['valid'] = datasets['valid'].add_column('llm_label', valid_llm_labels)
        datasets['valid'] = datasets['valid'].add_column('llm_rationale', valid_llm_rationales)
    else:
        #train_valid_datasets = datasets['train'].train_test_split(test_size=0.1, seed=0)

        datasets = DatasetDict({
            'train': datasets['train'],
            'test': datasets['test'],
        })

    if args.label_type == 'gt':
        pass
    elif args.label_type == 'llm' and args.llm is not None:
        if args.dataset not in ['svamp', 'asdiv']:
            train_label_acc = compute_text_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_text_acc(datasets['test']['llm_label'], datasets['test']['label'])
        else:
            train_label_acc = compute_equation_acc(datasets['train']['llm_label'], datasets['train']['label'])
            test_label_acc = compute_equation_acc(datasets['test']['llm_label'], datasets['test']['label'])

        print(f'LLM Train Acc: {train_label_acc:.4f}')
        print(f'LLM Test Acc: {test_label_acc:.4f}')

        datasets['train'] = datasets['train'].remove_columns('label')
        datasets['train'] = datasets['train'].add_column('label', datasets['train']['llm_label'])

    else:
        raise ValueError

    if args.llm is not None:
        if "qtc" in args.dataset or "cot" in args.dataset:
            pass
        else:
            if 'rationale' in datasets['train'].column_names:
                datasets = datasets.remove_columns('rationale')
            datasets = datasets.rename_column('llm_rationale', 'rationale')


    #### Prepare datasets Prepare data for training
    if args.llm == "bloom":
        tokenizer = BloomTokenizerFast.from_pretrained(args.from_pretrained,
                use_fast=False, trust_remote_code=True)
    else:
        if args.llm == "qwen15":
            tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained,
                    padding_side="right", use_fast=False, trust_remote_code=True)
            # tokenizer.pad_token_id = tokenizer.eod_id
        if args.llm == "qwen" or args.llm == "qwen_new":
            tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained,
                    padding_side="right", use_fast=False, trust_remote_code=True)
            tokenizer.pad_token_id = tokenizer.eod_id
        if args.llm == "qwen2" or args.llm == "qwen2_pw":
            tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained, model_max_length=args.max_input_length,
                    padding_side="right", use_fast=True, trust_remote_code=True)
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif args.llm == "YI":
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        elif args.llm == "glm":
            tokenizer.eod_id = tokenizer.eos_token_id
        elif args.llm == "chatGlm":
            tokenizer.eod_id = tokenizer.pad_token_id
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.from_pretrained,
                use_fast=False, trust_remote_code=True)


    if 'nli' in args.dataset:
        datasets = datasets.map(
            lambda example: {'input': tokenizer.eos_token.join([example['premise'], example['hypothesis']])},
            remove_columns=['premise', 'hypothesis'],
        )


    if args.model_type == 'task_prefix' and args.llm is not None:
        def tokenize_function(examples):
            # print("tokenize:")
            # print(examples.keys())
            if args.generate_cls:
                if args.is_instruct_finetune:
                    model_inputs = preprocess_chatml(tokenizer, examples, args.max_input_length)
                else:
                    if args.add_aggfeat:
                        model_inputs = preprocess_raw_aggfeat(tokenizer, examples, True, args.generate_cls)
                    elif args.add_rationale_loss or args.only_gen:
                        model_inputs = preprocess_raw_cot(tokenizer, examples, True, args.generate_cls)
                        # model_inputs = preprocess_raw_cot_new(tokenizer, examples, True, args.generate_cls)
                    else:
                        model_inputs = preprocess_raw(tokenizer, examples, True, args.generate_cls)
                        # model_inputs = preprocess_raw_new(tokenizer, examples, True, args.generate_cls)
                #print(model_inputs["input_ids"][:3])
                #print(model_inputs["labels"][:3])
                #print(model_inputs["cls_position"][:3])

            elif args.llm == "mt0":
                model_inputs = tokenizer([text.replace("</s>", "") for text in examples['input']], max_length=args.max_input_length,  truncation=True, padding=True)
                with tokenizer.as_target_tokenizer():
                    label_output_encodings = tokenizer(examples['label'], max_length=2, truncation=True)
                model_inputs['labels'] = label_output_encodings['input_ids']
                #print(model_inputs["labels"][:3])
            else:
                #model_inputs = tokenizer([text+"<SEP>" for text in examples['input']], padding='max_length', max_length=args.max_input_length, truncation=True)
                if args.loss_type == "RMloss":
                    model_inputs = preprocess_raw_groupby(tokenizer, examples, True, args.generate_cls, args.max_input_length)
                else:
                    model_inputs = preprocess_raw(tokenizer, examples, True, args.generate_cls)
                    model_inputs['labels']  = model_inputs['input_ids'].copy()
                # print(model_inputs["input_ids"][:3])
            return model_inputs

        def tokenize_function_test(examples):
            if args.generate_cls:
                if args.is_instruct_finetune:
                    model_inputs = preprocess_chatml(tokenizer, examples, args.max_input_length)
                else:
                    if args.add_aggfeat:
                        model_inputs = preprocess_raw_aggfeat(tokenizer, examples, True, args.generate_cls)
                    elif args.add_rationale_loss or args.only_gen:
                        model_inputs = preprocess_raw_cot(tokenizer, examples, True, args.generate_cls)
                        # model_inputs = preprocess_raw_cot_new(tokenizer, examples, True, args.generate_cls)

                    else:
                        model_inputs = preprocess_raw(tokenizer, examples, False, args.generate_cls)
                        # model_inputs = preprocess_raw_new(tokenizer, examples, True, args.generate_cls)
                #print(model_inputs["input_ids"][:3])
                #print(model_inputs["cls_position"][:3])
            elif args.llm == "mt0":
                model_inputs = tokenizer([text.replace("</s>", "") for text in examples['input']], max_length=args.max_input_length,  truncation=True, padding=True)
                with tokenizer.as_target_tokenizer():
                    label_output_encodings = tokenizer(examples['label'], max_length=2, truncation=True)
                model_inputs['labels'] = label_output_encodings['input_ids']
            else:
                #model_inputs = tokenizer([text+"<SEP>" for text in examples['input']], padding='max_length', max_length=args.max_input_length, truncation=True)
                model_inputs = preprocess_raw(tokenizer, examples, False, args.generate_cls)
                model_inputs['labels']  = model_inputs['input_ids'].copy()

                # print("sly tokenizer 3", tokenizer)
                # print("sly input_ids", model_inputs["input_ids"][:1])
                # print("sly cls_position", model_inputs["cls_position"][:1])
                # print("sly labels", model_inputs["labels"][:1])

            return model_inputs


    elif args.model_type == 'standard':
        def tokenize_function(examples):
            model_inputs = tokenizer(
                examples['input'],
                max_length=args.max_input_length,
                truncation=True
            )

            with tokenizer.as_target_tokenizer():
                label_output_encodings = tokenizer(examples['label'], max_length=256, truncation=True)

            model_inputs['labels'] = label_output_encodings['input_ids']

            return model_inputs

    else:
        raise ValueError

    def gen_from_iterable_dataset(iterable_ds):
        yield from iterable_ds

    # 【sunliuying debug】手动迭代生成器
    # print("Features:", datasets["test"].features)
    # for i, item in enumerate(partial(gen_from_iterable_dataset, datasets["test"])()):
    #     print(f"Item {i}: {item}")
    #     if i >= 10:
    #         break
    # def gen_from_iterable_dataset_wrapped(dataset):
    #     for i, item in enumerate(gen_from_iterable_dataset(dataset)):
    #         try:
    #             yield item
    #         except Exception as e:
    #             print(f"Error at item {i}: {item}")
    #             print(f"Exception: {e}")
    #             raise  # 重新抛出异常以终止流程

    # dataset_test = Dataset.from_generator(
    #     partial(gen_from_iterable_dataset_wrapped, datasets["test"]),
    #     features=datasets["test"].features  
    # )  

    dataset_test = Dataset.from_generator(
            partial(gen_from_iterable_dataset, datasets["test"]), 
            features=datasets["test"].features
    )
    # dataset_train = Dataset.from_generator(partial(gen_from_iterable_dataset,
    #     datasets["train"]), features=datasets["train"].features)
    # datasets["train"] = dataset_train
    # print(len(dataset_test))
    eval_querys = dataset_test['query'] 
    eval_urls = dataset_test['url'] 

    eval_data_type = dataset_test['filename'] 

    if args.llm is None:
        tokenized_datasets = datasets.map(
            tokenize_function,
            remove_columns=['input', 'label'],
            batched=True
        )
    else:
        if is_streaming:
            print('before_train: ', datasets["train"])
            print(datasets["train"].features)
            tokenized_datasets_train = datasets["train"].map(
                tokenize_function,
                # remove_columns=['input', 'rationale', 'label', 'query', 'url', 'filename'],
                # remove_columns = ['aggfeat'],
                batched=True
            )
            print('after_train: ', tokenized_datasets_train)
            print(tokenized_datasets_train.features)
        else:
            tokenized_datasets_train = datasets["train"].map(
                tokenize_function,
                #remove_columns=['input', 'rationale', 'label', 'query', 'url', 'filename'],
                batched=True,
                num_proc=32
            )
        print('before_test: ', dataset_test)
        tokenized_datasets_test = dataset_test.map(
            tokenize_function_test,
            # remove_columns=['input', 'rationale', 'label', 'query', 'url', 'filename'],
            # remove_columns = ['aggfeat'],
            batched=True
        )
        print('after_test: ', tokenized_datasets_test)
        


    if args.model_type == 'standard':
        if args.dataset not in ['svamp', 'asdiv']:
            compute_metrics = compute_metrics_text_aux(tokenizer)
        else:
            compute_metrics = compute_metrics_equation_aux(tokenizer)

    else:
        if args.dataset not in ['svamp', 'asdiv']:
            #compute_metrics = compute_metrics_text(tokenizer)
            compute_metrics = compute_metrics_cls_acc(tokenizer, eval_querys,
                    eval_urls, eval_data_type, args.loss_type)
        else:
            compute_metrics = compute_metrics_equation(tokenizer)

    
    #print("n_shards: {}".format(tokenized_datasets_train.n_shards))
    #tokenized_datasets_train = split_dataset_by_node(tokenized_datasets_train, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
    #tokenized_datasets_train.save_to_disk("data/train.dat")
    tokenized_datasets = DatasetDict({
        'train': tokenized_datasets_train,
        'test': tokenized_datasets_test,
    })

    # print('tokenized_datasets_train:', tokenized_datasets_train)
    # eixt()

    train_and_evaluate(args, args.run, tokenizer, tokenized_datasets, compute_metrics, dataset_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--subsample', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--max_steps', type=int, default=-1)
    parser.add_argument('--num_train_epochs', type=int, default=4)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=500)
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--optimizer_name', type=str, default='AdamW')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--adam_beta2', type=float, default=0.95)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--lr_scheduler_type', type=str, default='cosine')
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--from_pretrained', type=str, default='google/t5-v1_1-base')
    parser.add_argument('--output_dir', type=str, default='models/qwen/output')
    parser.add_argument('--label_type', type=str, default='gt')
    parser.add_argument('--llm', type=str, default='palm')
    parser.add_argument('--max_input_length', type=int, default=1024)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--eval_bs', type=int, default=8)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--gen_max_len', type=int, default=64)
    parser.add_argument('--parallelize', action='store_true')
    parser.add_argument('--freeze_transformer', action='store_true')
    parser.add_argument('--use_lora', action='store_true')
    parser.add_argument('--use_mean_embdding', action='store_true')
    parser.add_argument('--qwen_new', action='store_true')
    parser.add_argument('--resume_from_checkpoint', action='store_true')
    parser.add_argument('--model_type', type=str, default='task_prefix')
    parser.add_argument('--loss_type', type=str, default='default')
    parser.add_argument('--test_data', type=str, default='default')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout_p', type=float, default=0.05)
    parser.add_argument('--generate_cls', action='store_true')
    parser.add_argument('--only_evaluate', action='store_true')
    parser.add_argument('--only_predict', action='store_true')
    parser.add_argument('--no_log', action='store_true')
    parser.add_argument('--output_rationale', action='store_true')
    parser.add_argument('--add_rationale_loss', action='store_true')
    parser.add_argument('--only_gen', action='store_true')
    parser.add_argument('--distill', action='store_true')
    parser.add_argument('--add_aggfeat', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--is_streaming', action='store_true')
    parser.add_argument('--is_instruct_finetune', action='store_true')
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")

    args = parser.parse_args()

    run(args)
