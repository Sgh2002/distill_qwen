

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
        
        prefix_prompt_id = tokenizer("<|im_start|>user\n", padding=False, add_special_tokens=False).input_ids
        post_prompt_id = tokenizer("/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", padding=False, add_special_tokens=False).input_ids
        text_id = prefix_prompt_id + text_id[:args.max_input_length - len(prefix_prompt_id) - len(post_prompt_id) - len(label_id)] + post_prompt_id


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
    # for i in range(3):
    #     print(f'====第{i}条====')
    #     print(tokenizer.decode(model_inputs["input_ids"][i], skip_special_tokens=False))
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


    dataset_test = Dataset.from_generator(
            partial(gen_from_iterable_dataset, datasets["test"]), 
            features=datasets["test"].features
    )
    
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
                batched=True
            )
            print('after_train: ', tokenized_datasets_train)
            print(tokenized_datasets_train.features)
        else:
            tokenized_datasets_train = datasets["train"].map(
                tokenize_function,
                batched=True,
                num_proc=32
            )
        print('before_test: ', dataset_test)
        tokenized_datasets_test = dataset_test.map(
            tokenize_function_test,
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

    tokenized_datasets = DatasetDict({
        'train': tokenized_datasets_train,
        'test': tokenized_datasets_test,
    })

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
