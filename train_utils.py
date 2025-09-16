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


import sys
import os
import json
import shutil
import logging

from scipy.special import softmax, expit 
import torch
import deepspeed
import numpy as np
import pandas as pd
# from transformers.deepspeed import HfDeepSpeedConfig
from transformers.integrations import HfDeepSpeedConfig
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers import Seq2SeqTrainingArguments, TrainingArguments ,Seq2SeqTrainer,BitsAndBytesConfig
from transformers import BloomTokenizerFast, BloomForCausalLM,AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers.trainer_utils import set_seed
from scipy.special import softmax
from datasets.distributed import split_dataset_by_node
from datasets import Dataset
import datasets

from model_utils import TaskPrefixDataCollator, TaskPrefixTrainer, TaskPrefixSeq2SeqTrainer, PairWiseDataCollator, PairwiseTrainer, RMDataCollator
#sys.path.insert(0,'/mnt/data/distill-llms/Baichuan-13B-Chat')
# sys.path.insert(0,'/mnt/data/lhh/liuhuanhuan/qtc-llm/models/Qwen-14B-Chat')
# sys.path.insert(0,'/mnt/data/train_logs/Qwen2.5-14B-Instruct')

#from modeling_baichuan_qtc_nlu import BaichuanForCausalLM
#sys.path.insert(0,'/mnt/data/Qwen/qwen/Qwen-14B-Chat')
#from modeling_qwen_cls import QWenLMHeadModel 
# from modeling_qwen_cls import QWenLMHeadModelCls 
#sys.path.insert(0,'/mnt/workspace/YI-34B/01ai/Yi-34B-Chat')
#from modeling_yi import YiForCausalLM 
#sys.path.insert(0,'/mnt/data/glm-large-chinese')
#from modeling_glm import GLMForSequenceClassification, GLMForConditionalGeneration 
from utils import print_rank_0



def get_config_dir(args):
    return f'{args.dataset}/{args.from_pretrained.split("/")[1]}/{args.model_type}/{args.llm}/{args.subsample}/{args.label_type}/{args.alpha}/{args.max_input_length}/{args.grad_steps*args.batch_size}/{args.optimizer_name}/{args.lr}'

def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def train_and_evaluate(args, run, tokenizer, tokenized_datasets, compute_metrics, orig_dataset_test):
    set_seed(run)

    output_dir = args.output_dir
    logging_dir = f'{output_dir}/logs'  # for training logs

    if args.no_log:
        logging_strategy = 'no'
        logging_dir = None
    else:
        logging_strategy = 'steps'

    # 0611 qwen3版本
    if args.llm == "mt0":
        training_args = Seq2SeqTrainingArguments(
            output_dir,
            remove_unused_columns = False,
            eval_strategy = 'steps',
            eval_steps=args.eval_steps,
            save_strategy='steps',
            save_steps=args.save_steps,
            logging_strategy=logging_strategy,
            logging_steps=args.log_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            adam_beta2=args.adam_beta2,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            deepspeed=args.deepspeed,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_accumulation_steps=args.grad_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_train_batch_size*2,
            predict_with_generate=True,
            seed=run,
            local_rank=args.local_rank,
            prediction_loss_only=False,
            bf16=args.bf16,
            dataloader_drop_last=False,
            generation_max_length=args.gen_max_len,
            load_best_model_at_end=True,
            overwrite_output_dir=True,
            log_level="debug",
            report_to=["tensorboard"],
        )
    else:
        training_args = Seq2SeqTrainingArguments(
            output_dir,
            remove_unused_columns = False,
            # evaluation_strategy = 'steps',
            eval_strategy = 'steps',
            eval_steps=args.eval_steps,
            save_strategy='steps',
            save_steps=args.save_steps,
            logging_dir=logging_dir,
            logging_strategy=logging_strategy,
            logging_steps=args.log_steps,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            adam_beta2=args.adam_beta2,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_ratio=args.warmup_ratio,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            deepspeed=args.deepspeed,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_accumulation_steps=args.grad_steps,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.eval_bs,
            seed=run,
            local_rank=args.local_rank,
            # dispatch_batches=False,
            prediction_loss_only=False,
            bf16=args.bf16,
            fp16=args.fp16,
            dataloader_drop_last=False,
            load_best_model_at_end=True,
            overwrite_output_dir=True,
            log_level="debug",
            report_to=["tensorboard"],
        )
    print_rank_0(training_args)


    if args.llm == "bloom":
        model = BloomForCausalLM.from_pretrained(args.from_pretrained, torch_dtype=torch.float16, trust_remote_code=True)
    elif args.llm == "mt0":
        config = AutoConfig.from_pretrained(args.from_pretrained , trust_remote_code=True) 
        config.num_labels = 5
        from transformers import T5ForConditionalGeneration, T5ForConditionalGenerationCls, T5ForSequenceClassification
        if args.use_mean_embdding:
            model = T5ForConditionalGenerationCls.from_pretrained(args.from_pretrained, config=config, trust_remote_code=True)
        else:
            model = T5ForConditionalGeneration.from_pretrained(args.from_pretrained, config=config, trust_remote_code=True)
            model.config.num_labels = 5
            model.num_labels = 5

    elif args.llm == "qwen_new":
        model = QWenLMHeadModelCls.from_pretrained(args.from_pretrained,  trust_remote_code=True)
    elif args.llm == "qwen_pw":
        model = QWenLMHeadModelCls.from_pretrained(args.from_pretrained,  trust_remote_code=True)
        model.config.pad_token_id = model.config.eos_token_id
    elif args.llm == "qwen2":
        from transformers import Qwen2MoeForCausalLM
        model_config = AutoConfig.from_pretrained(args.from_pretrained, trust_remote_code=True)
        model_config.use_cache = False
        model = AutoModelForCausalLM.from_pretrained(args.from_pretrained, attn_implementation="flash_attention_2", config=model_config, torch_dtype=torch.bfloat16, trust_remote_code=True)
        #model = Qwen2MoeForCausalLM.from_pretrained(args.from_pretrained, attn_implementation="flash_attention_2", config=model_config, torch_dtype=torch.bfloat16)
        model.config.pad_token_id = model.config.eos_token_id
        #model.train()
        '''
        from llamafactory.hparams import get_infer_args, get_train_args
        from llamafactory.model import load_model, load_tokenizer
        TRAIN_ARGS = {
            "model_name_or_path": args.from_pretrained,
            "stage": "sft",
            "do_train": True,
            "finetuning_type": "full" if not args.use_lora else "lora",
            "deepspeed": args.deepspeed,
            "ddp_find_unused_parameters": False,
            "template": "qwen",
            "cutoff_len": 640,
            "flash_attn": "fa2",
            "output_dir": args.output_dir,
            "overwrite_cache": True,
            "use_fast_tokenizer": True,
            "bf16": True,
        }
        model_args, _, _, finetuning_args, _ = get_train_args(TRAIN_ARGS)
        tokenizer_module = load_tokenizer(model_args)
        model = load_model(tokenizer_module["tokenizer"], model_args, finetuning_args, is_trainable=True)
        '''
    elif args.llm == "qwen2_pw":
        from transformers import Qwen2ForSequenceClassification
        model = Qwen2ForSequenceClassification.from_pretrained(args.from_pretrained, num_labels=5, trust_remote_code=True)
        # model = AutoModelForSequenceClassification.from_pretrained(args.from_pretrained, num_labels=5, trust_remote_code=True)
        model.config.num_labels = 5
        model.config.pad_token_id = model.config.eos_token_id
        model.num_labels = 5
    elif args.llm == "bert_pw":
        model = AutoModelForSequenceClassification.from_pretrained(args.from_pretrained, num_labels=1, trust_remote_code=True)
        model.config.num_labels = 1
        model.num_labels = 1
    elif args.llm == "glm":
        config = AutoConfig.from_pretrained(args.from_pretrained , trust_remote_code=True) 
        config.num_labels = 5
        config.classifier_dropout = 0.1
        model = GLMForConditionalGeneration.from_pretrained(args.from_pretrained, trust_remote_code=True)
    elif args.llm == "chatGlm":
        config = AutoConfig.from_pretrained(args.from_pretrained , trust_remote_code=True) 
        model = AutoModelForCausalLM.from_pretrained(args.from_pretrained, trust_remote_code=True)
    elif args.llm == "YI":
        model_config = AutoConfig.from_pretrained(args.from_pretrained, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(args.from_pretrained, config=model_config, trust_remote_code=True)
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
    else:
        model = AutoModelForCausalLM.from_pretrained(args.from_pretrained, trust_remote_code=True)

    if args.use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        target_modules = find_all_linear_names(model)
        qwen_target_modules = ["c_attn", "c_proj", "w1", "w2"]
        print("target_modules:")
        print(target_modules)
        if args.llm == "mt0":
            target_modules = ['k', 'o', 'q', 'v']
        if args.llm == "YI":
            target_modules = ['k_proj', 'o_proj', 'q_proj', 'v_proj']
        if "qwen" in args.llm:
            target_modules = ["c_attn", "c_proj", "w1", "w2"]
        if "qwen2" in args.llm:
            target_modules = ["q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout_p,
        )
        model.enable_input_require_grads()
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        for name, param in model.named_parameters():
            if "cls" in name:
                param.requires_grad = True
    if args.parallelize:
        model.parallelize()
    
    # clear output dir if already exists
    #if os.path.exists(output_dir):
    #    logging.info('Found existing ckpt directory. Deleted the old directory for the latest run.')
    #    print(output_dir)
    #    shutil.rmtree(output_dir)

    if args.model_type == 'task_prefix':
        if args.loss_type == "RMloss":
            data_collator = RMDataCollator(tokenizer=tokenizer)
            train_dataset = tokenized_datasets["train"].remove_columns(['input', 'label',  'url', 'filename', 'rationale'])
            test_dataset = tokenized_datasets["test"].remove_columns(['input', 'label', 'url', 'filename', 'rationale'])
        else:
            data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
            # train_dataset = tokenized_datasets["train"].remove_columns(['input', 'label', 'query', 'url', 'filename', 'rationale'])
            # test_dataset = tokenized_datasets["test"].remove_columns(['input', 'label', 'query', 'url', 'filename', 'rationale'])
            train_dataset = tokenized_datasets["train"].remove_columns(['title', 'body', 'summary', 'label', 'query', 'url', 'filename', 'rationale', 'multi_label'])
            test_dataset = tokenized_datasets["test"].remove_columns(['title', 'body', 'summary', 'label', 'query', 'url', 'filename', 'rationale', 'multi_label'])
            print('train_util_train: ', train_dataset)
            print('train_util_test: ', test_dataset)
            print("len(test_dataset)")
            print(len(test_dataset))
    elif args.model_type == 'standard':
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    elif args.model_type == 'pairwise':
        data_collator = PairWiseDataCollator(tokenizer=tokenizer, model=model)
        train_dataset = tokenized_datasets["train"].remove_columns(['prompt_pos', 'prompt_neg', 'query', 'url_pos', 'url_neg', 'filename'])
        test_dataset = tokenized_datasets["test"].remove_columns(['prompt_pos', 'prompt_neg', 'query', 'url_pos', 'url_neg','filename'])
        #data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)
        #data_collator = PairWiseDataCollator(tokenizer=tokenizer, model=model)
    else:
        raise ValueError


    trainer_kwargs = {
        'alpha': args.alpha,
        'output_rationale': args.output_rationale,
        'add_rationale_loss': args.add_rationale_loss,
        'hidden_size': args.hidden_size, 
        'use_mean_embdding': args.use_mean_embdding, 
        'qwen_new': args.qwen_new, 
        'generate_cls': args.generate_cls, 
        'llm': args.llm, 
        'loss_type': args.loss_type, 
        'model': model,
        'args': training_args,
        'train_dataset': train_dataset,
        'eval_dataset': test_dataset,
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
        'only_gen': args.only_gen,
        'distill': args.distill
    }
    

    if args.model_type == 'task_prefix':
        if args.llm == "mt0":
            trainer = TaskPrefixSeq2SeqTrainer(**trainer_kwargs)
        else:
            trainer = TaskPrefixTrainer(**trainer_kwargs)
    elif args.model_type == 'standard':
        trainer_kwargs.pop('alpha')
        trainer_kwargs.pop('output_rationale')
        trainer = Seq2SeqTrainer(**trainer_kwargs)
    elif args.model_type == 'pairwise':
        trainer = PairwiseTrainer(**trainer_kwargs)
    else:
        raise ValueError
    
    if args.local_rank == 0:
        print("all parameters:")
        for name, param in model.named_parameters():
            print("+++++++",name,param.requires_grad)

    #trainer.train("checkpoint-20000")
    if args.only_evaluate:
        trainer.evaluate()
    elif args.only_predict:
        print_rank_0("*** Predict ***")

        orig_dataset_test = tokenized_datasets["test"]
        if "pairwise" in args.loss_type:
            pred_datasets = orig_dataset_test.remove_columns(['query', 'url_pos', 'url_neg', 'filename', 'prompt_pos', 'prompt_neg'])
        elif args.loss_type == "RMloss":
            pred_datasets = orig_dataset_test.remove_columns(['input', 'label', 'url', 'filename', 'rationale'])
        else:
            # pred_datasets = orig_dataset_test.remove_columns(['input', 'query', 'label', 'url', 'filename', 'rationale'])
            pred_datasets = orig_dataset_test.remove_columns(['title', 'body', 'summary', 'label', 'query', 'url', 'filename', 'rationale', 'multi_label'])
        predictions, _, _ = trainer.predict(pred_datasets)
        #print(predictions)
        #print(len(predictions))
        #print(len(predictions[0]))
        logits = predictions[0]

        if args.loss_type == "order_corn_loss":
            probas = expit(logits[:, 1:])
            probas2 = np.cumprod(probas, axis=1)
            predict_levels = probas2 > 0.5
            predictions = np.sum(predict_levels, -1)
            predict_scores = np.sum(probas * [1,2,3,4], -1)
            pred_labels = predictions
        elif args.loss_type == "order_coral_loss":
            probas = expit(logits)
            thresholds = np.array([0.136, 0.337, 0.505, 0.759])
            predict_levels = probas > thresholds 
            predictions = np.sum(predict_levels, -1)
            predict_scores = np.sum(probas * [1,2,3,4], -1)
            #predict_scores = np.sum(probas * [1,1,1,1], -1)
            pred_labels = predictions
        elif "RMloss" in args.loss_type:
            predict_scores = logits
            pred_labels = np.argmax(logits, axis=-1)
        elif "pairwise" in args.loss_type:
            predict_scores = logits
            pred_labels = np.argmax(logits, axis=-1)
        else:
            logits_softmax = softmax(logits, axis=-1)
            predict_scores = logits_softmax * [0,1,2,3,4]
            predict_scores = np.sum(predict_scores, axis=-1)
            pred_labels = np.argmax(logits_softmax, axis=-1)
        #querys = orig_dataset_test["query"] 
        #print(len(querys))
        #urls = orig_dataset_test["url"] 
        #labels = orig_dataset_test["label"] 
        label_map = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4,
        }
        if trainer.is_world_process_zero():
            output_predict_file = os.path.join(args.output_dir, "predictions_longdoc.txt")
            with open(output_predict_file, "w") as writer:
                #writer.write("index\tprediction\n")
                #for query, url, label, pred_label, pred_score, logits in zip(querys, urls, labels, pred_labels, predict_scores, predictions[0]):
                #    logits_str = ",".join([str(v) for v in logits])
                #    writer.write(f'{query}\t{url}\t{label}\t{pred_label}\t{pred_score}\t{logits_str}\n')
                for example, pred_logits, pred_label, pred_score in zip(orig_dataset_test, logits, pred_labels, predict_scores):
                    #logits_str = ",".join([str(v) for v in pred_logits])
                    #output_d = {}
                    #output_d["query"] = example["query"]
                    #output_d["url"] = example["url_pos"]
                    #output_d["label"] = example["label_pos"]
                    #output_d["data_type"] = example["filename"]

                    example["pred_label"] = int(pred_label)
                    example["label"] = label_map[example["label"]] 
                    example["logits"] = pred_logits.tolist()
                    example["score"] = float(pred_score)
                    #del example["labels"]
                    #del example["input_ids"]
                    #del example["attention_mask"]
                    #del example["q2d_input_ids"]
                    #del example["q2d_attention_masks"]
                    #del example["token_type_ids"]
                    #del example["input"]
                    writer.write(json.dumps(example, ensure_ascii=False)+"\n")
    else:
        if args.resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
            trainer.save_model()
        else:
            trainer.train()
            trainer.save_model()
