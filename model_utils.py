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
from dataclasses import dataclass
import pandas as pd
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import PreTrainedTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Trainer
from transformers import BloomTokenizerFast, BloomForCausalLM
from transformers.tokenization_utils_base import BatchEncoding
from losses import corn_loss, coral_loss, coral_loss_v2, OrdinalRegressionLoss, OR_loss, CondorOrdinalCrossEntropy, AM_Softmax_v2 ,AM_Softmax,AMSoftmax,RMLoss 
from utils import print_rank_0
from typing import List, Tuple

torch.set_printoptions(threshold=10000)


"""T5 Multi-Task by Task Prefix
"""
class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pred_features = features_df.loc[:,
                ~features_df.columns.isin(['sumary'])].to_dict('records')
        #pred_features = features_df
        #expl_features = features_df.loc[:, ~features_df.columns.isin(['labels','input_ids', 'attention_mask', 'label_onehot', 'cls_position','semantic_labels_softmax', 'logits'])].rename(
        #    columns={'q2d_labels': 'labels', 'q2d_input_ids': 'input_ids', 'q2d_attention_masks': 'attention_mask'}).to_dict('records')
        #extra_features = features_df.loc[:,
        #        features_df.columns.isin(['input_ids', 'label_onehot', 'labels', 'label_neg','logits', 'cls_position', 'semantic_labels_softmax'])].to_dict('records')

        pred_features = super().__call__(pred_features, return_tensors)
        #expl_features = super().__call__(expl_features, return_tensors)
        #extra_features = super().__call__(extra_features, return_tensors)

        return {
            'pred': pred_features,
            #'extra': extra_features,
        }

@dataclass
class RMDataCollator:
    tokenizer: PreTrainedTokenizer
    max_length: int = 640

    def __call__(
        self, examples: List[str] = []  
        ) -> Tuple[List[BatchEncoding], List[int]]:
        features_df = pd.DataFrame(examples)
        extra_features = features_df.loc[:,
                features_df.columns.isin(['input_ids', 'label_onehot', 'label_pos', 'label_neg','logits', 'cls_position', 'semantic_labels_softmax'])].to_dict('records')

        batch_k_lens = [0]
        input_ids, attention_masks, labels = [], [], []

        batch_inp_ids, batch_attn_masks, batch_label = [], [],[]
        last_query = None
        for i, example in enumerate(examples):
            inp_ids = example["input_ids"]
            attn_masks = example["attention_mask"]
            query = example["query"]
            label_onehot = example["label_onehot"]
            semantic_labels_softmax = example["semantic_labels_softmax"]
            label_modal = np.argmax(label_onehot)
            #label_modal = np.argmax(semantic_labels_softmax)
            #print(query)
            #print("--------")

            if last_query != None and query != last_query:
                if len(batch_inp_ids) > 0:
                    input_ids.extend(batch_inp_ids)
                    attention_masks.extend(batch_attn_masks)
                    labels.extend(batch_label)
                    batch_k_lens.append(batch_k_lens[-1] + len(batch_inp_ids))
                batch_inp_ids, batch_attn_masks, batch_label = [], [], []

            batch_inp_ids.append(inp_ids)
            batch_attn_masks.append(attn_masks)
            batch_label.append(label_modal)
            last_query = query

        if len(batch_inp_ids) > 0:
            input_ids.extend(batch_inp_ids)
            attention_masks.extend(batch_attn_masks)
            labels.extend(batch_label)
            batch_k_lens.append(batch_k_lens[-1] + len(batch_inp_ids))

        #print_rank_0(batch_k_lens)
        batch =  {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_masks),
            "labels": torch.tensor(labels),
            "k_lens": batch_k_lens
                }

        return batch


class PairWiseDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        features_df = pd.DataFrame(features)
        pos_features = features_df.loc[:,
                ~features_df.columns.isin(['input_ids_neg','attention_mask_neg', 'semantic_labels_softmax_neg', "labels2"])].to_dict('records')
        neg_features = features_df.loc[:,
                ~features_df.columns.isin(['input_ids', 'attention_mask', 'semantic_labels_softmax_pos', "labels"])].rename(
                        columns={'input_ids_neg': 'input_ids', 'attention_mask_neg': 'attention_mask', 'labels2': 'labels'}).to_dict('records')
        extra_features = features_df.loc[:,
                features_df.columns.isin(['input_ids', 'label_pos', 'label_neg', 'semantic_labels_softmax_pos', 'semantic_labels_softmax_neg'])].to_dict('records')

        pos_features = super().__call__(pos_features, return_tensors)
        neg_features = super().__call__(neg_features, return_tensors)
        extra_features = super().__call__(extra_features, return_tensors)

        return {
            'pred1': pos_features,
            'pred2': neg_features,
            'extra': extra_features,
        }

class TaskPrefixTrainer(Trainer):
    def __init__(self, alpha, output_rationale, add_rationale_loss,
            hidden_size, use_mean_embdding, qwen_new, generate_cls, llm, loss_type, only_gen, distill, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale
        self.add_rationale_loss = add_rationale_loss 
        self.only_gen = only_gen 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size 

        num_class = 5
        hidden_dropout = 0.1
        self.official = True 
        self.mean_embedding = use_mean_embdding
        self.qwen_new = qwen_new 
        self.generate_cls = generate_cls
        self.llm = llm
        self.distill = distill 
        self.take_softmax = False 
        self.use_sce = False 
        self.q2d_d = {}
        self.world_size = int(os.getenv('WORLD_SIZE', '1'))
        self.loss_type = loss_type 
        self.IMP_WEIGHT = False 

    def get_rce_loss(self, pred, cls_labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.clamp(cls_labels, min=1e-4, max=1.0)
        rce_loss = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1)).mean()
        return rce_loss

    # def compute_loss(self, model, inputs, return_outputs=False):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        if self.generate_cls:
            if self.distill:
                pred_outputs = model(input_ids=inputs['pred']['input_ids'], attention_mask=inputs['pred']['attention_mask'], labels=inputs['pred']['labels'], return_dict=True)
                lm_logits = pred_outputs.logits
                
                if self.take_softmax:
                    log_probs = F.log_softmax(lm_logits, dim=-1) 
                else:
                    log_probs = lm_logits 

                cls_position = inputs['pred']['cls_position']

                batchsize = lm_logits.shape[0]
                labels = inputs['pred']['labels']

                lm_logits_index = log_probs[torch.arange(0 , batchsize), cls_position]
                if "qwen" in self.llm:
                    target_index = [32,33,34,35,36] #"A,B,C,D,E"
                elif self.llm == "glm":
                    target_index = [196,415,228,424,794]
                elif self.llm == "YI":
                    target_index = [13420, 29503, 12654, 33328, 53092]
                elif self.llm == "chatGlm":
                    target_index = [316, 347, 319, 367, 397]
                lm_logits_cls_mlp = lm_logits_index[:, target_index] 
                teacher_logits = inputs['pred']['new_logits'].to(torch.float32)
                T = 2.0
                criterion  = nn.KLDivLoss(reduction="batchmean")#KL散度
                distill_loss = criterion(F.log_softmax(lm_logits_cls_mlp/T, dim=-1).to(self.device),F.softmax(teacher_logits/T, dim=-1).to(self.device)) * (T) ** 2
                # teacher_probs = inputs['pred']['probs'].to(torch.float32)
                # criterion  = nn.KLDivLoss(reduction="batchmean")#KL散度
                # T = 1.0
                # distill_loss = criterion(F.log_softmax(lm_logits_cls_mlp, dim=-1).to(self.device),teacher_probs.to(self.device)) * (T) ** 2
                loss = distill_loss
                # loss_ce_fn = CrossEntropyLoss()  # target-label onehot or int-tensor / logits not need softmax!
                # cls_loss = loss_ce_fn(input=lm_logits_cls_mlp, target=teacher_probs.to(self.device))
                # loss = cls_loss
            elif self.only_gen:
                answer_position = inputs['pred']['pre_answer_position']
                generated_ids_forward = inputs['pred']['expl_input_ids'].clone()
                expl_outputs = model(input_ids=inputs['pred']['expl_input_ids'], attention_mask=inputs['pred']['expl_attention_masks'], labels=inputs['pred']['expl_labels'], return_dict=True)
                expl_loss = expl_outputs.loss
                expl_loss = torch.where(torch.isnan(expl_loss), 0, expl_loss)
                lm_logits = expl_outputs.logits

                if self.take_softmax:
                    log_probs = F.log_softmax(lm_logits, dim=-1) 
                else:
                    log_probs = lm_logits 
                
                batchsize = lm_logits.shape[0]
                lm_logits_index = log_probs[torch.arange(0, batchsize), answer_position]
                softmaxt_T = 0.9
                if "qwen" in self.llm:
                    target_index = [32,33,34,35,36]
                elif self.llm == "glm":
                    target_index = [196,415,228,424,794]
                elif self.llm == "YI":
                    target_index = [13420, 29503, 12654, 33328, 53092]
                elif self.llm == "chatGlm":
                    target_index = [316, 347, 319, 367, 397]

                lm_logits_cls_mlp = lm_logits_index[:, target_index]               

                torch.set_printoptions(profile="full")
                lm_logits_cls_binary = lm_logits_cls_mlp

                teacher_logits = inputs['pred']['new_logits'].to(torch.float32)
                T = 2.0
                criterion  = nn.KLDivLoss(reduction="batchmean")#KL散度
                cls_loss = criterion(F.log_softmax(lm_logits_cls_mlp/T, dim=-1).to(self.device),F.softmax(teacher_logits/T, dim=-1).to(self.device)) * (T) ** 2
                loss = 0.5 * expl_loss + 0.5 * cls_loss
                # if self.distill:
                #     teacher_logits = inputs['pred']['logits'].to(torch.float32)
                #     T = 2.0
                #     criterion  = nn.KLDivLoss(reduction="batchmean")#KL散度
                #     distill_loss = criterion(F.log_softmax(lm_logits_cls_mlp/T, dim=-1).to(self.device),F.softmax(teacher_logits/T, dim=-1).to(self.device)) * (T) ** 2
                #     loss = 0.5 * expl_loss + 0.5 * (0.25 * cls_loss + 0.75 * distill_loss)
                # else:
                #     loss = 0.5 * expl_loss + 0.5 * cls_loss
                #     # loss = cls_loss
            else:
                pred_outputs = model(input_ids=inputs['pred']['input_ids'], attention_mask=inputs['pred']['attention_mask'], labels=inputs['pred']['labels'], return_dict=True)
                lm_logits = pred_outputs.logits
                
                if self.take_softmax:
                    log_probs = F.log_softmax(lm_logits, dim=-1) 
                else:
                    log_probs = lm_logits 

                cls_position = inputs['pred']['cls_position']

                batchsize = lm_logits.shape[0]
                labels = inputs['pred']['labels']

                lm_logits_index = log_probs[torch.arange(0 , batchsize), cls_position]
                if "qwen" in self.llm:
                    target_index = [32,33,34,35,36] #"A,B,C,D,E"
                elif self.llm == "glm":
                    target_index = [196,415,228,424,794]
                elif self.llm == "YI":
                    target_index = [13420, 29503, 12654, 33328, 53092]
                elif self.llm == "chatGlm":
                    target_index = [316, 347, 319, 367, 397]
                lm_logits_cls_mlp = lm_logits_index[:, target_index] 

                cls_labels = inputs['pred']['semantic_labels_softmax'].to(torch.float32)

                loss_ce_fn = CrossEntropyLoss()  # target-label onehot or int-tensor / logits not need softmax!
                loss_5cls = loss_ce_fn(input=lm_logits_cls_mlp, target=cls_labels.to(self.device))
                loss_rce = self.get_rce_loss(lm_logits_cls_mlp, cls_labels.to(self.device)) 
                loss = loss_5cls
                # print(lm_logits_cls_mlp[0], cls_labels[0], loss)
                if self.use_sce:
                    loss = 1.0 * loss_5cls + 0.3 * loss_rce
                # llm_loss = pred_outputs.loss
                # loss = 0.5 * loss_5cls + 0.5 * llm_loss
                # llm_loss_rd = torch.distributed.nn.all_reduce(llm_loss) / self.world_size 
                #if torch.distributed.get_rank() == 0:
                #    self.log({"llm_loss": llm_loss_rd.item()})
                #self.log({"loss_5cls": loss_5cls.item()}, sync_dist=True)

                if self.add_rationale_loss:
                    expl_outputs = model(input_ids=inputs['pred']['expl_input_ids'], attention_mask=inputs['pred']['expl_attention_masks'], labels=inputs['pred']['expl_labels'], return_dict=True)
                    expl_loss = expl_outputs.loss
                    expl_loss = torch.where(torch.isnan(expl_loss), 0, expl_loss)
                    loss = 0.5 * expl_loss + 0.5 * loss
        
            return loss

        elif self.official:
            pred_outputs = model(input_ids=inputs['pred']['input_ids'], attention_mask=inputs['pred']['attention_mask'])
            logits_all = pred_outputs.logits
            lm_logits_cls_mlp = logits_all
            # print("train input_ids", inputs['pred']['input_ids'])
            # print("train attention_mask", inputs['pred']['attention_mask'])
            # print("train lm_logits_cls_mlp", lm_logits_cls_mlp)

            loss_ce_fn = CrossEntropyLoss()  # target-label onehot or int-tensor / logits not need softmax!
            if self.loss_type == "RMloss":
                labels = input_ids=inputs['labels']
            else:
                cls_labels = inputs['pred']['semantic_labels_softmax'].to(torch.float32)
                label_index = torch.argmax(cls_labels, -1)


            # #cls_loss = loss_ce_fn(input=lm_logits_cls_mlp.to(torch.float32), target=cls_labels)

            # #binary_label = torch.greater(label_index, 0).to(torch.long)
            # #binary_cls_loss = loss_ce_fn(input=lm_logits_cls_binary.to(torch.float32), target=binary_label)

            # #label_index_relevance = torch.where(label_index > 2, 2, label_index)
            # #relevance_cls_loss = loss_ce_fn(input=lm_logits_cls_binary.to(torch.float32), target=label_index_relevance)

            # if self.loss_type == "OR_loss":
            #     bs = inputs['pred']['input_ids'].shape[0]
            #     target = torch.ones([bs, 4])
            #     for i, v in enumerate(label_index):
            #         target[i, v:] = 0

            #     cls_loss = OR_loss(lm_logits_cls_mlp.to(torch.float32), target.to(torch.float32).to(lm_logits_cls_mlp.device))
            # elif self.loss_type == "order_loss":
            #     bs = inputs['pred']['input_ids'].shape[0]
            #     target = torch.ones([bs, 4])
            #     for i, v in enumerate(label_index):
            #         target[i, v:] = 0

            #     criterion = nn.BCELoss()
            #     cls_loss = criterion(F.sigmoid(lm_logits_cls_mlp[:, 1:].to(torch.float32)), target.to(torch.float32).to(lm_logits_cls_mlp.device))
            # elif self.loss_type == "order_condor_loss":
            #     lm_logits_cls_mlp = logits_all
            #     bs = inputs['pred']['input_ids'].shape[0]
            #     target = torch.ones([bs, 4])
            #     for i, v in enumerate(label_index):
            #         target[i, v:] = 0

            #     cls_loss = CondorOrdinalCrossEntropy(lm_logits_cls_mlp.to(torch.float32), target.to(torch.float32).to(lm_logits_cls_mlp.device))
            # elif self.loss_type == "order_coral_loss":
            #     bs = inputs['pred']['input_ids'].shape[0]
            #     target = torch.ones([bs, 4])
            #     for i, v in enumerate(label_index):
            #         target[i, v:] = 0

            #     if self.IMP_WEIGHT:
            #         imp = torch.tensor([1.3, 1.0, 0.7, 0.6], dtype=torch.float)
            #     else:
            #         imp = torch.ones(4, dtype=torch.float)

            #     #print_rank_0(model.linear_1_bias)
            #     cls_loss = coral_loss(lm_logits_cls_mlp[:, 0:].to(torch.float32), target.to(torch.float32).to(lm_logits_cls_mlp.device), imp.to(lm_logits_cls_mlp.device))
            # elif self.loss_type == "order_coral_loss_v2":
            #     bs = inputs['pred']['input_ids'].shape[0]
            #     target = torch.ones([bs, 4])
            #     for i, v in enumerate(label_index):
            #         target[i, v:] = 0

            #     if self.IMP_WEIGHT:
            #         imp = torch.tensor([0.8, 1.2, 1.5, 1.6], dtype=torch.float)
            #     else:
            #         imp = torch.ones(4, dtype=torch.float)

            #     cls_loss = coral_loss_v2(lm_logits_cls_mlp[:, 1:].to(torch.float32), target.to(torch.float32).to(lm_logits_cls_mlp.device), imp.to(lm_logits_cls_mlp.device))
            # elif self.loss_type == "order_corn_loss":
            #     cls_loss = corn_loss(lm_logits_cls_mlp[:, 1:].to(torch.float32), label_index.to(torch.float32).to(lm_logits_cls_mlp.device), 5)
            # elif self.loss_type == "order_cls2":
            #     ord_loss = OrdinalRegressionLoss(5)
            #     label_index = torch.reshape(label_index, [-1, 1])
            #     cls_loss, likelihoods = ord_loss(lm_logits_cls_mlp[:, :1].to(torch.float32), cls_labels.to(torch.float32).to(lm_logits_cls_mlp.device))
            # elif self.loss_type == "am_softmax":
            #     criterion_xent = AMSoftmax()
            #     cls_loss, output_logits = criterion_xent(lm_logits_cls_mlp.to(torch.float32),  cls_labels.to(torch.float32).to(lm_logits_cls_mlp.device))
            # elif self.loss_type == "RMloss":
            #     criterion = RMLoss(reduction="mean")
            #     k_lens = inputs.pop("k_lens")
            #     #print_rank_0("k_lens")
            #     #print_rank_0(k_lens)
            #     cls_loss = criterion(lm_logits_cls_mlp, torch.reshape(labels, [-1, 1]), k_lens)


            if self.distill:
                teacher_logits = inputs['pred']['logits'].to(torch.float32)
                T = 2.0
                criterion  = nn.KLDivLoss(reduction="batchmean")#KL散度
                distill_loss = criterion(F.log_softmax(lm_logits_cls_mlp/T, dim=-1).to(self.device),F.softmax(teacher_logits/T, dim=-1).to(self.device)) * (T) ** 2
                # # [new] sunliuying reverse kl_div
                # distill_loss = criterion(F.log_softmax(teacher_logits/T, dim=-1).to(self.device), F.softmax(lm_logits_cls_mlp/T, dim=-1).to(self.device)) * (T) ** 2
                # #loss = 0.25 * cls_loss + 0.75 * distill_loss
                loss = distill_loss
            else:
                #loss = cls_loss + 0.7 * binary_cls_loss
                loss = cls_loss
                #loss = relevance_cls_loss 

        # else:
        #     cls_position = inputs['pred']['cls_position']
        #     cls_labels = inputs['pred']['semantic_labels_softmax'].to(torch.float32)
        #     outputs = model(input_ids=inputs['pred']['input_ids'].cuda(),
        #             cls_position=cls_position.cuda(),
        #             semantic_labels=cls_labels.cuda())
        #     loss = outputs[0]
        #     print("cls_loss: {}".format(loss))


        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        if self.generate_cls:
            #pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
            with torch.no_grad():
                #print_rank_0(inputs['pred']['input_ids'])
                labels = inputs["pred"]['label_onehot']
                pred_outputs = model(input_ids=inputs['pred']['input_ids'], attention_mask=inputs['pred']['attention_mask'])
                #loss = pred_outputs.loss
                lm_logits = pred_outputs.logits
                batchsize = lm_logits.shape[0]
                
                if self.take_softmax:
                    #log_probs = F.log_softmax(lm_logits, dim=-1) 
                    log_probs = lm_logits.log_softmax(-1)
                else:
                    log_probs = lm_logits 

                cls_position = inputs['pred']['cls_position']
                #print_rank_0(cls_position)
                lm_logits_index = log_probs[torch.arange(0 , batchsize), cls_position]
                softmaxt_T = 0.9
                if "qwen" in self.llm:
                    target_index = [32,33,34,35,36]
                elif self.llm == "glm":
                    target_index = [196,415,228,424,794]
                elif self.llm == "YI":
                    target_index = [13420, 29503, 12654, 33328, 53092]
                elif self.llm == "chatGlm":
                    target_index = [316, 347, 319, 367, 397]
                lm_logits_cls_mlp = lm_logits_index[:, target_index] 
                torch.set_printoptions(profile="full")
                #print_rank_0(lm_logits_cls_mlp)
                lm_logits_cls_binary = lm_logits_cls_mlp

                cls_labels = inputs['pred']['semantic_labels_softmax'].to(torch.float32)

                loss_ce_fn = CrossEntropyLoss()  # target-label onehot or int-tensor / logits not need softmax!
                loss = loss_ce_fn(input=lm_logits_cls_mlp, target=cls_labels.to(self.device))
        else: 
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    if self.official:
                        if self.loss_type == "RMloss":
                            labels = input_ids=inputs['labels']
                        else:
                            # sunliuying
                            labels = inputs["pred"]['label_onehot']
                            cls_labels = inputs['pred']['semantic_labels_softmax'].to(torch.float32)
                            label_index = torch.argmax(cls_labels, -1)

                            pred_outputs = model(input_ids=inputs['pred']['input_ids'], attention_mask=inputs['pred']['attention_mask'])
                            logits_all = pred_outputs.logits
                            lm_logits_cls_mlp = logits_all 
                            # print("sunliuying debug output logits", lm_logits_cls_mlp)

                            softmaxt_T = 1.0
                            if self.loss_type == "OR_loss":
                                bs = inputs['pred']['input_ids'].shape[0]
                                # target = torch.ones([bs, 4])
                                # for i, v in enumerate(label_index):
                                #     target[i, v:] = 0

                                # loss = OR_loss(logits_all.to(torch.float32), target.to(torch.float32).to(lm_logits_cls_mlp.device))

                                # lm_logits_cls_mlp = logits_all.view(-1, 4, 2)
                                # lm_logits_cls_mlp = F.softmax(lm_logits_cls_mlp, dim=2)[:, :, 1]
                            # elif self.loss_type == "order_corn_loss":
                            #     lm_logits_cls_mlp = logits_all
                            # elif self.loss_type == "order_coral_loss":
                            #     lm_logits_cls_mlp = logits_all[:, 0:]
                            #     bs = inputs['pred']['input_ids'].shape[0]
                            #     target = torch.ones([bs, 4])
                            #     for i, v in enumerate(label_index):
                            #         target[i, v:] = 0

                            #     if self.IMP_WEIGHT:
                            #         imp = torch.tensor([1.3, 1.0, 0.7, 0.5], dtype=torch.float)
                            #     else:
                            #         imp = torch.ones(4, dtype=torch.float)

                            #     loss = coral_loss(lm_logits_cls_mlp.to(torch.float32), target.to(torch.float32).to(lm_logits_cls_mlp.device), imp.to(lm_logits_cls_mlp.device))
                            # elif self.loss_type == "order_condor_loss":
                            #     lm_logits_cls_mlp = logits_all
                            #     bs = inputs['pred']['input_ids'].shape[0]
                            #     target = torch.ones([bs, 4])
                            #     for i, v in enumerate(label_index):
                            #         target[i, v:] = 0

                            #     loss = CondorOrdinalCrossEntropy(lm_logits_cls_mlp.to(torch.float32), target.to(torch.float32).to(lm_logits_cls_mlp.device))
                            # elif self.loss_type == "order_coral_loss_v2":
                            #     lm_logits_cls_mlp = logits_all[:, 1:]
                            #     bs = inputs['pred']['input_ids'].shape[0]
                            #     target = torch.ones([bs, 4])
                            #     for i, v in enumerate(label_index):
                            #         target[i, v:] = 0

                            #     if self.IMP_WEIGHT:
                            #         imp = torch.tensor([0.8, 1.2, 1.4, 1.5], dtype=torch.float)
                            #     else:
                            #         imp = torch.ones(4, dtype=torch.float)

                            #     loss = coral_loss_v2(lm_logits_cls_mlp.to(torch.float32), target.to(torch.float32).to(lm_logits_cls_mlp.device), imp.to(lm_logits_cls_mlp.device))
                            # elif self.loss_type == "order_cls2":
                            #     ord_loss = OrdinalRegressionLoss(5)
                            #     label_index = torch.reshape(label_index, [-1, 1])
                            #     cls_loss, likelihoods = ord_loss(logits_all[:, :1].to(torch.float32), cls_labels.to(torch.float32).to(lm_logits_cls_mlp.device))
                            #     lm_logits_cls_mlp = likelihoods
                            # elif self.loss_type == "order_cls1":
                            #     lm_logits_cls_mlp = logits_all[:, :5]
                            # elif self.loss_type == "am_softmax":
                            #     criterion_xent = AMSoftmax()
                            #     loss, output_logits = criterion_xent(lm_logits_cls_mlp.to(torch.float32),  cls_labels.to(torch.float32).to(lm_logits_cls_mlp.device))
                            #     lm_logits_cls_mlp = 20 * output_logits
                            # elif self.loss_type == "RMloss":
                            #     criterion = RMLoss(reduction="mean")
                            #     k_lens = inputs.pop("k_lens")
                            #     labels = inputs['labels']
                            #     loss = criterion(lm_logits_cls_mlp, labels, k_lens)
                            else:
                                # sunliuying
                                loss_ce_fn = CrossEntropyLoss()  # target-label onehot or int-tensor / logits not need softmax!
                                label_index = torch.reshape(label_index, [-1, 1])
                                # add by sunliuying
                                labels = labels.to(torch.float32)
                                loss = loss_ce_fn(input=lm_logits_cls_mlp, target=labels.to(self.device))
                    else:
                        cls_position = inputs['pred']['cls_position']
                        cls_labels = inputs['pred']['semantic_labels_softmax'].to(torch.float32)
                        outputs = model(input_ids=inputs['pred']['input_ids'].cuda(),
                                cls_position=cls_position.cuda(),
                                semantic_labels=cls_labels.cuda())
                        loss = outputs[0] 
                        lm_logits_cls_mlp = outputs.logits

        #if self.output_rationale:
        #    expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        #else:
        #    expl_outputs = pred_outputs2 # placeholder only
        
        # sunliuying time label
        float_labels = inputs["pred"]["float_label"]

        return (
            loss,
            [lm_logits_cls_mlp, lm_logits_cls_mlp],
            #[inputs["extra"]['label_onehot'].to(self.device), inputs["extra"]['label_onehot'].to(self.device)],
            # [labels.to(self.device), labels.to(self.device)],
            [labels.to(self.device), float_labels.to(self.device)]
        )

class PairwiseTrainer(Trainer):
    def __init__(self, alpha, output_rationale, add_rationale_loss,
            hidden_size, use_mean_embdding, qwen_new, generate_cls, llm, loss_type, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale
        self.add_rationale_loss = add_rationale_loss 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size 

        hidden_dropout = 0.1
        self.official = True 
        self.mean_embedding = use_mean_embdding
        self.qwen_new = qwen_new 
        self.generate_cls = generate_cls
        self.llm = llm
        self.loss_type = loss_type 

    def compute_loss(self, model, inputs, return_outputs=False):

        bs = inputs['pred1']['input_ids'].shape[0]
        input_ids = torch.concat([inputs['pred1']['input_ids'], inputs['pred2']['input_ids']], 0)
        attention_mask = torch.concat([inputs['pred1']['attention_mask'], inputs['pred2']['attention_mask']], 0)
        #labels = torch.concat([inputs['pred1']['labels'], inputs['pred2']['labels']], 0)

        pred_outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        #pred_outputs_neg = model(input_ids=inputs['neg']['input_ids'], attention_mask=inputs['neg']['attention_mask'], output_hidden_states=True, return_dict=True)
    
        logits_pos = pred_outputs.logits[:bs]
        logits_neg = pred_outputs.logits[bs:]
        #print_rank_0(logits_pos)
        #print_rank_0(logits_neg)
        #logits_neg = pred_outputs_neg.logits

        #print(inputs['pos']['input_ids'])
        label_pos = inputs['extra']['label_pos']
        label_neg = inputs['extra']['label_neg']
        #margin = (label_pos - label_neg) / 3.0 

        beta=0.001
        l2 = 0.5 * (logits_pos**2 + logits_neg**2).mean()
        if self.loss_type == "pairwise_loss":
            pair_loss = -nn.functional.logsigmoid(logits_pos - logits_neg).mean() + beta * l2
        elif self.loss_type == "pairwise_margin_loss":
            criterion = nn.MarginRankingLoss(margin=0.85)
            target = torch.ones([bs, 1]).to(logits_pos.device)
            pair_loss = criterion(logits_pos, logits_neg, target)
            pair_loss = pair_loss + beta * l2
        elif self.loss_type == "pairwise_margin_addpoint_loss":
            criterion = nn.MarginRankingLoss(margin=1.0)
            target = torch.ones([bs, 1]).to(logits_pos.device)
            pair_loss = criterion(logits_pos, logits_neg, target)
            mse_loss_f  = nn.MSELoss()
            #print_rank_0(logits_pos.shape)
            #print_rank_0(label_pos.shape)
            label_pos = torch.reshape(label_pos, [-1, 1])
            label_neg = torch.reshape(label_neg, [-1, 1])
            mse_loss = mse_loss_f(logits_pos, label_pos.to(logits_pos.dtype)) + mse_loss_f(logits_neg, label_neg.to(logits_pos.dtype))
            pair_loss = pair_loss + 0.6 * mse_loss
        #print_rank_0(pair_loss)
        #pair_loss = (-1 * nn.functional.logsigmoid(logits_pos - logits_neg) + beta * l2).mean()
        #print(loss)
        #loss = pair_loss + beta * l2
        loss = pair_loss 
        #logits = pred_outputs.logits
        '''
        batchsize = logits.shape[0]
        cls_position = inputs['extra']['cls_position']
        lm_logits_index = logits[torch.arange(0 , batchsize), cls_position]
        softmaxt_T = 0.9
        if self.llm == "qwen":
            #target_index = [32,33,34,35,36]
            target_index = [32, 33] # bad, "fair", "good", "perfect", "awesome"
        lm_logits_cls_mlp = lm_logits_index[:, target_index] 

        #cls_labels = inputs['extra']['semantic_labels_softmax'].to(torch.float32)
        cls_labels = torch.greater(inputs["extra"]['label_neg'], inputs["extra"]['label_pos']).to(torch.long)

        loss_ce_fn = CrossEntropyLoss()  # target-label onehot or int-tensor / logits not need softmax!
        cls_loss = loss_ce_fn(input=lm_logits_cls_mlp, target=cls_labels.to(self.device))
        #print(lm_logits_cls_mlp)
        #print(cls_labels)
        print(cls_loss)
        '''
        #llm_loss = self.label_smoother(pred_outputs, inputs['pred']['labels'])
        #llm_loss = pred_outputs.loss 
        #loss = 1.0 * llm_loss 
        #print(llm_loss)

        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        with torch.no_grad():
            bs = inputs['pred1']['input_ids'].shape[0]
            input_ids = torch.concat([inputs['pred1']['input_ids'], inputs['pred2']['input_ids']], 0)
            attention_mask = torch.concat([inputs['pred1']['attention_mask'], inputs['pred2']['attention_mask']], 0)

            pred_outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            #pred_outputs = model(input_ids=inputs['pos']['input_ids'], attention_mask=inputs['pos']['attention_mask'], output_hidden_states=True, return_dict=True)
            #pred_outputs_neg = model(input_ids=inputs['neg']['input_ids'], attention_mask=inputs['neg']['attention_mask'],output_hidden_states=True,  return_dict=True)
    
            logits_pos = pred_outputs.logits[:bs]
            logits_neg = pred_outputs.logits[bs:]

            label_pos = inputs['extra']['label_pos']
            label_neg = inputs['extra']['label_neg']
            margin = (label_pos - label_neg) / 3.0
            #loss = -nn.functional.logsigmoid(logits_pos - logits_neg - margin).mean()
            loss = -nn.functional.logsigmoid(logits_pos - logits_neg).mean()
            #print(loss)
            '''
            bs = inputs['pred1']['input_ids'].shape[0]
            input_ids = torch.concat([inputs['pred1']['input_ids'], inputs['pred2']['input_ids']], 0)
            attention_mask = torch.concat([inputs['pred1']['attention_mask'], inputs['pred2']['attention_mask']], 0)
            labels = torch.concat([inputs['pred1']['labels'], inputs['pred2']['labels']], 0)

            pred_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

            logits = pred_outputs.logits
            batchsize = logits.shape[0]
            cls_position = torch.concat([inputs['pred1']['cls_position'], inputs['pred1']['cls_position']], 0)
            lm_logits_index = logits[torch.arange(0 , batchsize), cls_position]
            softmaxt_T = 0.9
            if self.llm == "qwen":
                #target_index = [32,33,34,35,36]
                target_index = [32, 33] # bad, "fair", "good", "perfect", "awesome"
            lm_logits_cls_mlp = lm_logits_index[:, target_index] 
            lm_logits_cls_binary = lm_logits_cls_mlp
            loss = pred_outputs.loss
            '''
        return (
            loss,
            [logits_pos, logits_neg],
            [label_pos, label_neg]
            #[torch.concat([inputs["extra"]['label_pos'], inputs["extra"]['label_neg']], 0).to(self.device), torch.concat([inputs["extra"]['label_neg'], inputs["extra"]['label_pos']], 0).to(self.device)],
        )



class TaskPrefixSeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(self, alpha, output_rationale, add_rationale_loss,
            hidden_size, use_mean_embdding, qwen_new, generate_cls, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.output_rationale = output_rationale
        self.add_rationale_loss = add_rationale_loss 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hidden_size = hidden_size 

        def cls_mlp():
            mlp_layer = torch.nn.Sequential(torch.nn.Linear(hidden_size , 1024) ,  #
                                            # torch.nn.BatchNorm1d(4096, affine=True),
                                            torch.nn.ReLU(True) ,  # torch.nn.Tanh()
                                            torch.nn.Linear(1024 , 256) ,
                                            torch.nn.ReLU(True) ,  # torch.nn.Tanh()
                                            torch.nn.Linear(256 , 5))
            return mlp_layer
        self.cls_mlp = cls_mlp()
        self.cls_mlp.to(self.device)

        num_class = 5
        hidden_dropout = 0.1
        self.pool_layer = torch.nn.Linear(64000, hidden_size)
        self.multichoice_dropout = torch.nn.Dropout(hidden_dropout)
        self.multichoice_head = torch.nn.Linear(hidden_size, num_class)
        self.official = True 
        self.mean_embedding = use_mean_embdding
        self.distill = True

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.official:
            attention_mask = inputs['pred']['attention_mask']

            #pred_outputs = model(input_ids=inputs['pred']['input_ids'], output_hidden_states=True)
            if self.mean_embedding:
                pred_outputs = model(**inputs['pred'], output_hidden_states=True, return_dict=True)  
                #pred_outputs = model(input_ids=inputs['pred']['input_ids'], decoder_input_ids=inputs['pred']['input_ids'][:, :2],
                #        output_hidden_states=True, return_dict=True)
                lm_logits_cls_mlp = pred_outputs.logits
                #print(last_hidden_state)
                loss_ce_fn = CrossEntropyLoss()  # target-label onehot or int-tensor / logits not need softmax!
                cls_labels = inputs['extra']['semantic_labels_softmax'].to(torch.float32)
                cls_loss = loss_ce_fn(input=lm_logits_cls_mlp, target=cls_labels)
                if self.distill:
                    teacher_logits = inputs['extra']['logits'].to(torch.float32)
                    T = 2.0
                    criterion  = nn.KLDivLoss(reduction="batchmean")#KL散度
                    distill_loss = criterion(F.log_softmax(lm_logits_cls_mlp/T, dim=-1).to(self.device),F.softmax(teacher_logits/T, dim=-1).to(self.device)) * (T) ** 2
                    final_loss = 0.25 * cls_loss + 0.75 * distill_loss
                else:
                    final_loss = cls_loss

            else:
                pred_outputs = model(**inputs['pred'])
                lm_logits_cls_mlp = pred_outputs.logits
                #lm_logits_index = logits[:, 0]
                #lm_logits_cls_mlp = lm_logits_index[:, [15,16,17,18,19]]
                mt5_loss = pred_outputs.loss
                cls_loss = mt5_loss 
                #return loss

                #batch_ids = torch.arange(logits.size(0), dtype=attention_mask.dtype, device=attention_mask.device)
                #cls_position = inputs['extra']['cls_position'].to(attention_mask.dtype)
                #target_logits = logits[batch_ids, cls_position].to(self.device)
                #lm_logits_cls_mlp = self.cls_mlp(target_logits.to(torch.float32)).to(torch.float32)  # [batchsize, 3]

            #output = torch.tanh(self.pool_layer(target_logits))
            #multichoice_output = self.multichoice_dropout(output)
            #lm_logits_cls_mlp = self.multichoice_head(multichoice_output)
            
            #print(cls_labels)

            loss = final_loss

        else:
            cls_position = inputs['extra']['cls_position']
            cls_labels = inputs['extra']['semantic_labels_softmax'].to(torch.float32)
            outputs = model(input_ids=inputs['pred']['input_ids'].cuda(),
                    cls_position=cls_position.cuda(),
                    semantic_labels=cls_labels.cuda())
            loss = outputs[0]
            print("cls_loss: {}".format(loss))

        #return (loss, {'pred': pred_outputs, 'expl': pred_outputs}) if return_outputs else loss
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        #pred_outputs2 = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        
        with torch.no_grad():
            with self.compute_loss_context_manager():
                if self.official:
                    attention_mask = inputs['pred']['attention_mask']

                    if self.mean_embedding:
                        pred_outputs = model(**inputs['pred'], output_hidden_states=True, return_dict=True)
                        #last_hidden_state = pred_outputs.hidden_states[-1] 
                        lm_logits_cls_mlp = pred_outputs.logits 

                        cls_labels = inputs['extra']['semantic_labels_softmax'].to(lm_logits_cls_mlp.dtype)
                        loss_ce_fn = CrossEntropyLoss()  # target-label onehot or int-tensor / logits not need softmax!
                        loss = loss_ce_fn(input=lm_logits_cls_mlp, target=cls_labels.to(self.device))
                    else:
                        #pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
                        pred_outputs = model(**inputs['pred'])
                        logits = pred_outputs.logits
                        lm_logits_cls_mlp = pred_outputs.logits
                        lm_logits_index = logits[:, 0]
                        softmaxt_T = 0.9
                        #lm_logits_cls_mlp = lm_logits_index[:, [460, 353, 338, 328, 410]] / softmaxt_T
                        #lm_logits_cls_mlp = lm_logits_index[:, [497, 333, 356, 381, 419]] / softmaxt_T
                        lm_logits_cls_mlp = lm_logits_index[:, [1073, 94, 99, 100, 138]] / softmaxt_T
                        all_loss = pred_outputs.loss

                        cls_labels = inputs['extra']['semantic_labels_softmax'].to(lm_logits_cls_mlp.dtype)

                        loss_ce_fn = CrossEntropyLoss()  # target-label onehot or int-tensor / logits not need softmax!
                        loss = loss_ce_fn(input=lm_logits_cls_mlp, target=cls_labels.to(self.device))
                        print("all_loss: {}, cls_loss: {}".format(all_loss, loss))
                        #batch_ids = torch.arange(logits.size(0), dtype=attention_mask.dtype, device=attention_mask.device)
                        #cls_position = inputs['extra']['cls_position'].to(attention_mask.dtype)
                        #target_logits = logits[batch_ids, cls_position].to(self.device)
                        #lm_logits_cls_mlp = self.cls_mlp(target_logits.to(torch.float32)).to(torch.float32)  # [batchsize, 3]

                    #inputs['pred']['labels'] = inputs['pred']['labels'].to(self.device)
                    #pred_outputs = model(input_ids=inputs['pred']['input_ids'].to(self.device), output_hidden_states=True)

                    #logits = pred_outputs.hidden_states[-1]
                    #logits = pred_outputs[1][-1]
                else:
                    cls_position = inputs['extra']['cls_position']
                    cls_labels = inputs['extra']['semantic_labels_softmax'].to(torch.float32)
                    outputs = model(input_ids=inputs['pred']['input_ids'].cuda(),
                            cls_position=cls_position.cuda(),
                            semantic_labels=cls_labels.cuda())
                    loss = outputs[0] 
                    lm_logits_cls_mlp = outputs.logits

        #if self.output_rationale:
        #    expl_outputs = super().prediction_step(model, inputs['expl'], prediction_loss_only=False, ignore_keys=ignore_keys)
        #else:
        #    expl_outputs = pred_outputs2 # placeholder only

        return (
            loss,
            [lm_logits_cls_mlp, lm_logits_cls_mlp],
            [inputs["extra"]['label_onehot'].to(self.device), inputs["extra"]['label_onehot'].to(self.device)],
        )

