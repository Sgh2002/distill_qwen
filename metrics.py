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


import numpy as np
import torch
from tools import metric
from scipy.special import softmax, expit 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from utils import print_rank_0

def compute_text_acc(preds, labels):
    return np.mean(np.array(preds) == np.array(labels))


def compute_equation_acc(preds, labels):
    preds = [eval_equation(pred) for pred in preds]
    labels = [eval_equation(label) for label in labels]

    return np.mean(np.array(preds) == np.array(labels))


def eval_equation(equation):
    try:
        answer = eval(equation)
    except:
        answer = np.nan

    return answer

def add_acc(metrics, true_labels, predict_scores, querys, urls, data_type, cal_type):

    _mask = np.equal(data_type, cal_type)
    pairacc, pairacc_hard = metric.UserMetric.seq_pair(true_labels, predict_scores,
            querys, urls=urls, name=cal_type, masks=_mask)
    dcg3 = metric.UserMetric.dcg(true_labels, predict_scores, querys, masks=_mask, dcg_pos=3, dcg_type='dcg', name=cal_type+'_dcg')
    ndcg3 = metric.UserMetric.dcg(true_labels, predict_scores, querys, masks=_mask, dcg_pos=3, dcg_type='ndcg', name=cal_type+'_ndcg')

    metrics[cal_type + "_pairacc"] = pairacc 
    metrics[cal_type + "_pairacc_hard"] = pairacc_hard 
    metrics[cal_type + "_dcg3"] = dcg3 
    metrics[cal_type + "_ndcg3"] = ndcg3 
    print_rank_0("{} dcg3 : {}, ndcg3: {}".format(cal_type, dcg3, ndcg3))
 

def compute_metrics_pairwise(tokenizer, data_type):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        pred_logits = logits[0]
        logits_softmax = softmax(logits[0], axis=1)
        predictions = np.argmax(logits[0], axis=-1)

        predictions = np.greater(logits[0], logits[1]).astype(np.int32)
        predictions_hard = np.greater(logits[0], logits[1] + 0.1).astype(np.int32)
        true_labels = np.greater(labels[0], labels[1]).astype(np.int32)
        #print(predictions.shape)
        #print(true_labels.shape)
        #print(pred_logits[:10])
        #print(predictions[:20])
        #print(true_labels[:20])
        #print(true_labels)
        metrics = {}
        pair_acc = np.mean(np.array(true_labels) == np.array(predictions))
        hard_pair_acc = np.mean(np.array(true_labels) == np.array(predictions_hard))
        metrics["acc"] = pair_acc
        metrics["hard_acc"] = hard_pair_acc

        for dt in set(data_type):
            #print(dt)
            tl, pr, pr_hard = [], [], []
            for i in range(len(data_type)):
                if data_type[i] == dt:
                    tl.append(true_labels[i])
                    pr.append(predictions[i]) 
                    pr_hard.append(predictions_hard[i]) 

            pair_acc = np.mean(np.array(pr) == np.array(tl))
            hard_pair_acc = np.mean(np.array(pr_hard) == np.array(tl))
            metrics[dt.replace("\n", "") + "_pairacc"] = pair_acc
            metrics[dt.replace("\n", "") + "_hard_pairacc"] = hard_pair_acc
        return metrics 
    return compute_metrics

def compute_metrics_cls_acc(tokenizer, querys, urls, data_type, loss_type):
    def compute_metrics(eval_pred):
        # logits, labels = eval_pred
        # true_labels = np.argmax(labels[0], axis=-1)
        logits, all_labels = eval_pred
        labels = all_labels[0]
        float_labels = all_labels[1]
        true_labels = np.argmax(labels, axis=-1)
        # # true_labels = float_labels.astype(np.int64).squeeze()
        # print(type(float_labels), float_labels.shape, float_labels) # ndarray
        # print(type(true_labels), true_labels.shape, true_labels)
        
        if loss_type == "OR_loss":
            probas = logits[0]
            predict_levels = probas > 0.5
            predictions = np.sum(predict_levels, -1)
            predict_scores = np.sum(probas * [1,2,3,4], -1)
        # elif loss_type == "order_corn_loss":
        #     probas = expit(logits[0][:, 1:])
        #     probas2 = np.cumprod(probas, axis=1)
        #     predict_levels = probas2 > 0.5
        #     predictions = np.sum(predict_levels, -1)
        #     predict_scores = np.sum(probas * [1,2,3,4], -1)
        # elif loss_type == "order_condor_loss":
        #     probas = expit(logits[0])
        #     probas2 = np.cumprod(probas, axis=1)
        #     predict_levels = probas2 > 0.5
        #     predictions = np.sum(predict_levels, -1)
        #     predict_scores = np.sum(probas2 * [1,2,3,4], -1)
        # elif loss_type == "order_coral_loss":
        #     probas = expit(logits[0])
           
        #     #thresholds = np.array([0.5, 0.5, 0.5, 0.5])
        #     thresholds = np.array([0.136, 0.337, 0.505, 0.759])
        #     predict_levels = probas > thresholds 
        #     predictions = np.sum(predict_levels, -1)

        #     predict_scores = np.sum(probas * [1,2,3,4], -1)
        # elif loss_type == "order_coral_loss_v2":
        #     #th = np.array([-1.386, -0.405, 0.405, 1.386])
        #     th = np.array([-0.3, -0.1, 0.1, 0.3])
        #     probas = expit(logits[0])
           
        #     thresholds = np.array([0.2, 0.4, 0.6, 0.8])
        #     predict_levels = probas > thresholds 
        #     predictions = np.sum(predict_levels, -1)

        #     predict_scores = np.sum(probas * [1,2,3,4], -1)
        # elif "RMloss" == loss_type:
        #     predict_scores = logits[0]
        #     predictions = np.argmax(logits[0], axis=-1)
        #     true_labels = labels[0]
        # elif "pairwise" in loss_type:
        #     predict_scores = logits[0]
        #     predictions = np.argmax(logits[0], axis=-1)
        #     true_labels = labels[0]
        else:
            logits_softmax = softmax(logits[0], axis=1)
            predictions = np.argmax(logits[0], axis=-1)

            predict_scores = logits_softmax * [0,1,2,3,4]
            predict_scores = np.sum(predict_scores, -1)
            # print("logits", logits[0])
            # print("predictions", predictions)
            # print("predict_scores", predict_scores)
            
        acc = np.mean(np.array(predictions) == np.array(true_labels))
        metrics = {}
        metrics["accuracy"] = acc

        for dt in set(data_type):
            print_rank_0(dt)
            tl, pr = [], []
            for i in range(len(data_type)):
                if data_type[i] == dt:
                    tl.append(true_labels[i])
                    pr.append(predictions[i]) 

            acc = np.mean(np.array(pr) == np.array(tl))
            t = classification_report(tl, pr)
            print_rank_0(t)
            print_rank_0(confusion_matrix(tl, pr))
            metrics[dt.replace("\n", "") + "_acc"] = acc

        logits_softmax_binary = softmax(logits[1], axis=1)
        predictions_binary = np.argmax(logits[1], axis=-1)

        #binary_labels = np.greater(true_labels, 0).astype(np.int32)
        #binary_acc = np.mean(np.array(predictions_binary) == np.array(binary_labels))
        #metrics["binary_accuracy"] = binary_acc
        #t2 = classification_report(binary_labels, predictions_binary)
        #print_rank_0(t2)

        '''
        labels_relevance = np.where(true_labels > 2, 2, true_labels)
        predict_scores_relevance = np.sum(softmax(logits[1], axis=1) * [0,1,2], -1)
        t3 = classification_report(labels_relevance, predictions_binary)
        print_rank_0(t3)
        '''

        t = classification_report(true_labels, predictions)
        print_rank_0(t)
        print_rank_0(confusion_matrix(true_labels, predictions))

        for dt in set(data_type):

            # if dt not in ["fresh_test", "hq_250410_new", "scold_0417", "auth_test", "pingce_2412_2504_merge"]:
            #     continue

            #print_rank_0("labels")
            #print_rank_0(labels[0])
            true_labels = np.sum(labels[0] * [0,1,2,3,4], axis=-1)
            #print_rank_0("true_labels")
            #print_rank_0(true_labels)
            add_acc(metrics, float_labels, predict_scores, querys, urls, data_type, dt)

        return metrics
        #return {'accuracy': acc, 'qc_new_pairacc': qc_new_pairacc, 'qc_new_pairacc_hard': qc_new_pairacc_hard,
        #        'qc_bad_pairacc': qc_bad_pairacc, 'qc_bad_pairacc_hard': qc_bad_pairacc_hard} 

    return compute_metrics 

def compute_metrics_text(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        #print("compute_metrics")
        #print(decoded_labels)
        #print(decoded_preds)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics


def compute_metrics_text_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        acc = np.mean(np.array(decoded_preds) == np.array(decoded_labels))

        return {'accuracy': acc}

    return compute_metrics



def compute_metrics_equation(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:    
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:    
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}
    
    return compute_metrics


def compute_metrics_equation_aux(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = list()
        for pred in decoded_preds:    
            preds.append(eval_equation(pred))

        labels = list()
        for label in decoded_labels:    
            labels.append(eval_equation(label))

        acc = np.mean(np.array(preds) == np.array(labels))

        return {'accuracy': acc}
    
    return compute_metrics
