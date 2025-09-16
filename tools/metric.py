#encoding=utf-8

from __future__ import division
from __future__ import print_function
from unicodedata import name

import numpy as np
from tools.dcg import DcgMetric
from utils import print_rank_0


class VarMetric:
    def __init__(self, name):
        self.name = name
        self.values = []
        self.var = 0.
    def calc(self, values, masks):
        cnt = 0
        for val, mask in zip(values, masks):
            if not mask:
                continue
            cnt += 1
            self.values.append(val)
        if cnt > 0:
            self.var = float(np.var(self.values))
        return self.var

class PairAccMetric:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.cnt = 0.

    def calc(self, pos, neg, querys, masks):
        for pos_val, neg_val, mask in zip(pos, neg, masks):
            if not mask:
                continue
            self.total += 1
            # if float(pos_val.item()) > float(neg_val.item()):
            # if pos_val.eval() > neg_val.eval():
            if pos_val > neg_val:
                self.cnt += 1
        avg = self.cnt / (1e-8 + self.total)
        print("{}: {}/{}={}".format(self.name, self.cnt, self.total, avg))
        return avg

class MeanMetric:
    def __init__(self, name):
        self.name = name
        self.total = 0
        self.cnt = 0
    def calc(self, values, masks):
        for val, mask in zip(values, masks):
            if not mask:
                continue
            self.total += val
            self.cnt += 1
        avg = self.total / (1e-6 + self.cnt)
        return avg
class PrecisionRecallMetric:
    def __init__(self,name):
        self.name=name
        self.acc=0.
        self.positive_scores=[]
        self.negative_scores=[]
    
    def calc(self, labels, scores, querys, urls, masks):
        for label, score, query, url, mask in zip(labels, scores, querys, urls, masks):
            if not mask:
                continue
            if label==0:
                self.negative_scores.append(score)
            else:
                self.positive_scores.append(score)
        negative_scores_sort=sorted(self.negative_scores)
        threshold=0.5
        if len(negative_scores_sort)!=0:
            
            target_item=int(float(len(negative_scores_sort))*0.9)
            threshold=negative_scores_sort[target_item]
        correct_num=0
        total_num=0
        for item in self.negative_scores:
            if item<threshold:
                correct_num+=1
                total_num+=1
            else:
                total_num+=1
        for item in self.positive_scores:
            if item >=threshold:
                correct_num+=1
                total_num+=1
            else:
                total_num+=1
        if total_num==0:
            self.acc=0.
        else:
            self.acc=float(correct_num)/float(total_num)
        if "precision_recall" in self.name:
            info = ['']
            info.append('==================== ' + self.name)
            info.append('labels = ' + str(labels[:5]))
            info.append('scores = ' + str(scores[:5]))
            info.append('threshod = ' + str(threshold))

            info.append('correct_num[%d] / total[%d] = [%f]' % (int(correct_num), total_num, self.acc))
            info.append('')
            print ('\n'.join(info))
        return self.acc


class SeqPairMetric:
    def __init__(self, name):
        self.query_map = {}
        self.query_counter = {}
        self.query_counter_hard = {}
        self.name = name
        self.qu_set = set()
        self.acc = 0.
        self.acc_hard = 0.
        self.total_score = 0
        self.counter_total = 0
        self.counter_total_hard = 0
        self.counter_pos = 0.
        self.counter_pos_hard = 0.
    
    def calc(self, labels, scores, querys, urls, masks):
        query_update = set()
        valid = 0
        for label, score, query, url, mask in zip(labels, scores, querys, urls, masks):
            #print(query+"\t"+str(url)+"\t"+str(mask.astype(object)))
            if not mask:
                continue
            query_url = query + '_**_' + str(url)
            if url != 1 and query_url in self.qu_set:
                continue
            self.total_score += 1
            valid += 1
            self.qu_set.add(query_url)
            query_update.add(query)
            if query not in self.query_map:
                self.query_map[query] = {}
            if label not in self.query_map[query]:
                self.query_map[query][label] = []

            self.query_map[query][label].append(score)
            #self.query_map[query][label].append(l)
        if valid == 0:
            return self.acc, self.acc_hard
        total, pos, acc, acc_hard = self.calc_impl(query_update)
        #if 'pair_acc' in self.name:
        info = ['']
        info.append('==================== ' + self.name)
        info.append('total_score = ' + str(self.total_score))
        info.append('labels = ' + str(labels[:5]))
        info.append('scores = ' + str(scores[:5]))
        info.append('qu_cnt = ' + str(len(self.qu_set)))
        info.append('pos[%d] / total[%d] = [%f], hard_acc: [%f]' % (int(pos), total, acc, acc_hard))
        info.append('')
        print_rank_0('\n'.join(info))
        self.acc = acc
        self.acc_hard = acc_hard
        return acc, acc_hard
    
    def calc_impl(self, query_update):
        for query in query_update:
            label_score = self.query_map[query]
            label_score = sorted(label_score.items())
            #label_score = sorted(label_score.items(), key=lambda x:eval(x[0]), reverse=True)
            if query in self.query_counter: # minus previous value
                x, y = self.query_counter[query]
                self.counter_pos = self.counter_pos - x
                self.counter_total = self.counter_total - y
            if query in self.query_counter_hard: # minus previous value
                x, y = self.query_counter_hard[query]
                self.counter_pos_hard = self.counter_pos_hard - x
                self.counter_total_hard = self.counter_total_hard - y
            q_pos = 0
            q_pos_hard = 0
            q_total = 0
            for i in range(len(label_score)):
                for j in range(i + 1, len(label_score)):
                    for i_score in label_score[i][1]:
                        for j_score in label_score[j][1]:
                            if i_score + 1e-6 < j_score:
                                q_pos += 1
                            if i_score + 0.1 < j_score:
                                q_pos_hard += 1
                            q_total += 1
            self.query_counter[query] = (q_pos, q_total)
            self.counter_total += q_total
            self.counter_pos += q_pos

            self.query_counter_hard[query] = (q_pos_hard, q_total)
            self.counter_total_hard += q_total
            self.counter_pos_hard += q_pos_hard
        if self.counter_total == 0:
            return self.counter_total, self.counter_pos, 0., 0.
        return self.counter_total, self.counter_pos, self.counter_pos / self.counter_total, self.counter_pos_hard / self.counter_total_hard 

    def calc_confidence(self, labels, scores, querys, urls, masks):
        query_update = set()
        valid = 0
        for label, score, query, url, mask in zip(labels, scores, querys, urls, masks):
            if not mask:
                continue
            query_url = query + '_**_' + str(url)
            if url != 1 and query_url in self.qu_set:
                continue
            self.total_score += 1
            valid += 1
            self.qu_set.add(query_url)
            query_update.add(query)
            if query not in self.query_map:
                self.query_map[query] = {}
            if label not in self.query_map[query]:
                self.query_map[query][label] = []
            self.query_map[query][label].append(score)
        if valid == 0:
            return self.acc
        total, pos, acc = self.calc_impl_confidence(query_update)
        if 'pair_acc' in self.name:
            info = ['']
            info.append('==================== ' + self.name)
            info.append('total_score = ' + str(self.total_score))
            info.append('labels = ' + str(labels[:5]))
            info.append('scores = ' + str(scores[:5]))
            info.append('qu_cnt = ' + str(len(self.qu_set)))
            info.append('pos[%d] / total[%d] = [%f]' % (int(pos), total, acc))
            info.append('')
            print('\n'.join(info))
        self.acc = acc
        return acc

    def calc_impl_confidence(self, query_update):
        for query in query_update:
            label_score = self.query_map[query]
            label_score = sorted(label_score.items())
            if query in self.query_counter:  # minus previous value
                x, y = self.query_counter[query]
                self.counter_pos = self.counter_pos - x
                self.counter_total = self.counter_total - y
            q_pos = 0
            q_total = 0
            for i in range(len(label_score)):
                for j in range(i + 1, len(label_score)):
                    for i_score in label_score[i][1]:
                        for j_score in label_score[j][1]:
                            if i_score + 0.01 < j_score: # margin
                                q_pos += 1
                            q_total += 1
            self.query_counter[query] = (q_pos, q_total)
            self.counter_total += q_total
            self.counter_pos += q_pos
        if self.counter_total == 0:
            return self.counter_total, self.counter_pos, 0.
        return self.counter_total, self.counter_pos, self.counter_pos / self.counter_total
    

class UserMetric:
    @staticmethod
    def seq_pair(labels, predictions, querys, name='pair_acc', urls=None, masks=None):
        '''
        labels = tf.reshape(labels, [-1])
        predictions = tf.reshape(predictions, [-1])
        querys = tf.reshape(querys, [-1])
        if urls is None:
            urls = tf.ones_like(querys, tf.int32)
        else:
            urls = tf.reshape(urls, [-1])
        if masks is None:
            masks = tf.ones_like(querys, tf.int32)
        else:
            masks = tf.cast(masks, tf.int32)
            masks = tf.reshape(masks, [-1])
        '''
        num = len(predictions)
        if masks is None:
            masks = [1] * num
        if urls is None:
            urls = [1] * num
        metric = SeqPairMetric(name)
        args = [labels, predictions, querys, urls, masks]
        #update_op = tf.py_func(metric.calc, args, tf.float64)
        acc, acc_hard = metric.calc(labels, predictions, querys, urls, masks)
        return acc, acc_hard

    @staticmethod
    def dcg(labels, predictions, querys, urls=None, masks=None, dcg_pos=3, dcg_type='dcg', name='dcg'):
        num = len(predictions)
        if masks is None:
            masks = [1] * num
        if urls is None:
            urls = [1] * num
        metric = DcgMetric(dcg_pos, dcg_type, name)
        args = [querys, urls, predictions, labels, masks]
        update_op = metric.calc(querys, urls, predictions, labels, masks) 
        return update_op

    '''
    @staticmethod
    def seq_pair_confidence(labels, predictions, querys, name='pair_acc_confidence', urls=None, masks=None):
        labels = tf.reshape(labels, [-1])
        predictions = tf.reshape(predictions, [-1])
        querys = tf.reshape(querys, [-1])
        if urls is None:
            urls = tf.ones_like(querys, tf.int32)
        else:
            urls = tf.reshape(urls, [-1])
        if masks is None:
            masks = tf.ones_like(querys, tf.int32)
        else:
            masks = tf.cast(masks, tf.int32)
            masks = tf.reshape(masks, [-1])
        metric = SeqPairMetric(name)
        args = [labels, predictions, querys, urls, masks]
        update_op = tf.py_func(metric.calc_confidence, args, tf.float64)
        return update_op

    @staticmethod
    def pair_acc(predictions, querys, name='pair_acc', masks=None):
        predictions = tf.reshape(predictions, [-1])
        pos, neg = tf.split(predictions, axis=0, num_or_size_splits=2)
        pos = tf.reshape(pos, [-1])
        neg = tf.reshape(neg, [-1])

        querys = tf.reshape(querys, [-1])
        if masks is None:
            masks = tf.ones_like(querys, tf.int32)
        else:
            masks = tf.cast(masks, tf.int32)
            masks = tf.reshape(masks, [-1])
        metric = PairAccMetric(name)
        args = [pos, neg, querys, masks]
        update_op = tf.py_func(metric.calc, args, tf.float64)
        return update_op
    
    
    @staticmethod
    def calc_mean(values, masks, name):
        if masks is None:
            masks = tf.ones_like(values, tf.int32)
        values = tf.reshape(values, [-1])
        masks = tf.reshape(masks, [-1])
        metric = MeanMetric(name)
        args = [values, masks]
        update_op = tf.py_func(metric.calc, args, tf.float64)
        return update_op
    
    @staticmethod
    def calc_var(values, masks, name):
        if masks is None:
            masks = tf.ones_like(values, tf.int32)
        values = tf.reshape(values, [-1])
        masks = tf.reshape(masks, [-1])
        metric = VarMetric(name)
        args = [values, masks]
        update_op = tf.py_func(metric.calc, args, tf.float64)
        return update_op
    @staticmethod
    def calc_precison_recall(labels, predictions, querys, name='precision_recall', urls=None, masks=None):
        labels = tf.reshape(labels, [-1])
        predictions = tf.reshape(predictions, [-1])
        querys = tf.reshape(querys, [-1])
        if urls is None:
            urls = tf.ones_like(querys, tf.int32)
        else:
            urls = tf.reshape(urls, [-1])
        if masks is None:
            masks = tf.ones_like(querys, tf.int32)
        else:
            masks = tf.cast(masks, tf.int32)
            masks = tf.reshape(masks, [-1])
        metric = PrecisionRecallMetric(name)
        args = [labels, predictions, querys, urls, masks]
        update_op = tf.py_func(metric.calc, args, tf.float64)
        return update_op

def test_placeholder():
    print ('************ test_placeholder')
    labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])
    predictions = tf.placeholder(dtype=tf.int32, shape=[None, 1])
    querys = tf.placeholder(dtype=tf.string, shape=[None])
    update_op = UserMetric.seq_pair(labels, predictions, querys)
    with tf.Session() as sess:
        print (sess.run([update_op],
                        feed_dict={
                            querys: ['a', 'a'],
                            labels: [[1], [2]],
                            predictions: [[1], [3]]}))
        print (sess.run([update_op],
                        feed_dict={
                            querys: ['a', 'a'],
                            labels: [[2], [3]],
                            predictions: [[3], [2]]}))
        print (sess.run([update_op],
                        feed_dict={
                            querys: ['a'],
                            labels: [[4]],
                            predictions: [[1]]}))
        print ('=====================')

def test_one(querys, labels, predictions):
    update_op = UserMetric.seq_pair(labels, predictions, querys)
    with tf.Session() as sess:
        print (sess.run([update_op]))

def tets_seq_pair():
    querys = ['a', 'a', 'a', 'a', 'a']
    
    print ('************ 1')
    labels = [[1], [2], [2], [3], [4]]
    predictions = [[1], [3], [3], [2], [1]]
    test_one(querys, labels, predictions)
    
    print ('************ 2')
    labels = [[4], [3], [2], [2], [1]]
    predictions = [[4], [3], [3], [2], [1]]
    test_one(querys, labels, predictions)
    
    print ('************ 3')
    labels = [[4], [3], [2], [2], [1]]
    predictions = [[3], [5], [4], [2], [1]]
    test_one(querys, labels, predictions)
    
    print ('************ 4')
    labels = [[4], [3], [5], [2], [1]]
    predictions = [[3], [5], [1], [2], [1]]
    test_one(querys, labels, predictions)
    
    
if __name__ == '__main__':
    tets_seq_pair()
    test_placeholder()
'''
