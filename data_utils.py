
import os
import argparse
import re
import json
import numpy as np
import torch

from datasets import Dataset, DatasetDict, load_dataset
from urllib import parse


DATASET_ROOT = 'datasets'
# DATASET_ROOT = '/mnt/data/liuhuanhuan/qtc-llm-wt/qtc-llm/datasets'
DATASET_ROOT = '/mnt/data/lhh/liuhuanhuan/distill_process'


class DatasetLoader(object):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, test_data=None, is_streaming=False, valid_batch_idxs=None):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.source_dataset_name = source_dataset_name
        self.dataset_version = dataset_version
        self.has_valid = has_valid
        self.split_map = split_map

        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs
        self.is_streaming = is_streaming 
        self.test_data = test_data
        
        assert self.split_map is not None    


    def load_from_source(self):
        if self.source_dataset_name is None:
            self.source_dataset_name = self.dataset_name
        if self.dataset_version is None:
            datasets = load_dataset(self.source_dataset_name)
        else:
            datasets = load_dataset(self.source_dataset_name, self.dataset_version)
        return datasets


    def to_json(self, datasets):
        for k, v in self.split_map.items():
            datasets[v].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json')


    def load_from_json(self):
        data_files = {
            'train': os.environ['PRIMUS_DATA_PATH'] + '*',
            'test': '/'.join(os.environ['PRIMUS_TEST_DATA_PATH'].split(' ')[0].split('/')[:-1]) + "/*"
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',})

        print(data_files)
        if self.is_streaming:
            datasets = load_dataset('json', data_files=data_files, streaming=True)
            print('datasets: ', datasets)
            datasets = datasets.shuffle(buffer_size=100_000, seed=42)
        else:
            datasets = load_dataset('json', data_files=data_files, num_proc=1)
        datasets = self._post_process(datasets) 

        # subsample training dataset if needed
        #num_train = len(datasets['train'])
        idxs = list()
        #for idx in self.train_batch_idxs:
        #    idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)        
        #datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])

        return datasets


    def load_llm_preds(self, split):
        labels = list()
        rationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                outputs = json.load(f)

            for output in outputs:
                rationale, label = self._parse_llm_output(output)

                rationales.append(rationale)
                labels.append(label)

        return rationales, labels


    def load_gpt_preds(self, split):
        labels = list()
        rationales = list()
        
        with open(f'{self.data_root}/gpt-neox/{self.dataset_name}/{split}.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_gpt_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels


    def _post_process(self, datasets):
        raise NotImplementedError


    def _parse_llm_output(self, output):
        raise NotImplementedError


    def _parse_gpt_output(self, output):
        raise NotImplementedError



class QTCDatasetLoader(DatasetLoader):
    def __init__(self, data_version, test_data=None, is_streaming=False, subset='full'):
        dataset_name = data_version 
        source_dataset_name = data_version
        dataset_version = None
        has_valid = False 
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 55500
        if subset == 'full':
            train_batch_idxs = range(5100)
        elif subset == 'small':
            train_batch_idxs = range(10)
        else:
            raise ValueError
        test_batch_idxs = range(22)
        valid_batch_idxs = range(22)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, test_data, is_streaming, valid_batch_idxs=valid_batch_idxs)


    def _post_process(self, datasets):
        
        def prepare_input(example):
            example['idx'] = 0
            # add by sunliuying 
            if 'logits' not in example:
                example['logits'] = [0.0, 0.0, 0.0, 0.0, 0.0]
            # 增加时效特征
            if "elapsedMonth" in example:
                example["elapsedDays"] = int(example["elapsedMonth"])
            if "elapsedDays" not in example:
                example["elapsedDays"] = 38
            if example["elapsedDays"] == -999:
                example["elapsedDays"] = 38

            example['new_logits'] = example['logits']
            for k in ['wenda_summary', 'pr_score', 'tokens', 'weights']:
                example[k] = 0
            if 'summary' not in example:
                example['summary'] = ''
            if (not example['summary']) and 'term_summary' in example:
                example['summary'] = example['term_summary']
            if (not example['summary']) and 'sumary' in example:
                example['summary'] = example['sumary']
            
            example['term_summary'] = ''
            example['sumary'] = ''
            # 注意一些特征的处理，可能有坑
            if 'filename' not in example or example['filename'] is None:
                example['filename'] = 'cot_distill'
            if 'pred_label' in example:
                example['label'] = example['pred_label']
            if 'multi_label' not in example:
                ml = [0, 0, 0, 0, 0]
                ml[int(float(example['label']))] = 7
                example['multi_label'] = ','.join([str(item) for item in ml])
            if 'response' not in example:
                example['response'] = example['label']

            selected_cols = ['query', 'url', 'title', 'summary', 'body', 'label', 'filename', 
                        'response', 'logits', 'multi_label', 'probs', 'new_logits']
            selected_cols.append("elapsedDays")
            for key in example.keys():
                if key not in selected_cols:
                    example[key] = 0
            scores = example['multi_label']

            '''
            example['label'] = example['passParams']['label']
            example['query'] = example['passParams']['query']
            example['url'] = example['passParams']['url']
            example['scores'] = example['passParams']['scores']
            scores = example['passParams']['scores']
            '''

            semantic_labels = [float(i) for i in scores.split(',')]
            semantic_labels_log1p = np.log1p(semantic_labels)
            semantic_labels_log1p_scale = []
            for i in semantic_labels_log1p:
                if i == 0.0:
                    semantic_labels_log1p_scale.append(-20000.0)
                else:
                    semantic_labels_log1p_scale.append(i * 2.0)
            semantic_labels_softmax = np.exp(semantic_labels_log1p_scale)/sum(np.exp(semantic_labels_log1p_scale))
            example['semantic_labels_softmax'] = semantic_labels_softmax

            label_raw = float(example['label'])
            example['float_label'] = label_raw
            label = int(label_raw)
            if label == 0:
                example['label'] = "A"
                example['label_onehot'] = [1, 0, 0, 0, 0]
            elif label == 1:
                example['label'] = "B"
                example['label_onehot'] = [0, 1, 0, 0, 0]
            elif label == 2:
                example['label'] = "C"
                example['label_onehot'] = [0, 0, 1, 0, 0]
            elif label == 3:
                example['label'] = "D"
                example['label_onehot'] = [0, 0, 0, 1, 0]
            elif label == 4:
                example['label'] = "E"
                example['label_onehot'] = [0, 0, 0, 0, 1]
            elif label == 5:
                example['label'] = "E"
                example['label_onehot'] = [0, 0, 0, 0, 1]
            elif label == -1:
                example['label'] = "A"
                example['label_onehot'] = [1, 0, 0, 0, 0]
            else:
                example['label'] = "A"
                example['label_onehot'] = [1, 0, 0, 0, 0]

            # example['rationale'] = "fake"
            # 上面就有对response的赋值，重复赋值可能会存在问题，导致该列值类型不一致报错“pyarrow.lib.ArrowInvalid: Could not convert 'A' with type str: tried to convert to double”
            # if 'response' in example and not example['response']:
            #     example['response'] = example['label']
            # print(example)
            return example

        if self.is_streaming:
            datasets = datasets.map(prepare_input)
        else:
            datasets = datasets.map(prepare_input, num_proc=32)
        # datasets = datasets.remove_columns(['logits', 'multi_label'])
        # datasets = datasets.remove_columns(['passParams', 'category',
        #    'modelParams', 'tags', 'taskId', 'id', 'model'])
        # datasets = datasets.rename_column('prompt', 'input')

        datasets = datasets.rename_column('response', 'rationale')

        return datasets

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == 'cqa':
        dataset_loader = CQADatasetLoader()
    elif args.dataset == 'svamp':
        dataset_loader = SVAMPDatasetLoader()
    elif args.dataset == 'esnli':
        dataset_loader = ESNLIDatasetLoader()
    elif args.dataset == 'anli1':
        dataset_loader = ANLI1DatasetLoader()

    datasets = dataset_loader.load_from_source()
    dataset_loader.to_json(datasets)
