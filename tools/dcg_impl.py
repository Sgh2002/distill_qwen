#coding=utf-8

from math import log

class DCG:
    def __init__(self, name, pos, dcg_mode, irrelevant_zero):
        self.name = name
        self.pos = pos
        self.dcg_mode = dcg_mode
        self.query__result_cache = {} # cache the previous result
        if irrelevant_zero:
            self.irval = 0
        else:
            self.irval = 1
        self.result = 0.0
    
    def batch_calc(self, incr_query_dcg_map):
        for query, label_list in incr_query_dcg_map.items():
            self.calc(query, label_list)
        query_num = len(self.query__result_cache)
        # if query_num==1466:
        #     info = ['']
        #     for query,gain in self.query__result_cache.iteritems():  
        #         tmp_info=['']
        #         tmp_info.append('****:')
        #         tmp_info.append(query)
        #         tmp_info.append(str(gain))
        #         tmp_info.append('')
        #         info.append('\t'.join(tmp_info))
                
        #     print('\n'.join(info))
        ans = self.result / max(1, query_num)
        return ans
    
    def calc(self, query, rels):
        gain, ideal_gain = self.calc_impl(rels)
        if self.dcg_mode == 'dcg':
            self.calc_dcg(query, gain)
        elif self.dcg_mode=='ndcg':
            self.calc_ndcg(query, gain, ideal_gain)
        elif self.dcg_mode=='idcg':
            self.calc_idcg(query,ideal_gain)

    def calc_impl(self, rels):
        gain = 0.0
        ideal_gain = 0.0
        ideal = sorted(rels)[::-1] # reverse
        for i, rel in enumerate(rels):
            if i + 1 > self.pos:
                break
            gain += (float)(2 ** rel - 1) / log(i + 2)
            ideal_gain += (float)(2 ** ideal[i] - 1) / log(i + 2)
        return gain, ideal_gain
    
    def calc_dcg(self, query, gain):
        if query in self.query__result_cache:
            self.result -= self.query__result_cache[query]
        self.result += gain
        self.query__result_cache[query] = gain

    def calc_idcg(self,query,idel_gain):
        if query in self.query__result_cache:
            self.result -= self.query__result_cache[query]
        self.result += idel_gain
        self.query__result_cache[query] = idel_gain

    
    def calc_ndcg(self, query, gain, ideal_gain):
        if query in self.query__result_cache:
            self.result -= self.query__result_cache[query]
        if ideal_gain != 0:
            gain = gain / ideal_gain
        else:
            gain = self.irval
        self.result += gain
        self.query__result_cache[query] = gain
