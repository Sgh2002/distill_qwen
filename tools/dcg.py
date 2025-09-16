#coding=utf-8
from tools.dcg_impl import DCG

class DcgMetric:
    def __init__(self, dcg_pos, dcg_mode, name):
        self.query_map = {}
        self.dcg = DCG(name, dcg_pos, dcg_mode, irrelevant_zero=True)
    
    def calc(self, querys, urls, scores, labels, masks):
        incr_querys = set()
        for query, url, score, label, mask in zip(querys, urls, scores, labels, masks):
            if not mask:
                continue
            incr_querys.add(query)
            if query not in self.query_map:
                self.query_map[query] = []
            self.query_map[query].append((score, label, query, url))
        
        incr_query_dcg_map = {}
        for query in incr_querys:
            value_list = self.query_map[query]
            if len(value_list) >= 3:
                value_list.sort(key=lambda x: x[0], reverse=True)
                label_list = [v[1] for v in value_list]
                incr_query_dcg_map[query] = label_list
        
        return self.dcg.batch_calc(incr_query_dcg_map)

def check(querys, urls, scores, labels, dcg_type):
    metric = DcgMetric(1, dcg_type)
    print (dcg_type + '1 = ', metric.calc(querys, urls, scores, labels))
    
    metric = DcgMetric(1, dcg_type)
    idx = 101
    metric.calc(querys[:idx], urls[:idx], scores[:idx], labels[:idx])
    print (dcg_type + '1 = ', metric.calc(querys[idx:], urls[idx:], scores[idx:], labels[idx:]))
    
    
    metric = DcgMetric(3, dcg_type)
    print (dcg_type + '3 = ', metric.calc(querys, urls, scores, labels))
    
    metric = DcgMetric(3, dcg_type)
    idx = 1001
    metric.calc(querys[:idx], urls[:idx], scores[:idx], labels[:idx])
    print (dcg_type + '3 = ', metric.calc(querys[idx:], urls[idx:], scores[idx:], labels[idx:]))
    
    
    metric = DcgMetric(20, dcg_type)
    print (dcg_type + '20 = ', metric.calc(querys, urls, scores, labels))
    
    metric = DcgMetric(20, dcg_type)
    idx = 991
    metric.calc(querys[:idx], urls[:idx], scores[:idx], labels[:idx])
    print (dcg_type + '20 = ', metric.calc(querys[idx:], urls[idx:], scores[idx:], labels[idx:]))

def test():
    f = 'pos/sm1.txt'
    querys, urls, scores, labels = [], [], [], []
    with open(f, 'r') as fin:
        for data in fin.read().split('\n')[1:]:
            ts = data.split('^--jinru.yjr--^')
            if len(ts) != 4:
                continue
            querys.append(ts[0])
            urls.append(ts[1])
            scores.append(float(ts[2]))
            labels.append(float(ts[3]))
    
    check(querys, urls, scores, labels, 'dcg')
    check(querys, urls, scores, labels, 'ndcg')
    

if __name__ == '__main__':
    test()
