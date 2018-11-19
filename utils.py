import re,math,json
import numpy as np
import jieba

def law_to_list(path):
    with open(path,'r',encoding='utf-8') as f:
        law=[]
        for line in f:
            if line=='\n' or re.compile(r'第.*[节|章]').search(line[:10]) is not None:
                continue
            try:
                tmp=re.compile(r'第.*条').search(line.strip()[:8])[0]
                law.append(line.strip())
            except TypeError:
                law[-1]+=line.strip()
    return law

def cut_law(law_list):
    res=[]
    for each in law_list:
        index,content=each.split('　')
        index=hanzi_to_num(index[1:-1])
        charge,content=content[1:].split('】')
        content=list(jieba.cut(content))
        res.append([index,charge,content])
    return res

def hanzi_to_num(hanzi):
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,'':0}
    tmp = re.compile(r'[百|十|零]').split(hanzi.strip())
    res = 0
    for i in range(len(tmp)):
        res += d[tmp[i]] * math.pow(10, len(tmp) - i -1)
    return int(res)

def load_data(prefix='train'):
    dicts, accu_data ,law_data = [], [], []
    with open('data/data_{}.json'.format(prefix), 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            dicts.append(data)
            accu_data.append([data['fact'], data['meta']['accusation']])
            law_data.append([data['fact'], data['meta']['relevant_articles']])
    return dicts,accu_data,law_data

def cut_data(law_data):
    context,label,n_words=[],[],[]
    for each in law_data:
        context.append(' '.join(jieba.cut(each[0])))
        n_words.append(len(context[-1]))
        label.append(each[1])
    return context,label,n_words

def lookup_index(x,word2id,doc_len):
    res=[]
    for each in x:
        tmp=[word2id['BLANK']]*doc_len
        words=each.split()
        for i in range(len(words)):
            if i>=doc_len:
                break
            try:
                tmp[i]=word2id[words[i]]
            except KeyError:
                tmp[i]=word2id['UNK']
        res.append(tmp)
    return np.array(res)

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
	Generates a batch iterator for a dataset.
	"""
    data = np.array(data)
    data_size = len(data)
    # Original
    # num_batches_per_epoch = (int)(round(len(data)/batch_size))
    num_batches_per_epoch = int(((len(data)-.5)/batch_size)+1)
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__=='__main__':
    law_list=law_to_list('刑法.txt')
    laws=cut_law(law_list)