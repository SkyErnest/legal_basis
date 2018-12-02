import re,math,json
import numpy as np
import jieba
import tensorflow as tf
from collections import Iterable

def law_to_list(path):
    with open(path,'r',encoding='utf-8') as f:
        law=[]
        for line in f:
            if line=='\n' or re.compile(r'第.*[节|章]').search(line[:10]) is not None:
                continue
            try:
                tmp=re.compile(r'第.*条').search(line.strip()[:8]).group(0)
                law.append(line.strip())
            except (TypeError,AttributeError):
                law[-1]+=line.strip()
    return law

def cut_law(law_list, filter=None,cut_sentence=False):
    res=[]
    for each in law_list:
        index,content=each.split('　')
        index=hanzi_to_num(index[1:-1])
        charge,content=content[1:].split('】')
        # if charge[-1]!='罪':
        #     continue
        if filter is not None and index not in filter:
            continue
        if cut_sentence:
            context, n_words = [], []
            for i in content.split('。'):
                if i != '':
                    context.append(list(jieba.cut(i)))
                    n_words.append(len(context[-1]))
        else:
            context=list(jieba.cut(content))
            n_words=len(context[-1])
        res.append([index,charge,context,n_words])
    return res

def hanzi_to_num(hanzi):
    # for num<10000
    hanzi=hanzi.strip().replace('零', '')
    if(hanzi[0])=='十': hanzi = '一'+hanzi
    d = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,'':0}
    m = {'十':1e1,'百':1e2,'千':1e3}
    res = 0
    tmp=0
    for i in range(len(hanzi)):
        if hanzi[i] in d:
            tmp+=d[hanzi[i]]
        else:
            tmp*=m[hanzi[i]]
            res+=tmp
            tmp=0
    return int(res+tmp)

def load_data(prefix='train'):
    dicts, main_data = [], []
    with open('data/data_{}.json'.format(prefix), 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            dicts.append(data)
            main_data.append([data['fact'], data['meta']['accusation'], data['meta']['relevant_articles']])
    return dicts, main_data

def get_tensor_shape(T):
    return T.get_shape().as_list()

def cut_data(data, cut_sentence=False):
    context,accu_label,law_label,n_words=[],[],[],[]
    for each in data:
        if cut_sentence:
            sent_words, sent_n_words = [], []
            for i in each[0].split('。'):
                if i!='':
                    sent_words.append(list(jieba.cut(i)))
                    sent_n_words.append(len(sent_words[-1]))
            context.append(sent_words)
            n_words.append(sent_n_words)
        else:
            context.append(list(jieba.cut(each[0])))
            n_words.append(len(context[-1]))
        accu_label.append([i.replace('[','').replace(']','') for i in each[1]])
        law_label.append(each[2])
    return context,accu_label,law_label,n_words

# def cut_data_in_sentence(law_data):
#     context,label,n_sents,n_words=[],[],[],[]
#     for each in law_data:
#         sent_words,sent_n_words=[],[]
#         for i in each[0].split('。'):
#             sent_words.append(list(jieba.cut(i)))
#             sent_n_words.append(len(sent_words[-1]))
#         context.append(sent_words)
#         n_sents.append(len(context[-1]))
#         n_words.append(sent_n_words)
#         label.append(each[1])
#     return context,label,n_words,n_sents

def lookup_index(x,word2id,doc_len):
    res=[]
    for each in x:
        tmp=[word2id['BLANK']]*doc_len
        for i in range(len(each)):
            if i>=doc_len:
                break
            try:
                tmp[i]=word2id[each[i]]
            except KeyError:
                tmp[i]=word2id['UNK']
        res.append(tmp)
    return np.array(res)

def lookup_index_for_sentences(x,word2id,doc_len,sent_len):
    res=[]
    for each in x:
        # tmp = [[word2id['BLANK']] * sent_len for _ in range(doc_len)]
        tmp = lookup_index(each, word2id, sent_len)[:doc_len]
        tmp=np.pad(tmp,pad_width=[[0,doc_len-len(tmp)],[0,0]],mode='constant')
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

def attention(Q,K,V):
    #Q ...*N*F
    #K ...*M*F
    #V ...*M*L
    res_map=Q@tf.transpose(K,[-1,-2])/tf.sqrt(K.get_shape().as_list()[-1])
    res=tf.nn.softmax(res_map)@V
    return res

def multihead_atten(Q,K,V,F_=None,L_=None,num_attention_heads=1,initializer_range=.02):
    #Q ...*N*F
    #K ...*M*F
    #V ...*M*F
    def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                             seq_length, width):
        output_tensor = tf.reshape(
            input_tensor, [batch_size, seq_length, num_attention_heads, width])

        output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
        return output_tensor

    Q_shape=get_tensor_shape(Q)
    K_shape=get_tensor_shape(K)
    V_shape=get_tensor_shape(V)
    initializer=tf.truncated_normal_initializer(initializer_range)
    if F_ is None:
        F_ = Q_shape[-1]
    if L_ is None:
        L_ = V_shape[-1]
    Q_=tf.layers.dense(Q,num_attention_heads*F_,activation=None,kernel_initializer=initializer)
    K_=tf.layers.dense(K,num_attention_heads*F_,activation=None,kernel_initializer=initializer)
    V_=tf.layers.dense(V,num_attention_heads*L_,activation=None,kernel_initializer=initializer)

    Q_=transpose_for_scores(Q_,Q_shape[0],num_attention_heads,Q_shape[-2],F_)
    K_=transpose_for_scores(K_,K_shape[0],num_attention_heads,K_shape[-2],F_)
    V_=transpose_for_scores(V_,V_shape[0],num_attention_heads,V_shape[-2],L_)

    return tf.reshape(tf.transpose(attention(Q_,K_,V_),[0,2,1,3]),[V_shape[0],Q_shape[1],num_attention_heads*L_])


def index_to_label(index,batch_size):
    # convert batch index of one-hot to label
    res=[[] for i in range(batch_size)]
    for each in index:
        res[int(each[0])].append(int(each[1]))
    return res

def flatten(items):
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x

def align_flatten2d(items,align_len,flatten=True):
    res=[]
    for each in items:
        each=each[:align_len]
        res.append(np.pad(each,[0,align_len-len(each)],'constant'))
    res=np.array(res)
    if flatten:
        res=res.flatten()
    return res


def trun_n_words(n_words,sent_len):
    for i in range(len(n_words)):
        for j in range(len(n_words[i])):
            if n_words[i][j]>sent_len:
                n_words[i][j]=sent_len
    return n_words


def find_1_in_one_hot(matrix,f):
    for each in matrix:
        for i in range(len(each)):
            if each[i]==1:
                f(i)

if __name__=='__main__':
    law_list=law_to_list('data/criminal_law.txt')
    laws=cut_law(law_list)