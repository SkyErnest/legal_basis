import tensorflow as tf
import utils
from model import Model,ModelConfig
from sklearn.preprocessing import MultiLabelBinarizer
import json
import numpy as np
import pickle,os,time

num_epochs=31
lr=5e-3
print_epoch=1
law_path='data/刑法.txt'
vec_path='data/words.vec'
emb_path='data/few_shot_emb.npy'
w2id_path='data/w2id.pkl'
# vec_path='E:\\iCloudDrive\\Projects\\GitHub\\attribute_charge\\attribute_charge\\data\\words.vec'
# emb_path='E:\\iCloudDrive\\Projects\\GitHub\\attribute_charge\\attribute_charge\\few_shot_emb.npy'
# w2id_path='E:\\iCloudDrive\\Projects\\GitHub\\attribute_charge\\attribute_charge\\w2id.pkl'
data_path='data/legal_data.pkl'

model_config=ModelConfig()

with open('data/law.txt','r') as f:
    law_class=[int(i) for i in f.read().split('\n')[:-1]]

def process(param='train'):
    dicts_train, accu_data_train, law_data_train = utils.load_data(param)
    x, law, n_words = utils.cut_data(law_data_train)

    x=utils.lookup_index(x,word2id,model_config.doc_len)
    # x=[[word2id[j] for j in i.split()] for i in x]
    batches = list(zip(x,law,n_words))
    print(param,'loaded')
    return batches

def load_data(path):
    if os.path.exists(path):
        batches_train,batches_val,batches_test=pickle.load(open(path,'rb'))
    else:
        batches_train = process('train')
        batches_val = process('valid')
        batches_test = process('test')
        # batches_train=batches_val=batches_test = process('test')
        pickle.dump([batches_train,batches_val,batches_test],open(path,'wb'))
    global train_step_per_epoch,val_step_per_epoch,test_step_per_epoch
    train_step_per_epoch= int((len(batches_train) - .1) / model_config.batch_size) + 1
    val_step_per_epoch= int((len(batches_val) - .1) / model_config.batch_size) + 1
    test_step_per_epoch= int((len(batches_test) - .1) / model_config.batch_size) + 1
    batches_train = utils.batch_iter(batches_train,model_config.batch_size,num_epochs)
    batches_val = utils.batch_iter(batches_val,model_config.batch_size,num_epochs)
    batches_test = utils.batch_iter(batches_test,model_config.batch_size,num_epochs)

    law_list = utils.law_to_list(law_path)
    laws = utils.cut_law(law_list)
    laws=list(zip(*laws))
    laws_len=np.array([len(i) for i in laws[-1]])
    laws = utils.lookup_index(laws[-1], word2id, model_config.law_doc_len)

    return batches_train,batches_val,batches_test,laws,laws_len

def load_embeddings(vec_path):
    word_embeddings = []
    word2id = {}
    with open(vec_path, "r", encoding='utf-8') as f:
        _ = f.readline()
        while True:
            content = f.readline()
            if content == "":
                break
            content = content.split(' ')[1:-1]
            word2id[content[0]] = len(word2id)
            content = [(float)(i) for i in content]
            word_embeddings.append(content)
            word2id['UNK'] = len(word2id)
            word2id['BLANK'] = len(word2id)
            lists = [0.0 for i in range(len(word_embeddings[0]))]
            word_embeddings.append(lists) #UNK
            word_embeddings.append(lists) #BLANK
            word_embeddings = np.array(word_embeddings, dtype='float32')
    return word_embeddings,word2id

def load_embeddings2(emb_path,w2id_path):
    emb=np.load(emb_path)
    w2id=pickle.load(open(w2id_path,'rb'))
    return emb,w2id

def get_feed_dict(batch):
    while len(batch) < model_config.batch_size:
        batch = np.concatenate([batch, batch[:model_config.batch_size - len(batch)]])
    x, law, n_words = list(zip(*batch))
    law = MultiLabelBinarizer(classes=law_class).fit_transform(law)
    feed_dict = {train_model.input: x,
                 train_model.law_label: law,
                 train_model.keep_prob: 1.,
                 train_model.input_length: n_words
                 }
    return feed_dict

def evaluate():
    law_pred=[]
    count=0
    for batch in batches_val:
        count += 1
        feed_dict=get_feed_dict(batch)
        law_pred_b=sess.run([train_model.law_pred],feed_dict=feed_dict)
        law_pred+=[[law_class[j] for j in i] for i in utils.index_to_label(law_pred_b, model_config.batch_size)][:len(batch)]
        if count==val_step_per_epoch:
            break


def chime(step, start):
    duration=time.time()-start
    start=time.time()
    print('step:', step, 'duration:', duration, 's')
    return start


# word_embeddings, word2id=load_embeddings(vec_path)
word_embeddings, word2id=load_embeddings2(emb_path,w2id_path)
batches_train,batches_val,batches_test,laws,laws_len=load_data(data_path)

train_model=Model(model_config,word_embeddings=word_embeddings,law_input=laws,law_length=laws_len)
global_step=tf.Variable(0,trainable=False)
optimizer=tf.train.AdamOptimizer(lr)
train_op=optimizer.minimize(train_model.loss_total,global_step=global_step)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess=tf.Session(config=tf_config)
# sess=tf.Session()
sess.run(tf.global_variables_initializer())
start = time.time()

for batch in batches_train:
    feed_dict=get_feed_dict(batch)
    _,loss_law,loss_total,law_output,chapter_output,output,step,loss_law2=sess.run([train_op,train_model.loss_law,train_model.loss_total,train_model.law_output,train_model.chapter_output,train_model.output,global_step,train_model.loss_law2],feed_dict=feed_dict)
    if step%100==0:
        start=chime(step,start)
        print('loss_law',loss_law,'loss_total',loss_total,'loss_law2:',loss_law2)
    if (step+1-train_step_per_epoch)%(train_step_per_epoch*print_epoch)==0:
        start=chime(step,start)
        print('epoch:', step // train_step_per_epoch)
        evaluate()