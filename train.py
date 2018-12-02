import tensorflow as tf
import utils
from model import Model,ModelConfig
from sklearn.preprocessing import MultiLabelBinarizer
import json
import numpy as np
import pickle,os,time
from judger import Judger


num_epochs=31
print_epoch=1
restore=True
law_path='data/criminal_law.txt'
vec_path='data/words.vec'
emb_path='data/few_shot_emb.npy'
w2id_path='data/w2id.pkl'
# vec_path='E:\\iCloudDrive\\Projects\\GitHub\\attribute_charge\\attribute_charge\\data\\words.vec'
# emb_path='E:\\iCloudDrive\\Projects\\GitHub\\attribute_charge\\attribute_charge\\few_shot_emb.npy'
# w2id_path='E:\\iCloudDrive\\Projects\\GitHub\\attribute_charge\\attribute_charge\\w2id.pkl'
data_path='data/legal_data.pkl'
checkpoint_dir='model_save'
restore_path=None
save_path='model_save/legal_basis-time-{}-{}-{}'.format(time.localtime(time.time()).tm_mon, time.localtime(time.time()).tm_mday, time.localtime(time.time()).tm_hour)
# NO SPACE in path !!!!!!!!!!!!!!!!!!!
model_config=ModelConfig()
lr=model_config.learning_rate

if restore and restore_path is None:
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt:
        restore_path=ckpt.model_checkpoint_path
print('restore_path:',restore_path)

with open('data/accu.txt','r',encoding='utf-8')as f:
    accu_class=[i for i in f.read().split('\n')[:-1]]

with open('data/law.txt','r') as f:
    law_class=[int(i) for i in f.read().split('\n')[:-1]]
    file_order={law_class[i]:i+1 for i in range(len(law_class))}
    law_class=sorted(law_class)
    # score corresponding to law_set, law_label fed into model correspoding to law_class

def process(param='train'):
    dicts_train, data_train = utils.load_data(param)
    x, accu, law, n_words = utils.cut_data(data_train,cut_sentence=True)
    n_sent=[len(i) if len(i)<model_config.doc_len else model_config.doc_len for i in n_words]
    n_words=utils.trun_n_words(n_words,model_config.sent_len)
    n_words=utils.align_flatten2d(n_words,model_config.doc_len,flatten=False)
    x=utils.lookup_index_for_sentences(x,word2id,model_config.doc_len,model_config.sent_len)
    # x=[[word2id[j] for j in i.split()] for i in x]
    batches = list(zip(x,accu,law,n_sent,n_words))
    print(param,'loaded')
    return batches

def load_data(path):
    if os.path.exists(path):
        batches_train,batches_val,batches_test=pickle.load(open(path,'rb'))
    else:
        batches_train = process('train')
        batches_val = process('valid')
        batches_test = process('test')
        # batches_train=batches_val=batches_test = process('valid')
        pickle.dump([batches_train,batches_val,batches_test],open(path,'wb'))
    global train_step_per_epoch,val_step_per_epoch,test_step_per_epoch
    train_step_per_epoch= int((len(batches_train) - .1) / model_config.batch_size) + 1
    val_step_per_epoch= int((len(batches_val) - .1) / model_config.batch_size) + 1
    test_step_per_epoch= int((len(batches_test) - .1) / model_config.batch_size) + 1
    batches_train = utils.batch_iter(batches_train,model_config.batch_size,num_epochs)
    batches_val = utils.batch_iter(batches_val,model_config.batch_size,num_epochs,shuffle=False)
    batches_test = utils.batch_iter(batches_test,model_config.batch_size,num_epochs,shuffle=False)

    law_list = utils.law_to_list(law_path)
    laws = utils.cut_law(law_list,filter=law_class,cut_sentence=True)
    # pickle.dump(law_list,open('data/law.pkl','wb'))

    model_config.n_law=len(laws)
    laws=list(zip(*laws))
    # pickle.dump({laws[1][i]:laws[0][i] for i in range(len(laws[0]))},open('data/accu2law_dict.pkl','wb'))
    # law_set=laws[0]
    laws_doc_len=[len(i) if len(i)<model_config.law_doc_len else model_config.law_doc_len for i in laws[-1]]
    laws_sent_len=utils.trun_n_words(laws[-1],model_config.law_sent_len)
    laws_sent_len=utils.align_flatten2d(laws_sent_len,model_config.law_doc_len,flatten=False)
    laws = utils.lookup_index_for_sentences(laws[-2], word2id, model_config.law_doc_len,model_config.law_sent_len)

    return batches_train,batches_val,batches_test,laws,laws_doc_len,laws_sent_len

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
    x, accu, law, n_sent,n_words = list(zip(*batch))
    accu = MultiLabelBinarizer(classes=accu_class).fit_transform(accu)
    law = MultiLabelBinarizer(classes=law_class).fit_transform(law)
    feed_dict = {train_model.input: x,
                 train_model.label: accu,
                 train_model.law_label : law,
                 train_model.keep_prob: 1.,
                 train_model.input_sent_length: n_words,
                 train_model.input_doc_length: n_sent
                 }
    return feed_dict

def evaluate():
    accu_pred,law_pred=[],[]
    ground_truth=[]
    count=0
    for batch in batches_val:
        count += 1
        feed_dict=get_feed_dict(batch)
        law_score,law_pred_b,accu_pred_b,loss=sess.run([train_model.law_score,
                                                        train_model.law_prediction,
                                                        train_model.prediction,
                                                        train_model.loss],
                                                       feed_dict=feed_dict)
        if count%100==0:
            print('valid_step:',count,'valid loss:',loss)
        # accu_pred+= [[accu_class[j] for j in i] for i in utils.index_to_label(accu_pred_b, model_config.batch_size)][:len(batch)]
        accu_pred+= [[j+1 for j in i] for i in utils.index_to_label(accu_pred_b, model_config.batch_size)][:len(batch)]
        law_pred+=law_pred_b.tolist()
        ground_truth+=list(zip(feed_dict[train_model.label].tolist(),feed_dict[train_model.law_label].tolist()))
        # if count%10==0:
        #     break
        if count==val_step_per_epoch:
            break

    with open('data/valid_label.txt','w',encoding='utf-8') as f:
        for each in ground_truth:
            for i in range(len(each[0])):
                if each[0][i] ==1:
                    f.write(str(accu_class[i]))
            for i in range(len(each[1])):
                if each[1][i] ==1:
                    f.write(', '+str(law_class[i]))
            f.write('\n')

    with open('data/data_valid_predict.json', 'w',encoding='utf-8') as f:
        for i in range(len(accu_pred)):
            rex = {"accusation": [], "articles": [], "imprisonment": 0}
            rex["accusation"]=accu_pred[i]
            for each in law_pred[i]:
                # each is the index of law predicted in law_class
                if each > 0:
                    rex["articles"].append(file_order[law_class[int(each)]])
            print(json.dumps(rex,ensure_ascii=False),file=f)
            # print(rex)
            # f.write('{{"accusation": [0], "articles": {}, "imprisonment": 0}}'.format(law_pred[i]))
    J = Judger('data/accu.txt', 'data/law.txt')
    res = J.test('data/data_valid.json', 'data/data_valid_predict.json')
    total_score = 0
    scores=[]
    for task_idx in range(2):
        TP_micro = 0
        FP_micro = 0
        FN_micro = 0
        f1 = []
        for class_idx in range(len(res[task_idx])):
            if res[task_idx][class_idx]["TP"] == 0:
                f1.append(0)
                continue
            TP_micro += res[task_idx][class_idx]["TP"]
            FP_micro += res[task_idx][class_idx]["FP"]
            FN_micro += res[task_idx][class_idx]["FN"]
            precision = res[task_idx][class_idx]["TP"] * 1.0 / (res[task_idx][class_idx]["TP"] + res[task_idx][class_idx]["FP"])
            recall = res[task_idx][class_idx]["TP"] * 1.0 / (res[task_idx][class_idx]["TP"] + res[task_idx][class_idx]["FN"])
            f1.append(2 * precision * recall / (precision + recall))
        precision_micro = TP_micro * 1.0 / (TP_micro + FP_micro+1e-6)
        recall_micro = TP_micro * 1.0 / (TP_micro + FN_micro+1e-6)
        F1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro+1e-6)
        F1_macro = np.sum(f1) / len(f1)
        total_score += 100.0 * (F1_micro + F1_macro)/2
        print('task id: {}, F1_micro: {}, F1_macro: {}, final score: {}'.format(task_idx + 1, F1_micro, F1_macro, 100.0 * (F1_micro + F1_macro)/2))
        scores.append([F1_micro,F1_macro])
    total_score += res[2]['score'] / res[2]['cnt'] * 100
    print('task id: 3, score:{}'.format(res[2]['score'] / res[2]['cnt'] * 100))
    print('total score:', total_score)
    return total_score,scores


def chime(step, start):
    duration=time.time()-start
    start=time.time()
    print('step:', step, 'duration:', duration, 's')
    return start


# word_embeddings, word2id=load_embeddings(vec_path)
word_embeddings, word2id=load_embeddings2(emb_path,w2id_path)
batches_train,batches_val,batches_test,laws,laws_doc_len,laws_sent_len=load_data(data_path)

train_model=Model(model_config,word_embeddings=word_embeddings,law_input=laws,law_doc_length=laws_doc_len,law_sent_length=laws_sent_len)
global_step=tf.Variable(0,trainable=False)
optimizer=tf.train.AdamOptimizer(lr)
# optimizer=tf.train.GradientDescentOptimizer(lr)
train_op=optimizer.minimize(train_model.loss,global_step=global_step)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
sess=tf.Session(config=tf_config)
saver=tf.train.Saver()
# sess=tf.Session()
sess.run(tf.global_variables_initializer())

if restore:
    saver.restore(sess,restore_path)
    print('model loaded from',restore_path)

start = time.time()
max_total_score=0.

# evaluate()

for batch in batches_train:
    feed_dict=get_feed_dict(batch)
    _,loss_main,loss_law,loss,output,step=sess.run([train_op,train_model.loss_main,train_model.loss_law,train_model.loss,train_model.outputs,global_step],feed_dict=feed_dict)
    if step%100==0:
        start=chime(step,start)
        print('loss_main',loss_main,'loss_law',loss_law,'loss',loss)
    if (step-train_step_per_epoch)%(train_step_per_epoch*print_epoch)==0:
        start=chime(step,start)
        print('epoch:', step // train_step_per_epoch)
        total_score,scores=evaluate()
        if total_score>=max_total_score:
            max_total_score=total_score
            saver.save(sess,save_path,step)
            print('model saved in',save_path)