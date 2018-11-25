import tensorflow as tf
from tensorflow.contrib import layers,rnn
import utils
import numpy as np

class Model:
    def __init__(self,config,word_embeddings,law_input,law_doc_length,law_sent_length):
        self.lstm_size = config.lstm_size
        self.lstm_law_size = config.lstm_law_size
        self.k_laws=config.k_laws
        self.doc_len=config.doc_len
        self.sent_len=config.sent_len
        self.law_sent_len=config.law_sent_len
        self.law_doc_len=config.law_doc_len
        self.batch_size = config.batch_size
        self.n_law = config.n_law
        self.n_class=config.n_class
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.input=tf.placeholder(tf.int32, [self.batch_size,self.doc_len,self.sent_len])
        self.input_doc_length = tf.placeholder(tf.int32, [self.batch_size])
        self.input_sent_length = tf.placeholder(tf.int32, [self.batch_size,self.doc_len])
        self.label=tf.placeholder(tf.float32, [self.batch_size, self.n_class])
        self.law_label=tf.placeholder(tf.float32, [self.batch_size, self.n_law])
        self.embedding=tf.Variable(word_embeddings,trainable=False)
        inputs = tf.nn.embedding_lookup(self.embedding, self.input)
        laws = tf.nn.embedding_lookup(self.embedding, law_input)

        inputs_shape=utils.get_tensor_shape(inputs)
        self.n_fact_feat=self.n_law_feat=inputs_shape[-1]

        with tf.name_scope('get_laws'):
            inputs_2d = tf.reshape(inputs, [self.batch_size, self.doc_len * self.sent_len * self.n_fact_feat])
            law_index=layers.fully_connected(inputs_2d,self.n_law,activation_fn=tf.nn.softmax)
            # law_index=tf.nn.top_k(law_index,self.k_laws)
            self.law_index=tf.contrib.framework.argsort(law_index,-1)[:,-self.k_laws:]
            self.law_index=tf.reshape(self.law_index,[self.batch_size,self.k_laws])
            self.laws=tf.nn.embedding_lookup(laws,self.law_index)
            self.k_law_label=tf.map_fn(lambda x:tf.gather(x[0],x[1]), (self.law_label, self.law_index), dtype=tf.float32)
            law_doc_length=tf.gather(law_doc_length,self.law_index)
            law_sent_length=tf.gather(law_sent_length,self.law_index)

        with tf.name_scope('fact_encoder'):
            self.n_fact_feat=self.lstm_size*2
            # inputs=tf.reshape(inputs, [self.batch_size * self.doc_len, self.sent_len, self.n_fact_feat])
            u_fw=tf.get_variable('u_fw', shape=[1, self.n_fact_feat], initializer=layers.xavier_initializer())
            u_fs=tf.get_variable('u_fs', shape=[1, self.n_fact_feat], initializer=layers.xavier_initializer())
            sent_encoded,_=self.seq_encoder(inputs,u_fw,config.lstm_size,self.input_sent_length,'fact_sent')
            # sent_encoded=tf.reshape(sent_encoded, [self.batch_size, self.doc_len, self.n_fact_feat])
            self.d_f,_=self.seq_encoder(sent_encoded,u_fs,config.lstm_size,self.input_doc_length,'fact_doc')

        with tf.name_scope('law_encoder'):
            self.n_law_feat=self.lstm_law_size*2
            u_aw=tf.reshape(tf.layers.dense(self.d_f, self.n_law_feat),[self.batch_size,1,1,1,self.n_law_feat])
            u_as=tf.reshape(tf.layers.dense(self.d_f, self.n_law_feat),[self.batch_size,1,1,self.n_law_feat])
            # laws=tf.reshape(self.laws, [self.batch_size * self.k_laws * self.law_doc_len, self.law_sent_len, self.n_law_feat // 2])
            # law_sent_length=tf.reshape(law_sent_length,[self.batch_size*self.k_laws*self.law_doc_len])
            sent_encoded,_=self.seq_encoder(self.laws,u_aw,config.lstm_law_size,law_sent_length,'law_sent')
            # sent_encoded=tf.reshape(sent_encoded, [self.batch_size * self.k_laws, self.doc_len, self.n_law_feat])
            law_repr,_=self.seq_encoder(sent_encoded,u_as,config.lstm_law_size,law_doc_length,'law_doc')
            # law_repr=tf.reshape(law_repr, [self.batch_size, self.k_laws, self.n_law_feat])

        with tf.name_scope('law_aggregator'):
            u_ad=tf.reshape(tf.layers.dense(self.d_f, self.n_fact_feat),[self.batch_size,1,self.n_law_feat])
            self.d_a,self.law_score=self.seq_encoder(law_repr, u_ad, config.lstm_law_size, [self.k_laws]*self.batch_size, 'aggregator')

        with tf.name_scope('softmax'):
            self.outputs=layers.fully_connected(tf.concat([self.d_f,self.d_a],-1),config.fc_size1)
            self.outputs=layers.fully_connected(self.outputs,config.fc_size2)
            self.outputs=tf.layers.dense(self.outputs,self.n_class)

        # with tf.name_scope('lstm'):
        #     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=0.0)
        #     lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        #     cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
        #     self.initial_state = cell.zero_state(self.batch_size, tf.float32)
        #     with tf.variable_scope('context'):
        #         outputs, _ = tf.nn.dynamic_rnn(cell, inputs, initial_state=self.initial_state,
        #                                        sequence_length=self.input_length)
        #
        # output = tf.expand_dims(tf.reshape(outputs, [self.batch_size, -1, self.lstm_size]), -1)
        #
        # with tf.name_scope("lstm_maxpool"):
        #     output_pooling = tf.nn.max_pool(output,
        #                                     ksize=[1, self.doc_len, 1, 1],
        #                                     strides=[1, 1, 1, 1],
        #                                     padding='VALID',
        #                                     name="pool")


        # with tf.name_scope('lstm_law') as scope:
        #
        #     lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_law_size, forget_bias=0.0)
        #     lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
        #     cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
        #     self.initial_state = cell.zero_state(self.batch_size*self.k_laws, tf.float32)
        #     laws_extracted=tf.reshape(self.laws,[self.batch_size*self.k_laws,self.doc_len,self.embedding.get_shape().as_list()[-1]])
        #     with tf.variable_scope('law'):
        #         outputs_law, _ = tf.nn.dynamic_rnn(cell, laws_extracted, initial_state=self.initial_state,
        #                                        sequence_length=tf.reshape(law_length,[-1]))
        #     outputs_law=tf.reshape(outputs_law,[self.batch_size,self.k_laws,self.doc_len,self.lstm_law_size])

        self.prediction=tf.where(tf.nn.softmax(self.outputs)>config.threshold)
        self.loss_main=tf.losses.softmax_cross_entropy(self.norm_sum(self.label), self.outputs)
        self.loss_law=tf.losses.log_loss(self.norm_sum(self.k_law_label),self.law_score)
        loss_reg=tf.losses.get_regularization_loss()*config.l2_ratio
        self.loss=self.loss_main+loss_reg+config.attention_loss_ratio*self.loss_law

    def norm_sum(self,x):
        return x/(tf.reduce_sum(x,-1,keepdims=True)+1e-7)

    def atten_encoder(self,Q,K):
        #Q ...*seq_len_q*F
        #K=V ...*seq_len_k*F
        K_shape=utils.get_tensor_shape(K)
        K=tf.layers.dense(K,K_shape[-1],activation=tf.nn.tanh)
        # ======================================================
        # Q=tf.transpose(Q,[-1,-2])
        # scores=tf.map_fn(lambda x:x@Q,K,dtype=tf.float32)
        # -------------another implementation-------------------
        scores=tf.reduce_sum(K*Q,-1,keepdims=True)
        #=======================================================
        scores=tf.nn.softmax(scores,-1)
        return tf.reduce_sum(scores*K,-2),tf.squeeze(scores)

    def gru_encoder(self,input,cell_size,length,scope):
        with tf.variable_scope(scope):
            cells=[]
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    cells.append(rnn.GRUCell(cell_size))
            outputs, final_states=tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cells[0],
                cell_bw=cells[1],
                inputs=input,
                sequence_length=length,
                dtype=tf.float32)
        return tf.concat(outputs,-1)

    def seq_encoder(self,input,u,cell_size,length,scope):
        input_shape=utils.get_tensor_shape(input)
        input=tf.reshape(input,[np.prod(input_shape[:-2])]+input_shape[-2:])
        length=tf.reshape(length,[-1])
        rep=self.gru_encoder(input,cell_size,length,scope)
        rep=tf.reshape(rep,input_shape[:-1]+[cell_size*2])
        return self.atten_encoder(u,rep)


class ModelConfig:
    def __init__(self):
        self.learning_rate=0.1
        self.n_class=183
        self.n_law = 183
        self.lstm_size = 75
        self.lstm_law_size=75
        self.fc_size1=200
        self.fc_size2=150
        self.k_laws=20
        self.doc_len = 15
        self.law_doc_len=10
        self.batch_size = 8
        self.num_layers = 2
        self.threshold = .4
        self.l2_ratio=.0
        self.sent_len=self.law_sent_len=100
        self.attention_loss_ratio=.1