import tensorflow as tf
from tensorflow.contrib import layers,rnn
import utils

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
        self.fc1_hid_size = config.fc1_hid_size
        self.n_law = config.n_law
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.input=tf.placeholder(tf.int32, [None,self.doc_len,self.sent_len])
        self.input_doc_length = tf.placeholder(tf.int32, [None])
        self.input_sent_length = tf.placeholder(tf.int32, [None])
        self.law_label=tf.placeholder(tf.float32,[None,self.n_law])
        self.embedding=tf.Variable(word_embeddings,trainable=False)
        inputs = tf.nn.embedding_lookup(self.embedding, self.input)
        laws = tf.nn.embedding_lookup(self.embedding, law_input)

        inputs_shape=utils.get_tensor_shape(inputs)
        n_feat=inputs_shape[-1]

        with tf.name_scope('get_laws'):
            inputs_2d = tf.reshape(inputs, [self.batch_size, -1])
            law_index=layers.fully_connected(inputs_2d,self.n_law,activation_fn=tf.nn.softmax)
            # law_index=tf.nn.top_k(law_index,self.k_laws)
            self.law_index=tf.contrib.framework.argsort(law_index,-1)[:,-self.k_laws:]
            self.laws=tf.nn.embedding_lookup(laws,self.law_index)
            law_doc_length=tf.gather(law_doc_length,self.law_index)
            law_sent_length=tf.gather(law_sent_length,self.law_index)

        with tf.name_scope('fact_encoder'):
            u_fs=tf.get_variable('u_fs',initializer=layers.xavier_initializer())
            u_fw=tf.get_variable('u_fw',initializer=layers.xavier_initializer())
            inputs=tf.reshape(inputs,[self.batch_size*self.doc_len,self.sent_len,n_feat])
            sent_encoded=self.seq_encoder(inputs,u_fw,config.lstm_size,self.input_sent_length,'fact_sent')
            sent_encoded=tf.reshape(sent_encoded,[self.batch_size,self.doc_len,n_feat])
            self.d_f=self.seq_encoder(sent_encoded,u_fs,config.lstm_size,self.input_doc_length,'fact_doc')

        with tf.name_scope('law_encoder'):
            u_aw=tf.layers.dense(self.d_f,n_feat)
            u_as=tf.layers.dense(self.d_f,n_feat)
            laws=tf.reshape(self.laws,[self.batch_size*self.k_laws*self.law_doc_len,self.law_sent_len,n_feat])
            sent_encoded=self.seq_encoder(laws,u_aw,config.lstm_law_size,law_sent_length,'law_sent')
            sent_encoded=tf.reshape(sent_encoded,[self.batch_size*self.k_laws,self.doc_len,n_feat])
            law_repr=self.seq_encoder(sent_encoded,u_as,config.lstm_law_size,law_doc_length,'law_doc')
            law_repr=tf.reshape(law_repr,[self.batch_size,self.k_laws,n_feat])

        with tf.name_scope('law_aggregator'):
            u_ad=tf.layers.dense(self.d_f,n_feat)
            self.d_a=self.seq_encoder(law_repr, u_ad, config.lstm_law_size, self.k_laws, 'aggregator')

        with tf.name_scope('softmax'):
            self.outputs=tf.layers.dense(tf.concat([self.d_f,self.d_a],-1),self.n_law)

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
        self.loss=tf.losses.softmax_cross_entropy(self.law_label,self.outputs)
        loss_reg=tf.losses.get_regularization_loss()*config.l2_ratio
        self.loss=self.loss+loss_reg

    def atten_encoder(self,Q,K):
        #Q ...*seq_len_q*F
        #K=V ...*seq_len_k*F
        K_shape=utils.get_tensor_shape(K)
        scores=tf.layers.dense(K,K_shape[-1],activation=tf.nn.tanh)@tf.transpose(Q,[-1,-2])
        scores=tf.nn.softmax(scores)
        return tf.reduce_sum(scores*K,-2)

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
                initial_state_fw = cells[0].zero_state(self.batch_size, tf.float32),
                initial_state_bw = cells[1].zero_state(self.batch_size, tf.float32),
                sequence_length=length,
                dtype=tf.float32)
        return outputs

    def seq_encoder(self,input,u,cell_size,length,scope):
        return self.atten_encoder(u,self.gru_encoder(input,cell_size,length,scope))


class ModelConfig:
    def __init__(self):
        self.lstm_size = 100
        self.lstm_law_size=100
        self.k_laws=10
        self.doc_len = 10
        self.law_doc_len=500
        self.batch_size = 64
        self.fc1_hid_size = 512
        self.n_law = 183
        self.num_layers = 2
        self.threshold = .5
        self.l2_ratio=.0
        self.sent_len=self.law_sent_len=100