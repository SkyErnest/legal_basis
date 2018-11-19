import tensorflow as tf
import tensorflow.contrib.layers as layers

class Model:
    def __init__(self,config,word_embeddings,law_input,law_length):
        self.lstm_size = config.lstm_size
        self.lstm_law_size = config.lstm_law_size
        self.k_laws=config.k_laws
        self.doc_len=config.doc_len
        self.law_doc_len=config.law_doc_len
        self.batch_size = config.batch_size
        self.fc1_hid_size = config.fc1_hid_size
        self.n_law = config.n_law
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.input=tf.placeholder(tf.int32, [None,self.doc_len])
        self.input_length = tf.placeholder(tf.int32, [None])
        self.law_label=tf.placeholder(tf.float32,[None,self.n_law])
        self.embedding=tf.Variable(word_embeddings,trainable=False)
        inputs = tf.nn.embedding_lookup(self.embedding, self.input)
        laws = tf.nn.embedding_lookup(self.embedding, law_input)

        with tf.name_scope('lstm'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_size, forget_bias=0.0)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
            self.initial_state = cell.zero_state(self.batch_size, tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(cell, inputs, initial_state=self.initial_state,
                                           sequence_length=self.input_length)
        with tf.name_scope('get_laws'):
            law_index=layers.fully_connected(outputs,self.n_law,activation_fn=tf.nn.softmax)
            self.law_index=tf.contrib.framework.argsort(law_index,-1)
            self.laws=tf.nn.embedding_lookup(laws,self.law_index)

        with tf.name_scope('lstm_law'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_law_size, forget_bias=0.0)
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=self.keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
            self.initial_state = cell.zero_state(self.batch_size, tf.float32)
            outputs_law, _ = tf.nn.dynamic_rnn(cell, self.laws, initial_state=self.initial_state,
                                           sequence_length=law_length)

        output = tf.expand_dims(tf.reshape(outputs, [self.batch_size, -1, self.lstm_size]), -1)


        with tf.name_scope("lstm_maxpool"):
            output_pooling = tf.nn.max_pool(output,
                                            ksize=[1, self.doc_len, 1, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='VALID',
                                            name="pool")
            self.output = tf.reshape(output_pooling, [-1, self.lstm_size])

        with tf.name_scope("fc"):
            self.law_output=layers.fully_connected(self.output,self.n_law,activation_fn=None)
            self.law_pred=tf.where(tf.nn.sigmoid(self.law_output)>config.law_threshold)

        self.loss_law=tf.losses.sigmoid_cross_entropy(self.law_label,self.law_output)
        loss_reg=tf.losses.get_regularization_loss()*config.l2_ratio
        self.loss_total=+self.loss_law+loss_reg

    def attention(self,Q,K,V,F_=None,L_=None):
        #Q ...*N*F
        #K ...*M*F
        #V ...*M*L
        if F_ is None:
            F_=Q.get_shape().as_list()[-1]
        if L_ is None:
            L_=V.get_shape().as_list()[-1]
        WQ=tf.get_variable(tf.get_variable_scope()+'WQ',shape=[Q.get_shape().as_list()[-1],F_],initializer=tf.initializers.xavier_initializer())
        WK=tf.get_variable(tf.get_variable_scope()+'WK',shape=[K.get_shape().as_list()[-1],F_],initializer=tf.initializers.xavier_initializer())
        WV=tf.get_variable(tf.get_variable_scope()+'WV',shape=[V.get_shape().as_list()[-1],L_],initializer=tf.initializers.xavier_initializer())
        res_map=(Q@WQ)@tf.transpose(K@WK,[-1,-2])
        res_map=tf.reduce_mean(res_map,-2)
        return V@WV*tf.expand_dims(res_map,-1)

class ModelConfig:
    def __init__(self):
        self.lstm_size = 100
        self.lstm_law_size=100
        self.k_laws=10
        self.doc_len = 500
        self.law_doc_len=500
        self.batch_size = 64
        self.fc1_hid_size = 512
        self.n_law = 183
        self.loss_radio = .5
        self.num_layers = 2
        self.law_threshold = .5