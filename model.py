import tensorflow as tf
import utils
import config
import func

class Model(object):
    def __init__(self, vocab_size, ckpt_folder, name='model'):
        self.name = name
        self.ckpt_folder = ckpt_folder
        self.vocab_size = vocab_size
        if self.ckpt_folder is not None:
            utils.mkdir(self.ckpt_folder)
        initializer = tf.random_uniform_initializer(-0.05, 0.05)
        with tf.variable_scope(self.name, initializer=initializer):
            self.initialize()


    def initialize(self):
        self.create_inputs()
        self.create_embeddings()
        self.create_encoder()
        self.create_decoder()
        self.create_attention()


    def create_inputs(self):
        with tf.name_scope('input'):
            self.input_passage_word = tf.placeholder(tf.int32, shape=[None, None], name='passage_word')
            self.input_question_word = tf.placeholder(tf.int32, shape=[None, None], name='question_word')
            self.input_keep_prob = tf.placeholder(tf.float32, name='keep_prob')
            self.batch_size = tf.shape(self.input_passage_word)[0]
            self.passage_mask, self.passage_len = func.tensor_to_mask(self.input_passage_word)
            self.question_mask, self.question_len = func.tensor_to_mask(self.input_question_word)


    def create_embeddings(self):
        with tf.name_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=[self.vocab_size, config.embedding_dim])
            self.passage_emb = tf.nn.embedding_lookup(self.embedding, self.input_passage_word, name='passage_emb')
            tf.summary.histogram('embedding/embedding', self.embedding)
            tf.summary.histogram('embedding/passage_emb', self.passage_emb)


    def create_encoder(self):
        with tf.name_scope('encoder'):
            self.passage_enc, _ = func.rnn('bi-lstm', self.passage_emb, self.passage_len, config.encoder_hidden_dim, 2, self.input_keep_prob)
            self.passage_enc = tf.nn.dropout(self.passage_enc, self.input_keep_prob, name='passage_enc')
            b_fw = self.passage_enc[:,-1,:config.encoder_hidden_dim]
            b_bw = self.passage_enc[:,0,config.encoder_hidden_dim:]
            self.init_state = tf.concat([b_fw, b_bw], -1)
            tf.summary.histogram('encoder/passage_enc', self.passage_enc)
            #tf.summary.histogram('encoder/passage_state', self.passage_state)
            tf.summary.histogram('encoder/init_state', self.init_state)


    def create_decoder(self):
        with tf.name_scope('decoder'):
            initial_state = tf.contrib.rnn.LSTMStateTuple(self.init_state[:,:config.decoder_hidden_dim], self.init_state[:,config.decoder_hidden_dim:])
            output_sequence, self.decoder_length = func.rnn_decode(
                'lstm', self.batch_size, config.decoder_hidden_dim, self.embedding,
                config.SOS_ID, config.EOS_ID, (initial_state,), config.max_question_len)
            self.decoder_h = tf.identity(output_sequence.rnn_output, 'decoder_h')


    def create_attention(self):
        with tf.name_scope('attention'):
            self.ct = func.dot_attention(self.decoder_h, self.passage_enc, self.passage_mask, config.dot_attention_dim, self.input_keep_prob)
            self.combined_h = tf.concat([self.decoder_h, self.ct], -1, name='combined_h')#[batch, question_len, 450]           
            self.wt = tf.get_variable('wt', shape=[config.max_question_len, self.combined_h.get_shape()[-1], config.decoder_hidden_dim])
            self.ws = tf.get_variable('ws', shape=[config.decoder_hidden_dim, self.vocab_size])
            question_len = tf.shape(self.combined_h)[1]
            self.wt_h = tf.einsum('bij,ijk->bik', self.combined_h, self.wt[:question_len,:,:], name='wt_h')
            self.ws_tanh_wt = tf.einsum('bik,kj->bij', tf.tanh(self.wt_h), self.ws)


if __name__ == '__main__':
    model = Model(10000, None)