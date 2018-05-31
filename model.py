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
        self.create_encoding()
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


    def create_encoding(self):
        with tf.name_scope('encoding'):
            self.passage_enc, self.passage_state = func.rnn('bi-lstm', self.passage_emb, self.passage_len, config.hidden_dim, 2, self.input_keep_prob)
            self.passage_enc = tf.nn.dropout(self.passage_enc, self.input_keep_prob, name='passage_enc')
            tf.identity(self.passage_state, 'passage_state')
            tf.summary.histogram('encoding/passage_enc', self.passage_enc)
            tf.summary.histogram('encoding/passage_state', self.passage_state)

    
    def create_attention(self):
        with tf.name_scope('attention'):
            self.ct = func.dot_attention(self.passage_enc, self.passage_enc, self.passage_mask, config.dot_attention_dim, self.input_keep_prob)


if __name__ == '__main__':
    model = Model(10000, None)