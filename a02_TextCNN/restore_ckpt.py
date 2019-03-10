import tensorflow as tf
import numpy as np
from p7_TextCNN_model import TextCNN
from data_util import create_vocabulary, load_data_multilabel
import pickle
import h5py
import os
import random
from sklearn.metrics import accuracy_score

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("traning_data_path", "/Users/sunyitao/Documents/Projects/GitHub/PsychicLearners/data/fashion_train_invert.txt",
                           "path of traning data.")  # ../data/sample_multiple_label.txt
tf.app.flags.DEFINE_integer("vocab_size", 100000, "maximum vocab size.")
tf.app.flags.DEFINE_float("learning_rate", 0.0003, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 1000, "how many steps before decay learning rate.")  # 6000批处理的大小 32-->128
tf.app.flags.DEFINE_float("decay_rate", 1.0, "Rate of decay for learning rate.")  # 0.65一次衰减多少
tf.app.flags.DEFINE_string(
    "ckpt_dir", "text_cnn_title_desc_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("sentence_len", 15, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 128, "embedding size")
tf.app.flags.DEFINE_boolean(
    "is_training_flag", True, "is training.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 15, "number of epochs to run.")
tf.app.flags.DEFINE_integer(
    "validate_every", 1, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding", False,
                            "whether to use embedding or not.")
tf.app.flags.DEFINE_integer(
    "num_filters", 128, "number of filters")  # 256--->512
tf.app.flags.DEFINE_string(
    "word2vec_model_path", "word2vec-title-desc.bin", "word2vec's vocabulary and vectors")
tf.app.flags.DEFINE_string("name_scope", "cnn", "name scope value.")
tf.app.flags.DEFINE_boolean(
    "multi_label_flag", False, "use multi label or single label.")
filter_sizes = [6, 7, 8]

print("Restoring Variables from Checkpoint.")
saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
trainX, trainY, testX, testY = None, None, None, None
vocabulary_word2index, vocabulary_index2word, vocabulary_label2index, _ = create_vocabulary(FLAGS.traning_data_path, FLAGS.vocab_size, name_scope=FLAGS.name_scope)
vocab_size = len(vocabulary_word2index)
print("cnn_model.vocab_size:", vocab_size)
num_classes = len(vocabulary_label2index)
print("num_classes:", num_classes)
#num_examples,FLAGS.sentence_len=trainX.shape
#print("num_examples of training:",num_examples,";sentence_len:",FLAGS.sentence_len)
train, test = load_data_multilabel(
    FLAGS.traning_data_path, vocabulary_word2index, vocabulary_label2index, FLAGS.sentence_len)
trainX, trainY = train
testX, testY = test
