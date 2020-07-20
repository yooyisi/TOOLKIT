#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
from io import open

import re
import sys
import tensorflow as tf
import numpy as np
import os

import time
from text_cnn import TextCNN
import data_helpers
from tensorflow.contrib import learn

FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
PARENT_PATH = os.path.dirname(FILE_PATH)
sys.path.append(PARENT_PATH)

import jieba

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("fine_tuning", True, "Fine-tuning or nothing (default: Flase)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
NUM_CLASSES = 6

stop_word_file = FILE_PATH + '/corpus/stop_word.txt'
stopword_list = list(open(stop_word_file, "r", encoding='utf-8').readlines())
stopword_list = [x.strip() for x in stopword_list]

# Output directory for models and summaries
timestamp = 'MAXTIME'
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Reading from {}\n".format(out_dir))

with open('./runs/label.dict', 'r') as infile:
    label_dict = json.load(infile)


class Predict:
    def __init__(self):
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(os.path.join(out_dir, "vocab"))
        self.embeddings = np.float32(np.random.random([len(self.vocab_processor.vocabulary_), FLAGS.embedding_dim]))
        with tf.Graph().as_default():
            session_conf = tf.compat.v1.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.compat.v1.Session(config=session_conf)
            with self.sess.as_default():
                self.cnn = TextCNN(
                    sequence_length=self.vocab_processor.max_document_length,
                    num_classes=NUM_CLASSES,
                    vocab_size=len(self.vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda,
                    trained_embeddings=self.embeddings,
                    fine_tuning=FLAGS.fine_tuning
                )

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                # Initialize all variables
                self.sess.run(tf.compat.v1.global_variables_initializer())
                saver = tf.compat.v1.train.Saver()
                checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
                path = saver.restore(self.sess, checkpoint)
                print("Restore model checkpoint from {}\n".format(path))

    def preprocess(self, origin_text):
        text_after_seg = []
        word_list = jieba.lcut(origin_text)  # cut_all=True
        seg_list = []
        for w in word_list:
            seg_list.append(w)
        text_after_seg.append(' '.join(seg_list))

        vector_text = np.array(list(self.vocab_processor.fit_transform(text_after_seg)))

        return vector_text

    def predict(self, sent):
        """
        Evaluates model on a dev set
        """
        x = self.preprocess(sent)
        x_batch = (x)
        feed_dict = {
            self.cnn.input_x: x_batch,
            self.cnn.dropout_keep_prob: 1.0
        }
        prediction, softmax_scores = self.sess.run(
            [self.cnn.predictions, self.cnn.softmax_scores],
            feed_dict)
        print(softmax_scores)
        return label_dict.get(str(prediction[0]))


def predict_score(in_path):
    model = Predict()
    in_file = open(in_path, "r", encoding="utf-8")
    num_correct = 0
    num_text = 0
    for line in in_file:
        y, x_text = line.strip("\n").split()
        print(x_text)
        y_p = model.predict(x_text)
        print(y_p)
        if y_p == y:
            num_correct += 1
        num_text += 1
    print(1.0 * num_correct / num_text)


def main(argv=None):
    model = Predict()
    # st = time.time()
    #
    # x_text, y = data_helpers.load_data_and_strlabels_multi_classes('./corpus/train.txt')
    # num_correct = 0
    # for xi, yi in zip(x_text, y):
    #     y_p = model.predict(xi)
    #     if y_p == yi:
    #         num_correct += 1
    # print((time.time()-st)/len(x_text))
    # print(1.0*num_correct/len(x_text))
    predict_score('./corpus/specialcase.txt')
    model.predict("fff")


if __name__ == '__main__':
    tf.compat.v1.app.run()
