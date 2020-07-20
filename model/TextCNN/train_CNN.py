# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
import os
import random
import shutil
import sys

import data_helpers
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from text_cnn import TextCNN


FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
PARENT_PATH = os.path.dirname(FILE_PATH)
sys.path.append(PARENT_PATH)

import jieba

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
                       "Data source for the negative data.")
tf.flags.DEFINE_string("embedding_file", "./corpus/vectors.bin", "Data source for the trained embeddings.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 200, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 1000, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("fine_tuning", False, "Fine-tuning or nothing (default: Flase)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS


# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")

def preprocess_(train_file, test_file):
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text_train, y_str_train = data_helpers.load_data_and_strlabels_multi_classes(train_file)
    x_text_test, y_str_test = data_helpers.load_data_and_strlabels_multi_classes(test_file)

    y_dev = data_helpers.one_hot(y_str_test, use_fixed_map=True)

    def fenci(x_text):
        x_text_after_seg = []
        for xi in x_text:
            word_list = jieba.lcut(xi)  # cut_all=True
            seg_list = []
            for w in word_list:
                seg_list.append(w)
            x_text_after_seg.append(u' '.join(seg_list))
        return x_text_after_seg
    x_text = fenci(x_text_train)
    x_text_test = fenci(x_text_test)

    def two_list2dict(key_list, val_list):
        dic = {}
        for a, b in zip(key_list, val_list):
            dic[a] = dic.get(a, [])
            dic[a].append(b)

        return dic

    def balance_train_by_sampling(train_clas_sample_dict, num):
        train_data, labels = [], []
        for k, v in train_clas_sample_dict.items():
            for i in range(num):
                train_data.append(random.choice(v))
                labels.append(k)
        return train_data, labels

    dic = two_list2dict(y_str_train, x_text)
    x_text, y_str_train = balance_train_by_sampling(dic, 50)

    y_train = data_helpers.one_hot(y_str_train)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    vocab_processor.fit(x_text + x_text_test)
    x_train = np.array(list(vocab_processor.transform(x_text)))
    x_dev = np.array(list(vocab_processor.transform(x_text_test)))

    # Load the trained embeddings
    embeddings = list()
    if FLAGS.fine_tuning:
        vocab_dict = vocab_processor.vocabulary_._mapping
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
        vocabulary = list(list(zip(*sorted_vocab))[0])
        word_emb = data_helpers.load_vector_bin(FLAGS.embedding_file, binary=True)
        for word in vocabulary:
            try:
                embeddings.append(word_emb.get(word).tolist())
            except:
                embeddings.append(np.random.uniform(-1.0, 1.0, size=FLAGS.embedding_dim).tolist())

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev, embeddings


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y_str = data_helpers.load_data_and_strlabels_multi_classes('./corpus/train.txt')
    y = data_helpers.one_hot(y_str)

    x_text_after_seg = []
    for xi in x_text:
        word_list = jieba.lcut(xi)  # cut_all=True
        seg_list = []
        for w in word_list:
            seg_list.append(w)
        x_text_after_seg.append(u' '.join(seg_list))
    x_text = x_text_after_seg

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Load the trained embeddings
    embeddings = list()
    if FLAGS.fine_tuning:
        vocab_dict = vocab_processor.vocabulary_._mapping
        sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
        vocabulary = list(list(zip(*sorted_vocab))[0])
        word_emb = data_helpers.load_vector_bin(FLAGS.embedding_file, binary=True)
        for word in vocabulary:
            try:
                embeddings.append(word_emb.get(word).tolist())
            except:
                embeddings.append(np.random.uniform(-1.0, 1.0, size=FLAGS.embedding_dim).tolist())

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_text_shuffled = [y_str[i]+'\t'+x_text[i] for i in shuffle_indices]
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_text_train, x_text_dev = x_text_shuffled[:dev_sample_index], x_text_shuffled[dev_sample_index:]
    open('corpus/train_set.txt','w',encoding='utf-8').write('\n'.join(x_text_train))
    open('corpus/test_set.txt', 'w', encoding='utf-8').write('\n'.join(x_text_dev))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev, embeddings


def train(x_train, y_train, vocab_processor, x_dev, y_dev, embeddings):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                training=True,
                trained_embeddings=embeddings,
                fine_tuning=FLAGS.fine_tuning)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = 'MAXTIME'
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            # region clean output directory
            for the_file in os.listdir(out_dir):
                file_path = os.path.join(out_dir, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(e)
            # endregion

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Add restore
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print("Restore from the checkpoint {0}".format(checkpoint))
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                print(current_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, embeddings = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev, embeddings)


if __name__ == '__main__':
    tf.app.run()
