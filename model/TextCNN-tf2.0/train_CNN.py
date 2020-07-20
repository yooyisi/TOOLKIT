# -*- coding: utf-8 -*-

import shutil
from io import open

import sys
import tensorflow as tf
import numpy as np
import os
import datetime
import data_helpers
from text_cnn import TextCNN

FILE_PATH = os.path.split(os.path.realpath(__file__))[0]
PARENT_PATH = os.path.dirname(FILE_PATH)
sys.path.append(PARENT_PATH)
from vocab import Vocab

import jieba
import gensim

# Parameters
# ==================================================

# Data loading params
dev_sample_percentage = 0.1  # , "Percentage of the training data to use for validation")
positive_data_file = "./data/rt-polaritydata/rt-polarity.pos"  # "Data source for the positive data.")
negative_data_file = "./data/rt-polaritydata/rt-polarity.neg"  # "Data source for the negative data.")
embedding_file = "./corpus/vector.bin"  # , "Data source for the trained embeddings.")

# Model Hyperparameters
embedding_dim = 300  # , "Dimensionality of character embedding (default: 128)")
filter_sizes = "3,4,5"  # , "Comma-separated filter sizes (default: '3,4,5')")
num_filters = 128  # , "Number of filters per filter size (default: 128)")
dropout_keep_prob = 0.5  # , "Dropout keep probability (default: 0.5)")
l2_reg_lambda = 0.0  # , "L2 regularization lambda (default: 0.0)")

# Training parameters
batch_size = 64  # , "Batch Size (default: 64)")
num_epochs = 200  # , "Number of training epochs (default: 200)")
evaluate_every = 100  # , "Evaluate model on dev set after this many steps (default: 100)")
checkpoint_every = 100  # , "Save model after this many steps (default: 100)")
num_checkpoints = 5  # , "Number of checkpoints to store (default: 5)")
fine_tuning = True  # , "Fine-tuning or nothing (default: Flase)")

# Misc Parameters
allow_soft_placement = True  # , "Allow device soft device placement")
log_device_placement = False  # , "Log placement of ops on devices")

stop_word_file = FILE_PATH + '/corpus/stop_word.txt'
stopword_list = list(open(stop_word_file, "r", encoding='utf-8').readlines())
stopword_list = [x.strip() for x in stopword_list]


def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels_multi_classes('./corpus/train.txt')

    x_text_after_seg = []
    for xi in x_text:
        word_list = jieba.lcut(xi)
        x_text_after_seg.append(u' '.join(word_list))
    x_text = x_text_after_seg

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = Vocab()
    x = np.array(list(vocab_processor.build(x_text)))

    # Load the trained embeddings
    embeddings = list()
    # if fine_tuning:
    #     vocab_dict = vocab_processor.vocabulary_._mapping
    #     sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    #     vocabulary = list(list(zip(*sorted_vocab))[0])
    #     model = gensim.models.KeyedVectors.load_word2vec_format(embedding_file, binary=True)
    #     for word in vocabulary:
    #         try:
    #             embeddings.append(model.get_vector(word).tolist())
    #         except:
    #             embeddings.append(np.random.uniform(-1.0, 1.0, size=embedding_dim).tolist())

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(dev_sample_percentage * float(len(y)))
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
        session_conf = tf.compat.v1.ConfigProto(
            allow_soft_placement=allow_soft_placement,
            log_device_placement=log_device_placement)
        sess = tf.compat.v1.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=embedding_dim,
                filter_sizes=list(map(int, filter_sizes.split(","))),
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda,
                training=True,
                trained_embeddings=embeddings,
                fine_tuning=fine_tuning)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.compat.v1.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.compat.v1.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.compat.v1.summary.scalar("{}/grad/sparsity".format(v.name),
                                                                   tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.compat.v1.summary.merge(grad_summaries)

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
            loss_summary = tf.compat.v1.summary.scalar("loss", cnn.loss)
            acc_summary = tf.compat.v1.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.compat.v1.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.compat.v1.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.compat.v1.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.compat.v1.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: dropout_keep_prob
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
                list(zip(x_train, y_train)), batch_size, num_epochs)
            # Add restore
            checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print("Restore from the checkpoint {0}".format(checkpoint))
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.compat.v1.train.global_step(sess, global_step)
                print(current_step)
                if current_step % evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % checkpoint_every == 0:
                    # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev, embeddings = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev, embeddings)


if __name__ == '__main__':
    tf.compat.v1.app.run()
