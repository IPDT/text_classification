#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/15/2018 1:21 PM
# @Author  : SkullFang
# @Contact : yzhang.private@gmail.com
# @File    : train.py
# @Software: PyCharm
import os
import time
import shutil
import logging
import numpy as np
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN
from sklearn.model_selection import train_test_split
from data_preprocess.data_helper import load_data, load_embeddings, batch_iter
from cnn_rnn_config import CNNRNNConfig

# logging.getLogger().setLevel(logging.INFO)
model_config = CNNRNNConfig()


def train_cnn_rnn():
    # input_file = sys.argv[1]
    # input_file = 'data/simple.csv'
    # x_, y_, vocabulary, vocabulary_inv, df, labels = data_helper.load_data(input_file)
    # print(type(x_))
    # training_config = sys.argv[2]
    # training_config = 'training_config.json'
    # params = json.loads(open(training_config).read())

    # Prepare data and vocabulary
    x, y, vocabulary_list, vocabulary_dict, labels = load_data(model_config.forced_sequence_length,
                                                               './data_preprocess/vocab_en.txt')
    print('-------------------load data successfully-------------------')

    # Split the original dataset into train set and test set
    x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.1)
    # Split the train set into train set and dev set
    x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.1)

    # Assign a 300 dimension vector to each word
    embedding_mat = load_embeddings(vocabulary_list, vocabulary_dict, model_config.embedding_dim)
    print('-------------------embed data successfully------------------')

    print('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    print('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    # Create a directory, everything related to the training will be saved in this directory
    trained_dir = './trained_results/'
    if os.path.exists(trained_dir):
        shutil.rmtree(trained_dir)
    os.makedirs(trained_dir)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            cnn_rnn = TextCNNRNN(labels=labels,
                                 embedding_mat=embedding_mat,
                                 sequence_length=model_config.forced_sequence_length,
                                 num_classes=y_train.shape[1],
                                 non_static=model_config.non_static,
                                 hidden_unit=model_config.hidden_unit,
                                 max_pool_size=model_config.max_pool_size,
                                 filter_sizes=map(int, model_config.filter_sizes.split(",")),
                                 num_filters=model_config.num_filters,
                                 embedding_size=model_config.embedding_dim,
                                 l2_reg_lambda=model_config.l2_reg_lambda)
            print('-------------build model structure successfully-------------')

            global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            # optimizer = tf.train.RMSPropOptimizer(1e-3, decay=0.9)
            grads_and_vars = optimizer.compute_gradients(cnn_rnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            print('----------------build optimizer successfully----------------')
            # Checkpoint files will be saved in this directory during training
            timestamp = str(int(time.time()))
            checkpoint_dir = 'checkpoints/checkpoints_' + timestamp + '/'
            if os.path.exists(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)
            os.makedirs(checkpoint_dir)
            checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())

            # Training starts here
            train_batches = batch_iter(list(zip(x_train, y_train)), model_config.batch_size, model_config.num_epochs)
            best_accuracy, best_at_step = 0, 0

            # Train the model with x_train and y_train
            for train_batch in train_batches:
                x_train_batch, y_train_batch = zip(*train_batch)
                # print("y_train_batch:",y_train_batch[0])
                feed_dict = train_step(cnn_rnn, x_train_batch, y_train_batch)
                _, step, loss, accuracy = sess.run([train_op, global_step, cnn_rnn.loss, cnn_rnn.accuracy], feed_dict)

                # print("train_step",)
                current_step = tf.train.global_step(sess, global_step)
                # if current_step > 200:
                #     break
                # Evaluate the model with x_dev and y_dev
                if current_step % model_config.evaluate_every == 0:
                    dev_batches = batch_iter(list(zip(x_dev, y_dev)), model_config.batch_size, 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        feed_dict = dev_step(cnn_rnn, x_dev_batch, y_dev_batch)
                        step, loss, accuracy, num_dev_correct, predictions = sess.run(
                            [global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions],
                            feed_dict)
                        # acc, loss, num_dev_correct, predictions = dev_step(cnn_rnn, x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct
                    accuracy = float(total_dev_correct) / len(y_dev)
                    logging.info('Accuracy on dev set: {}'.format(accuracy))

                    if accuracy >= best_accuracy:
                        best_accuracy, best_at_step = accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))
            logging.critical('Training is complete, testing the best model on x_test and y_test')

            # Save the model files to trained_dir. predict.py needs trained model files.
            saver.save(sess, trained_dir + "best_model.ckpt")

            # Save pb file
            graph_def = tf.get_default_graph().as_graph_def()
            graph_pb = tf.graph_util.convert_variables_to_constants(sess=sess, input_graph_def=graph_def,
                                                                    output_node_names=['labels', 'scores/scores',
                                                                                       'scores/predictions'])
            tf.train.write_graph(graph_pb, '.', trained_dir + "graph.pb", as_text=False)

            # Evaluate x_test and y_test
            saver.restore(sess, checkpoint_prefix + '-' + str(best_at_step))
            test_batches = batch_iter(list(zip(x_test, y_test)), model_config.batch_size, 1, shuffle=False)
            total_test_correct = 0
            for test_batch in test_batches:
                x_test_batch, y_test_batch = zip(*test_batch)
                feed_dict = dev_step(cnn_rnn, x_test_batch, y_test_batch)
                step, loss, accuracy, num_test_correct, predictions = sess.run(
                    [global_step, cnn_rnn.loss, cnn_rnn.accuracy, cnn_rnn.num_correct, cnn_rnn.predictions], feed_dict)
                # acc, loss, num_test_correct, predictions = dev_step(cnn_rnn,x_test_batch, y_test_batch)

                total_test_correct += int(num_test_correct)
            print('Accuracy on test set: {}'.format(float(total_test_correct) / len(y_test)))

    # Save trained parameters and files since predict.py needs them
    # with open(trained_dir + 'words_index.json', 'w') as outfile:
    #     json.dump(vocabulary, outfile, indent=4, ensure_ascii=False)
    # with open(trained_dir + 'embeddings.pickle', 'wb') as outfile:
    #     pickle.dump(embedding_mat, outfile, pickle.HIGHEST_PROTOCOL)
    # with open(trained_dir + 'labels.json', 'w') as outfile:
    #     json.dump(labels, outfile, indent=4, ensure_ascii=False)

    # params['sequence_length'] = x_train.shape[1]
    # with open(trained_dir + 'trained_parameters.json', 'w') as outfile:
    #     json.dump(params, outfile, indent=4, sort_keys=True, ensure_ascii=False)


def real_len(batches):
    return [np.ceil(np.argmin(batch + [0]) * 1.0 / model_config.max_pool_size) for batch in batches]


def train_step(model, x_batch, y_batch):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: model_config.dropout_keep_prob,
        model.batch_size: len(x_batch),
        model.pad: np.zeros([len(x_batch), 1, model_config.embedding_dim, 1]),
        model.real_len: real_len(x_batch),
    }
    # print("batch_size:", len(x_batch))
    # print('pad', np.zeros([len(x_batch), 1, model_config.embedding_dim, 1])[1])
    # print("real_len:", (real_len(x_batch))[1])
    # print("real_len:", real_len(x_batch))
    return feed_dict


def dev_step(model, x_batch, y_batch):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.dropout_keep_prob: 1.0,
        model.batch_size: len(x_batch),
        model.pad: np.zeros([len(x_batch), 1, model_config.embedding_dim, 1]),
        model.real_len: real_len(x_batch),
    }
    return feed_dict


if __name__ == '__main__':
    train_cnn_rnn()
