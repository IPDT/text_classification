#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/16/2018 10:23 AM
# @Author  : SkullFang
# @Contact : yzhang.private@gmail.com
# @File    : predict.py
# @Software: PyCharm
import os
import sys
import json
import shutil
import pickle
import logging
from test import data_helper
import numpy as np
import pandas as pd
import tensorflow as tf
from text_cnn_rnn import TextCNNRNN

logging.getLogger().setLevel(logging.INFO)

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(open(trained_dir + 'words_index.json').read())
	labels = json.loads(open(trained_dir + 'labels.json').read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
		print(fetched_embedding)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)
	return params, words_index, labels, embedding_mat

def load_test_data(test_file, labels):
	# df = pd.read_csv(test_file, sep='|')
	# select = ['Descript']

	#x变成用，隔开的数据
	test_examples_test=[]
	# df = df.dropna(axis=0, how='any', subset=select)
	test_str=sys.argv[1]
	#test_examples = df[select[0]].apply(lambda x: data_helper.clean_str(x).split(' ')).tolist()
	clean_str = list(data_helper.clean_str(test_str).split(' '))
	test_examples_test.append(clean_str)
	# print("test example:",test_examples)
	# print("test example type:",type(test_examples))
	# print('test example test:',test_examples_test)
	# print('test example test type:', type(test_examples_test))
	test_examples=test_examples_test

	#把所有的label变成onehot存储下来
	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))
	#print("label_dict",label_dict)

	#找到测试案例的label对应的onehot
	y_ = None
	# if 'Category' in df.columns:
	# 	select.append('Category')
	# 	y_ = df[select[1]].apply(lambda x: label_dict[x]).tolist()
	#print("y_",y_)

	#去掉重复的内容
	# not_select = list(set(df.columns) - set(select))
	# print('not_select',not_select)
	# df = df.drop(not_select, axis=1)
	# print("df:",df)
	df=pd.DataFrame()
	return test_examples, y_, df

def map_word_to_index(examples, words_index):
	"""
	:param examples: 一条输入数据
	:param words_index:
	:return: 训练向量
	"""
	x_ = []
	for example in examples:
		temp = []
		for word in example:
			if word in words_index:
				temp.append(words_index[word])
			else:
				temp.append(0)
		x_.append(temp)

	print('x_',x_)
	return x_

def predict_unseen_data():
	trained_dir = "./trained_results_test/"#字典啥玩意的
	print('trained_dir',trained_dir)
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	test_file = "./data/test.csv"
	#print('test_file', test_file)#测试集在的地方
	params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
	x_, y_, df = load_test_data(test_file, labels)

	x_ = data_helper.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
	x_ = map_word_to_index(x_, words_index) #看x需不需要加pandding

	x_test, y_test = np.asarray(x_), None
	if y_ is not None:
		y_test = np.asarray(y_)

	print('need_x:',x_)
	print('need_y',y_)
	timestamp = trained_dir.split('/')[-2].split('_')[-1]
	predicted_dir = './predicted_results_' + timestamp + '/'
	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
	os.makedirs(predicted_dir)

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto()
		sess = tf.Session()
		with sess.as_default():
			cnn_rnn = TextCNNRNN(
				embedding_mat = embedding_mat,
				non_static = params['non_static'],
				hidden_unit = params['hidden_unit'],
				sequence_length = len(x_test[0]),
				max_pool_size = params['max_pool_size'],
				filter_sizes = map(int, params['filter_sizes'].split(",")),
				num_filters = params['num_filters'],
				num_classes = len(labels),
				embedding_size = params['embedding_dim'],
				l2_reg_lambda = params['l2_reg_lambda'])

			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

			def predict_step(x_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.dropout_keep_prob: 1.0,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				predictions = sess.run([cnn_rnn.predictions], feed_dict)
				return predictions

			checkpoint_file = trained_dir + 'best_model.ckpt'
			saver = tf.train.Saver(tf.all_variables())
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
			saver.restore(sess, checkpoint_file)
			logging.critical('{} has been loaded'.format(checkpoint_file))

			batches = data_helper.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)

			predictions, predict_labels = [], []
			for x_batch in batches:
				batch_predictions = predict_step(x_batch)[0]
				for batch_prediction in batch_predictions:
					predictions.append(batch_prediction)
					predict_labels.append(labels[batch_prediction])

			print('predict_labels: ',predict_labels)
			# Save the predictions back to file
			# df['NEW_PREDICTED'] = predict_labels
			# columns = sorted(df.columns, reverse=True)
			# df.to_csv(predicted_dir + 'predictions_all.csv', index=False, columns=columns, sep='|')

			if y_test is not None:
				y_test = np.array(np.argmax(y_test, axis=1))
				accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
				logging.critical('The prediction accuracy is: {}'.format(accuracy))

			logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))

if __name__ == '__main__':
	predict_unseen_data()
	# python3 predict.py ./trained_results_1478563595/ ./data/small_samples.csv
	# predict_unseen_data(x_=[[18, 3, 91, 1, 51, 22, 0, 0, 0, 0, 0, 0, 0, 0]],
	# 					y_=[([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])
	#x : [[18, 3, 91, 1, 51, 22, 0, 0, 0, 0, 0, 0, 0, 0]]
	#y: [array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
