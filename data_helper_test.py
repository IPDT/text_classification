#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/15/2018 10:13 AM
# @Author  : SkullFang
# @Contact : yzhang.private@gmail.com
# @File    : data_helper_test.py
# @Software: PyCharm
import os
import re
import sys
import json
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
import gensim as gs
from pprint import pprint
from collections import Counter
from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

def clean_str(s):
	s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
	s = re.sub(r" : ", ":", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s.strip().lower()

def load_embeddings(vocabulary):
	word_embeddings = {}
	for word in vocabulary:
		word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
	return word_embeddings

def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
	"""Pad setences during training or prediction"""
	if forced_sequence_length is None: # Train
		sequence_length = max(len(x) for x in sentences)
	else: # Prediction
		logging.critical('This is prediction, reading the trained sequence length')
		sequence_length = forced_sequence_length
	logging.critical('The maximum length is {}'.format(sequence_length))

	padded_sentences = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		num_padding = sequence_length - len(sentence)

		if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
			logging.info('This sentence has to be cut off because it is longer than trained sequence length')
			padded_sentence = sentence[0:sequence_length]
		else:
			padded_sentence = sentence + [padding_word] * num_padding
		padded_sentences.append(padded_sentence)
	return padded_sentences

def build_vocab(sentences):
	word_counts = Counter(itertools.chain(*sentences))
	vocabulary_inv = [word[0] for word in word_counts.most_common()]
	vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
	return vocabulary, vocabulary_inv

def load_data(filename):
	df = pd.read_csv(filename, sep='|')
	selected = ['Category', 'Descript']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how='any', subset=selected)
	df = df.reindex(np.random.permutation(df.index))

	labels = sorted(list(set(df[selected[0]].tolist())))
	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	x_raw= df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

	x_raw = pad_sentences(x_raw)
	vocabulary, vocabulary_inv = build_vocab(x_raw)

	x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
	y = np.array(y_raw)
	return x, y, vocabulary, vocabulary_inv, df, labels

if __name__ == '__main__':
    train_file = 'simple.csv.zip'
    x, y, vocabulary, vocabulary_inv, df, labels = load_data(train_file)
    #clean_str test
    # clean_test=clean_str("The standard powerplant—excluding the F-450—is a 385-hp 6.2-liter V-8 that produces a best-in-class 430 lb-ft of torque")
    # print(clean_test)
    pass