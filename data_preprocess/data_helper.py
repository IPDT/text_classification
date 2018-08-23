from data_preprocess.mysql_config import MySQLConfig
from collections import Counter
from data_preprocess.normalize_data import NormalizeData
import numpy as np
import itertools
import os
import json


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_embeddings(vocabulary_dict, vocabulary_list, embedding_dim):
    word_embeddings = {}
    for word in vocabulary_list:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, embedding_dim)
    embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary_dict)]
    embedding_mat = np.array(embedding_mat, dtype=np.float32)
    return embedding_mat


def load_data(forced_sequence_length, vocab_file):
    mysql_config = MySQLConfig()
    raw_label, raw_content = mysql_config.read_raw_data()
    print('---------------read data from db successfully---------------')
    normalize_data = NormalizeData(raw_content)
    normalized_content = normalize_data.normalize_data()
    print('----------------normalize data successfully-----------------')
    padding_content = pad_sentences(normalized_content, forced_sequence_length)
    print('-------------------pad data successfully--------------------')
    vocabulary_list, vocabulary_dict = build_vocab(vocab_file, raw_content=padding_content)
    print('---------------build vocabulary successfully----------------')
    x = np.array([[vocabulary_dict[word] for word in sentence] for sentence in padding_content])
    y, labels = get_one_hot_label(raw_label)
    return x, y, vocabulary_list, vocabulary_dict, labels


def build_vocab(vocab_file, raw_content=None, custom_content=None):
    vocabulary_list = []
    if os.path.exists(vocab_file):
        f_vocab = open(vocab_file, 'r', encoding='utf=8')
        vocabulary_list = json.load(f_vocab)
    elif not os.path.exists(vocab_file) and raw_content is not None:
        word_counts = Counter(itertools.chain(*raw_content))
        vocabulary_list = [word[0] for word in word_counts.most_common()]
        f_vocab = open(vocab_file, 'w', encoding='utf=8')
        f_vocab.write(json.dumps(vocabulary_list))
    if custom_content is not None:
        for sentence in custom_content:
            for word in sentence:
                if word not in vocabulary_list:
                    vocabulary_list.append(word)
    vocabulary_dict = {word: index for index, word in enumerate(vocabulary_list)}
    return vocabulary_list, vocabulary_dict


def get_one_hot_label(raw_labels):
    labels = sorted(list(set(raw_labels)))
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))
    y_raw = []
    for label in raw_labels:
        y_raw.append(label_dict[label])
    return np.array(y_raw), labels


def get_data_from_db():
    mysql_config = MySQLConfig()
    raw_label, raw_content = mysql_config.read_raw_data()
    return raw_label, raw_content


def pad_sentences(content, forced_sequence_length, padding_word='<PAD/>'):
    print('The maximum length is {}'.format(forced_sequence_length))

    padding_content = []
    for i in range(len(content)):
        sentence = content[i]
        num_padding = forced_sequence_length - len(sentence)
        if num_padding < 0:
            padding_sentence = sentence[0:forced_sequence_length]
        else:
            padding_sentence = sentence + [padding_word] * num_padding
        padding_content.append(padding_sentence)
    return padding_content
