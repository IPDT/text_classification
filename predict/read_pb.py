import tensorflow as tf
import os
import numpy as np
import json
import itertools
import re
from collections import Counter


class CNNRNNConfig(object):
    forced_sequence_length = 500
    batch_size = 128
    dropout_keep_prob = 0.5
    embedding_dim = 300
    evaluate_every = 200
    filter_sizes = "3,4,5"
    hidden_unit = 300
    l2_reg_lambda = 0.0
    max_pool_size = 4
    non_static = False
    num_filters = 32
    num_epochs = 1


def real_len(batches):
    return [np.int32(np.ceil(np.argmin(batch + [0]) * 1.0 / model_config.max_pool_size)) for batch in batches]


def build_vocab(vocab_file, raw_content=None, custom_content=None):
    vocabulary_list = []
    if os.path.exists(vocab_file):
        f_vocab = open(vocab_file, 'r', encoding='utf-8')
        vocabulary_list = json.load(f_vocab)
    elif not os.path.exists(vocab_file) and raw_content is not None:
        word_counts = Counter(itertools.chain(*raw_content))
        vocabulary_list = [word[0] for word in word_counts.most_common()]
        f_vocab = open(vocab_file, 'w', encoding='utf-8')
        f_vocab.write(json.dumps(vocabulary_list))
    if custom_content is not None:
        for sentence in custom_content:
            for word in sentence:
                if word not in vocabulary_list:
                    vocabulary_list.append(word)
    vocabulary_dict = {word: index for index, word in enumerate(vocabulary_list)}
    return vocabulary_list, vocabulary_dict


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


def native_content(content):
    punc = '[.!\'?:"@#$%^&*()+=_:;“”‘’]'
    no_punc_content = re.sub(punc, ' ', content.replace('\n', ''))
    return re.sub(r'\s+', ' ', no_punc_content)


folder_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
model_config = CNNRNNConfig()

# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True
with tf.Session() as sess:
    forced_sequence_length = model_config.forced_sequence_length
    # with tf.device('/gpu:1'):
    graph = open(folder_path+'/trained_results/graph.pb', 'rb')
    f_test = open(folder_path + '/predict/test.txt', 'r', encoding='utf-8')
    raw_content = [list(native_content(line).split()) for line in f_test]
    padding_content = pad_sentences(raw_content, forced_sequence_length)
    vocabulary_list, vocabulary_dict = build_vocab(folder_path + '/data_preprocess/vocab_en.txt',
                                                   raw_content=padding_content, custom_content=raw_content)
    x = [[vocabulary_dict[word] for word in sentence] for sentence in padding_content]
    x_test, y_test = np.asarray(x), None
    train_batches = batch_iter(list(x_test), model_config.batch_size, 1, shuffle=False)

    graph_def = tf.GraphDef()
    graph_def.ParseFromString(graph.read())

    for x_batch in train_batches:
        x_batch = x_batch.astype(np.int32)
        dropout_keep_prob = model_config.dropout_keep_prob
        batch_size = len(x_batch)
        pad = np.zeros([len(x_batch), 1, model_config.embedding_dim, 1], dtype=np.float32)
        real = real_len(x_batch)
        output1, output2, output3 = tf.import_graph_def(graph_def,
                                                        input_map={'input_x': x_batch,
                                                                   'dropout_keep_prob': dropout_keep_prob,
                                                                   'batch_size': batch_size,
                                                                   'pad': pad,
                                                                   'real_len': real},
                                                        return_elements=['labels:0', 'scores/scores:0',
                                                                         'scores/predictions:0'],
                                                        name='output')
        labels = sess.run(output1)
        result2 = sess.run(output2)
        result3 = sess.run(output3)

        for result in result3:
            print(labels[result])
