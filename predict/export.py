# !/usr/bin/env python2.7
"""Export text classification model given existing training pb file.

The model is exported as SavedModel with proper signatures that can be loaded by
standard tensorflow_model_server.
"""

import os.path
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import utils
from cnn_rnn_config import CNNRNNConfig as model_config

tf.app.flags.DEFINE_string('pb_dir', 'origin_model/graph.pb',
                           """Directory where to read training pb.""")
tf.app.flags.DEFINE_string('output_dir', 'empty',
                           """Directory where to export inference model.""")
tf.app.flags.DEFINE_integer('model_version', 1,
                            """Version number of the model.""")

FLAGS = tf.app.flags.FLAGS

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
folder_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
graph_file = folder_path + "/trained_results/graph.pb"


def real_len(batches):
    return [np.int32(np.ceil(np.argmin(batch + [0]) * 1.0 / model_config.max_pool_size)) for batch in batches]


def export():
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        with tf.gfile.FastGFile(graph_file, 'rb') as f:
            print(graph_file)
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

            x_batch = tf.placeholder(tf.int32, [None, model_config.forced_sequence_length], name='input_x')
            batch_size = tf.placeholder(tf.int32, [], name='batch_size')
            pad = tf.placeholder(tf.float32, [None, 1, model_config.embedding_dim, 1], name='pad')
            real = tf.placeholder(tf.int32, [None], name='real_len')
            # batch_size = len(x_batch)
            # pad = np.zeros([len(x_batch), 1, model_config.embedding_dim, 1], dtype=np.float32)
            # real = real_len(x_batch)

            output1, output2, output3 = tf.import_graph_def(graph_def,
                                                            input_map={'input_x': x_batch,
                                                                       'dropout_keep_prob': model_config.dropout_keep_prob,
                                                                       'batch_size': batch_size,
                                                                       'pad': pad,
                                                                       'real_len': real},
                                                            return_elements=['labels:0', 'scores/scores:0',
                                                                             'scores/predictions:0'],
                                                            name='output')

            # x_train = tf.placeholder(dtype=tf.int32, shape=[None, 600], name='input_x')
            # output = tf.import_graph_def(graph_def, input_map={'input_x': x_train, 'keep_prob': 0.5},
            #                              return_elements=['score/logits:0', 'score/softmax_logits:0',
            #                                               'score/y_pred_cls:0'], name='output')
            # output = tf.import_graph_def(graph_def, input_map={'input_x': x_train},
            #                              return_elements=['score/logits:0', 'score/softmax_logits:0',
            #                                               'score/y_pred_cls:0'],
            #                              name='output')
            #
            output_path = os.path.join(tf.compat.as_bytes(FLAGS.output_dir),
                                       tf.compat.as_bytes(str(FLAGS.model_version)))
            # print('Exporting trained model to ', output_path)

            builder = tf.saved_model.builder.SavedModelBuilder(output_path)

            label_output_constant_info = tf.saved_model.utils.build_tensor_info(output1)
            scores_output_tensor_info = tf.saved_model.utils.build_tensor_info(output2)
            label_output_tensor_info = tf.saved_model.utils.build_tensor_info(output3)

            tensor_info_x_train = utils.build_tensor_info(x_batch)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        'text': tensor_info_x_train
                    },
                    outputs={
                        'label_output_constant_info': label_output_constant_info,
                        'scores_output_tensor_info': scores_output_tensor_info,
                        'label_output_tensor_info': label_output_tensor_info

                    },
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
                ))

            legacy_init_op = tf.group(
                tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    'predict_label':
                        prediction_signature
                },
                legacy_init_op=legacy_init_op)

            builder.save()
            print('Successfully exported model to %s' % FLAGS.output_dir)


def main(unused_argv=None):
    export()


if __name__ == '__main__':
    tf.app.run()
