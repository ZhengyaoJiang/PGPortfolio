#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
import tflearn


class NeuralNetWork:
    def __init__(self, feature_number, rows, columns, layers, device):
        tf_config = tf.ConfigProto()
        self.session = tf.Session(config=tf_config)
        if device == "cpu":
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0
        else:
            tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
        self.input_num = tf.placeholder(tf.int32, shape=[])
        self.input_tensor = tf.placeholder(tf.float32, shape=[None, feature_number, rows, columns])
        self.previous_w = tf.placeholder(tf.float32, shape=[None, rows])
        self._rows = rows
        self._columns = columns
        self.output = self._build_network(layers)

    def _build_network(self, layers):
        pass


class CNN(NeuralNetWork):
    # input_shape (features, rows, columns)
    def __init__(self, feature_number, rows, columns, layers, device):
        NeuralNetWork.__init__(self, feature_number, rows, columns, layers, device)

    # grenrate the operation, the forward computaion
    def _build_network(self, layers):
        network = tf.transpose(self.input_tensor, [0, 2, 3, 1])
        # [batch, assets, window, features]
        network = network / network[:, :, -1, 0, None, None]
        for layer_number, layer in enumerate(layers):
            if layer["type"] == "DenseLayer":
                network = tflearn.layers.core.fully_connected(network,
                                                              int(layer["neuron_number"]),
                                                              layer["activation_function"],
                                                              regularizer=layer["regularizer"],
                                                              weight_decay=layer["weight_decay"] )
            elif layer["type"] == "DropOut":
                network = tflearn.layers.core.dropout(network, layer["keep_probability"])
            elif layer["type"] == "EIIE_Dense":
                width = network.get_shape()[2]
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 [1, width],
                                                 [1, 1],
                                                 "valid",
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
            elif layer["type"] == "ConvLayer":
                network = tflearn.layers.conv_2d(network, int(layer["filter_number"]),
                                                 allint(layer["filter_shape"]),
                                                 allint(layer["strides"]),
                                                 layer["padding"],
                                                 layer["activation_function"],
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
            elif layer["type"] == "MaxPooling":
                network = tflearn.layers.conv.max_pool_2d(network, layer["strides"])
            elif layer["type"] == "AveragePooling":
                network = tflearn.layers.conv.avg_pool_2d(network, layer["strides"])
            elif layer["type"] == "LocalResponseNormalization":
                network = tflearn.layers.normalization.local_response_normalization(network)
            elif layer["type"] == "EIIE_Output":
                width = network.get_shape()[2]
                network = tflearn.layers.conv_2d(network, 1, [1, width], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                network = network[:, :, 0, 0]
                btc_bias = tf.ones((self.input_num, 1))
                network = tf.concat([btc_bias, network], 1)
                network = tflearn.layers.core.activation(network, activation="softmax")
            elif layer["type"] == "Output_WithW":
                network = tflearn.flatten(network)
                network = tf.concat([network,self.previous_w], axis=1)
                network = tflearn.fully_connected(network, self._rows+1,
                                                  activation="softmax",
                                                  regularizer=layer["regularizer"],
                                                  weight_decay=layer["weight_decay"])
            elif layer["type"] == "EIIE_Output_WithW":
                width = network.get_shape()[2]
                height = network.get_shape()[1]
                features = network.get_shape()[3]
                network = tf.reshape(network, [self.input_num, int(height), 1, int(width*features)])
                w = tf.reshape(self.previous_w, [-1, int(height), 1, 1])
                network = tf.concat([network, w], axis=3)
                network = tflearn.layers.conv_2d(network, 1, [1, 1], padding="valid",
                                                 regularizer=layer["regularizer"],
                                                 weight_decay=layer["weight_decay"])
                network = network[:, :, 0, 0]
                #btc_bias = tf.zeros((self.input_num, 1))
                btc_bias = tf.get_variable("btc_bias", [1, 1], dtype=tf.float32,
                                       initializer=tf.zeros_initializer)
                btc_bias = tf.tile(btc_bias, [self.input_num, 1])
                network = tf.concat([btc_bias, network], 1)
                self.voting = network
                network = tflearn.layers.core.activation(network, activation="softmax")

            elif layer["type"] == "EIIE_LSTM" or\
                            layer["type"] == "EIIE_RNN":
                network = tf.transpose(network, [0, 2, 3, 1])
                resultlist = []
                reuse = False
                for i in range(self._rows):
                    if i > 0:
                        reuse = True
                    if layer["type"] == "EIIE_LSTM":
                        result = tflearn.layers.lstm(network[:, :, :, i],
                                                     int(layer["neuron_number"]),
                                                     dropout=layer["dropouts"],
                                                     scope="lstm"+str(layer_number),
                                                     reuse=reuse)
                    else:
                        result = tflearn.layers.simple_rnn(network[:, :, :, i],
                                                           int(layer["neuron_number"]),
                                                           dropout=layer["dropouts"],
                                                           scope="rnn"+str(layer_number),
                                                           reuse=reuse)
                    resultlist.append(result)
                network = tf.stack(resultlist)
                network = tf.transpose(network, [1, 0, 2])
                network = tf.reshape(network, [-1, self._rows, 1, int(layer["neuron_number"])])
            else:
                raise ValueError("the layer {} not supported.".format(layer["type"]))
        return network


def allint(l):
    return [int(i) for i in l]

