from __future__ import absolute_import, print_function, division
import tflearn
import tensorflow as tf
import numpy as np
from pgportfolio.constants import *
import pgportfolio.learn.network as network

class NNAgent:
    def __init__(self, config, restore_dir=None, device="cpu"):
        self.__config = config
        self.__coin_number = config["input"]["coin_number"]
        self.__net = network.CNN(config["input"]["feature_number"],
                                 self.__coin_number,
                                 config["input"]["window_size"],
                                 config["layers"],
                                 device=device)
        self.__global_step = tf.Variable(0, trainable=False)
        self.__train_operation = None
        self.__y = tf.placeholder(tf.float32, shape=[None,
                                                     self.__config["input"]["feature_number"],
                                                     self.__coin_number])
        self.__future_price = tf.concat([tf.ones([self.__net.input_num, 1]),
                                       self.__y[:, 0, :]], 1)
        self.__future_omega = (self.__future_price * self.__net.output) /\
                              tf.reduce_sum(self.__future_price * self.__net.output, axis=1)[:, None]
        # tf.assert_equal(tf.reduce_sum(self.__future_omega, axis=1), tf.constant(1.0))
        self.__commission_ratio = self.__config["trading"]["trading_consumption"]
        self.__pv_vector = tf.reduce_sum(self.__net.output * self.__future_price, reduction_indices=[1]) *\
                           (tf.concat([tf.ones(1), self.__pure_pc()], axis=0))
        self.__log_mean_free = tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output * self.__future_price,
                                                                   reduction_indices=[1])))
        self.__portfolio_value = tf.reduce_prod(self.__pv_vector)
        self.__mean = tf.reduce_mean(self.__pv_vector)
        self.__log_mean = tf.reduce_mean(tf.log(self.__pv_vector))
        self.__standard_deviation = tf.sqrt(tf.reduce_mean((self.__pv_vector - self.__mean) ** 2))
        self.__sharp_ratio = (self.__mean - 1) / self.__standard_deviation
        self.__loss = self.__set_loss_function()
        self.__train_operation = self.init_train(learning_rate=self.__config["training"]["learning_rate"],
                                                 decay_steps=self.__config["training"]["decay_steps"],
                                                 decay_rate=self.__config["training"]["decay_rate"],
                                                 training_method=self.__config["training"]["training_method"])
        self.__saver = tf.train.Saver()
        if restore_dir:
            self.__saver.restore(self.__net.session, restore_dir)
        else:
            self.__net.session.run(tf.global_variables_initializer())

    @property
    def session(self):
        return self.__net.session

    @property
    def pv_vector(self):
        return self.__pv_vector

    @property
    def standard_deviation(self):
        return self.__standard_deviation

    @property
    def portfolio_weights(self):
        return self.__net.output

    @property
    def sharp_ratio(self):
        return self.__sharp_ratio

    @property
    def log_mean(self):
        return self.__log_mean

    @property
    def log_mean_free(self):
        return self.__log_mean_free

    @property
    def portfolio_value(self):
        return self.__portfolio_value

    @property
    def loss(self):
        return self.__loss

    def recycle(self):
        tf.reset_default_graph()
        self.__net.session.close()

    def __set_loss_function(self):
        def loss_function4():
            return -tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output[:] * self.__future_price,
                                                        reduction_indices=[1])))

        def loss_function5():
            return -tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output * self.__future_price, reduction_indices=[1]))) + \
                   LAMBDA * tf.reduce_mean(tf.reduce_sum(-tf.log(1 + 1e-6 - self.__net.output), reduction_indices=[1]))

        def loss_function6():
            return -tf.reduce_mean(tf.log(self.pv_vector))

        def loss_function7():
            return -tf.reduce_mean(tf.log(self.pv_vector)) + \
                   LAMBDA * tf.reduce_mean(tf.reduce_sum(-tf.log(1 + 1e-6 - self.__net.output), reduction_indices=[1]))

        def with_last_w():
            return -tf.reduce_mean(tf.log(tf.reduce_sum(self.__net.output[:] * self.__future_price, reduction_indices=[1])
                                          -tf.reduce_sum(tf.abs(self.__net.output[:, 1:] - self.__net.previous_w)
                                                         *self.__commission_ratio, reduction_indices=[1])))

        loss_function = loss_function5
        if self.__config["training"]["loss_function"] == "loss_function4":
            loss_function = loss_function4
        elif self.__config["training"]["loss_function"] == "loss_function5":
            loss_function = loss_function5
        elif self.__config["training"]["loss_function"] == "loss_function6":
            loss_function = loss_function6
        elif self.__config["training"]["loss_function"] == "loss_function7":
            loss_function = loss_function7
        elif self.__config["training"]["loss_function"] == "loss_function8":
            loss_function = with_last_w

        loss_tensor = loss_function()
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if regularization_losses:
            for regularization_loss in regularization_losses:
                loss_tensor += regularization_loss
        return loss_tensor

    def init_train(self, learning_rate, decay_steps, decay_rate, training_method):
        learning_rate = tf.train.exponential_decay(learning_rate, self.__global_step,
                                                   decay_steps, decay_rate, staircase=True)
        if training_method == 'GradientDescent':
            train_step = tf.train.GradientDescentOptimizer(learning_rate).\
                         minimize(self.__loss, global_step=self.__global_step)
        elif training_method == 'Adam':
            train_step = tf.train.AdamOptimizer(learning_rate).\
                         minimize(self.__loss, global_step=self.__global_step)
        elif training_method == 'RMSProp':
            train_step = tf.train.RMSPropOptimizer(learning_rate).\
                         minimize(self.__loss, global_step=self.__global_step)
        else:
            raise ValueError()
        return train_step

    def train(self, x, y, last_w, setw):
        tflearn.is_training(True, self.__net.session)
        self.evaluate_tensors(x, y, last_w, setw, [self.__train_operation])

    def evaluate_tensors(self, x, y, last_w, setw, tensors):
        """
        :param x:
        :param y:
        :param last_w:
        :param setw: a function, pass the output w to it to fill the PVM
        :param tensors:
        :return:
        """
        tensors = list(tensors)
        tensors.append(self.__net.output)
        assert not np.any(np.isnan(x))
        assert not np.any(np.isnan(y))
        assert not np.any(np.isnan(last_w)),\
            "the last_w is {}".format(last_w)
        results = self.__net.session.run(tensors,
                                         feed_dict={self.__net.input_tensor: x,
                                                    self.__y: y,
                                                    self.__net.previous_w: last_w,
                                                    self.__net.input_num: x.shape[0]})
        setw(results[-1][:, 1:])
        return results[:-1]

    # save the variables path including file name
    def save_model(self, path):
        self.__saver.save(self.__net.session, path)

    # consumption vector (on each periods)
    def __pure_pc(self):
        c = self.__commission_ratio
        w_t = self.__future_omega[:self.__net.input_num-1]  # rebalanced
        w_t1 = self.__net.output[1:self.__net.input_num]
        mu = 1 - tf.reduce_sum(tf.abs(w_t1[:, 1:]-w_t[:, 1:]), axis=1)*c
        """
        mu = 1-3*c+c**2

        def recurse(mu0):
            factor1 = 1/(1 - c*w_t1[:, 0])
            if isinstance(mu0, float):
                mu0 = mu0
            else:
                mu0 = mu0[:, None]
            factor2 = 1 - c*w_t[:, 0] - (2*c - c**2)*tf.reduce_sum(
                tf.nn.relu(w_t[:, 1:] - mu0 * w_t1[:, 1:]), axis=1)
            return factor1*factor2

        for i in range(20):
            mu = recurse(mu)
        """
        return mu

    # the history is a 3d matrix, return a asset vector
    def decide_by_history(self, history, last_w):
        assert isinstance(history, np.ndarray),\
            "the history should be a numpy array, not %s" % type(history)
        assert not np.any(np.isnan(last_w))
        assert not np.any(np.isnan(history))
        tflearn.is_training(False, self.session)
        history = history[np.newaxis, :, :, :]
        return np.squeeze(self.session.run(self.__net.output, feed_dict={self.__net.input_tensor: history,
                                                                         self.__net.previous_w: last_w[np.newaxis, 1:],
                                                                         self.__net.input_num: 1}))
