#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import os
import time
import collections
import tflearn
import numpy as np
import pandas as pd
import tensorflow as tf
from pgportfolio.learn.nnagent import NNAgent
from pgportfolio.marketdata.datamatrices import DataMatrices
import logging
Result = collections.namedtuple("Result",
                                [
                                 "test_pv",
                                 "test_log_mean",
                                 "test_log_mean_free",
                                 "test_history",
                                 "config",
                                 "net_dir",
                                 "backtest_test_pv",
                                 "backtest_test_history",
                                 "backtest_test_log_mean",
                                 "training_time"])

class TraderTrainer:
    def __init__(self, config, fake_data=False, restore_dir=None, save_path=None, device="cpu",
                 agent=None):
        """
        :param config: config dictionary
        :param fake_data: if True will use data generated randomly
        :param restore_dir: path to the model trained before
        :param save_path: path to save the model
        :param device: the device used to train the network
        :param agent: the nnagent object. If this is provides, the trainer will not
        create a new agent by itself. Therefore the restore_dir will not affect anything.
        """
        self.config = config
        self.train_config = config["training"]
        self.input_config = config["input"]
        self.save_path = save_path
        self.best_metric = 0
        np.random.seed(config["random_seed"])

        self.__window_size = self.input_config["window_size"]
        self.__coin_number = self.input_config["coin_number"]
        self.__batch_size = self.train_config["batch_size"]
        self.__snap_shot = self.train_config["snap_shot"]
        config["input"]["fake_data"] = fake_data

        self._matrix = DataMatrices.create_from_config(config)

        self.test_set = self._matrix.get_test_set()
        if not config["training"]["fast_train"]:
            self.training_set = self._matrix.get_training_set()
        self.upperbound_validation = 1
        self.upperbound_test = 1
        tf.set_random_seed(self.config["random_seed"])
        self.device = device
        if agent:
            self._agent = agent
        else:
            if device == "cpu":
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                with tf.device("/cpu:0"):
                    self._agent = NNAgent(config, restore_dir, device)
            else:
                self._agent = NNAgent(config, restore_dir, device)

    def _evaluate(self, set_name, *tensors):
        if set_name == "test":
            feed = self.test_set
        elif set_name == "training":
            feed = self.training_set
        else:
            raise ValueError()
        result = self._agent.evaluate_tensors(feed["X"],feed["y"],last_w=feed["last_w"],
                                              setw=feed["setw"], tensors=tensors)
        return result

    @staticmethod
    def calculate_upperbound(y):
        array = np.maximum.reduce(y[:, 0, :], 1)
        total = 1.0
        for i in array:
            total = total * i
        return total

    def log_between_steps(self, step):
        fast_train = self.train_config["fast_train"]
        tflearn.is_training(False, self._agent.session)

        summary, v_pv, v_log_mean, v_loss, log_mean_free, weights= \
            self._evaluate("test", self.summary,
                           self._agent.portfolio_value,
                           self._agent.log_mean,
                           self._agent.loss,
                           self._agent.log_mean_free,
                           self._agent.portfolio_weights)
        self.test_writer.add_summary(summary, step)

        if not fast_train:
            summary, loss_value = self._evaluate("training", self.summary, self._agent.loss)
            self.train_writer.add_summary(summary, step)

        # print 'ouput is %s' % out
        logging.info('='*30)
        logging.info('step %d' % step)
        logging.info('-'*30)
        if not fast_train:
            logging.info('training loss is %s\n' % loss_value)
        logging.info('the portfolio value on test set is %s\nlog_mean is %s\n'
                     'loss_value is %3f\nlog mean without commission fee is %3f\n' % \
                     (v_pv, v_log_mean, v_loss, log_mean_free))
        logging.info('='*30+"\n")

        if not self.__snap_shot:
            self._agent.save_model(self.save_path)
        elif v_pv > self.best_metric:
            self.best_metric = v_pv
            logging.info("get better model at %s steps,"
                         " whose test portfolio value is %s" % (step, v_pv))
            if self.save_path:
                self._agent.save_model(self.save_path)
        self.check_abnormal(v_pv, weights)

    def check_abnormal(self, portfolio_value, weigths):
        if portfolio_value == 1.0:
            logging.info("average portfolio weights {}".format(weigths.mean(axis=0)))


    def next_batch(self):
        batch = self._matrix.next_batch()
        batch_input = batch["X"]
        batch_y = batch["y"]
        batch_last_w = batch["last_w"]
        batch_w = batch["setw"]
        return batch_input, batch_y, batch_last_w, batch_w

    def __init_tensor_board(self, log_file_dir):
        tf.summary.scalar('benefit', self._agent.portfolio_value)
        tf.summary.scalar('log_mean', self._agent.log_mean)
        tf.summary.scalar('loss', self._agent.loss)
        tf.summary.scalar("log_mean_free", self._agent.log_mean_free)
        self.summary = tf.summary.merge_all()
        location = log_file_dir
        self.network_writer = tf.summary.FileWriter(location + '/network',
                                                    self._agent.session.graph)
        self.test_writer = tf.summary.FileWriter(location + '/test')
        self.train_writer = tf.summary.FileWriter(location + '/train')

    def __print_upperbound(self):
        upperbound_test = self.calculate_upperbound(self.test_set["y"])
        logging.info("upper bound in test is %s" % upperbound_test)

    def train_net(self, log_file_dir="./tensorboard", index="0"):
        """
        :param log_file_dir: logging of the training process
        :param index: sub-folder name under train_package
        :return: the result named tuple
        """
        self.__print_upperbound()
        if log_file_dir:
            if self.device == "cpu":
                with tf.device("/cpu:0"):
                    self.__init_tensor_board(log_file_dir)
            else:
                self.__init_tensor_board(log_file_dir)
        starttime = time.time()

        total_data_time = 0
        total_training_time = 0
        for i in range(self.train_config["steps"]):
            step_start = time.time()
            x, y, last_w, setw = self.next_batch()
            finish_data = time.time()
            total_data_time += (finish_data - step_start)
            self._agent.train(x, y, last_w=last_w, setw=setw)
            total_training_time += time.time() - finish_data
            if i % 1000 == 0 and log_file_dir:
                logging.info("average time for data accessing is %s"%(total_data_time/1000))
                logging.info("average time for training is %s"%(total_training_time/1000))
                total_training_time = 0
                total_data_time = 0
                self.log_between_steps(i)

        if self.save_path:
            self._agent.recycle()
            best_agent = NNAgent(self.config, restore_dir=self.save_path)
            self._agent = best_agent

        pv, log_mean = self._evaluate("test", self._agent.portfolio_value, self._agent.log_mean)
        logging.warning('the portfolio value train No.%s is %s log_mean is %s,'
                        ' the training time is %d seconds' % (index, pv, log_mean, time.time() - starttime))

        return self.__log_result_csv(index, time.time() - starttime)

    def __log_result_csv(self, index, time):
        from pgportfolio.trade import backtest
        dataframe = None
        csv_dir = './train_package/train_summary.csv'
        tflearn.is_training(False, self._agent.session)
        v_pv, v_log_mean, benefit_array, v_log_mean_free =\
            self._evaluate("test",
                           self._agent.portfolio_value,
                           self._agent.log_mean,
                           self._agent.pv_vector,
                           self._agent.log_mean_free)

        backtest = backtest.BackTest(self.config.copy(),
                                     net_dir=None,
                                     agent=self._agent)

        backtest.start_trading()
        result = Result(test_pv=[v_pv],
                        test_log_mean=[v_log_mean],
                        test_log_mean_free=[v_log_mean_free],
                        test_history=[''.join(str(e)+', ' for e in benefit_array)],
                        config=[json.dumps(self.config)],
                        net_dir=[index],
                        backtest_test_pv=[backtest.test_pv],
                        backtest_test_history=[''.join(str(e)+', ' for e in backtest.test_pc_vector)],
                        backtest_test_log_mean=[np.mean(np.log(backtest.test_pc_vector))],
                        training_time=int(time))
        new_data_frame = pd.DataFrame(result._asdict()).set_index("net_dir")
        if os.path.isfile(csv_dir):
            dataframe = pd.read_csv(csv_dir).set_index("net_dir")
            dataframe = dataframe.append(new_data_frame)
        else:
            dataframe = new_data_frame
        if int(index) > 0:
            dataframe.to_csv(csv_dir)
        return result

