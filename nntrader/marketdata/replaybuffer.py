from __future__ import division,absolute_import,print_function
import numpy as np
import logging


class ReplayBuffer:
    def __init__(self, start_index, end_index, batch_size, is_permed, coin_number, sample_bias=1.0):
        """
        :param start_index: start index of the training set on the global data matrices
        :param end_index: end index of the training set on the global data matrices
        """
        self.__coin_number = coin_number
        self.__experiences = [Experience(i) for i in range(start_index, end_index)]
        self.__is_permed = is_permed
        # NOTE: in order to achieve the previous w feature
        self.__batch_size = batch_size
        self.__sample_bias = sample_bias
        logging.debug("buffer_bias is %f" % sample_bias)

    def append_experience(self, state_index):
        self.__experiences.append(Experience(state_index))
        logging.debug("a new experience, indexed by %d, was appended" % state_index)

    def __sample(self, start, end, bias):
        """
        @:param end: is excluded
        @:param bias: value in (0, 1)
        """
        # TODO: deal with the case when bias is 0
        ran = np.random.geometric(bias)
        while ran > end - start:
            ran = np.random.geometric(bias)
        result = end - ran
        return result

    def next_experience_batch(self):
        # First get a start point randomly
        batch = []
        if self.__is_permed:
            for i in range(self.__batch_size):
                batch.append(self.__experiences[self.__sample(self.__experiences[0].state_index,
                                                              self.__experiences[-1].state_index,
                                                              self.__sample_bias)])
        else:
            batch_start = self.__sample(0, len(self.__experiences) - self.__batch_size,
                                        self.__sample_bias)
            batch = self.__experiences[batch_start:batch_start+self.__batch_size]
        return batch


class Experience:
    def __init__(self, state_index):
        self.state_index = int(state_index)
