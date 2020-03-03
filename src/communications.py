# -*- coding: utf-8 -*-

import numpy as np


class Transmitter:
    def __init__(self, modulation='BPSK'):
        self.modulation = modulation

    def transmitt(self, b):
        # Map symbols
        return 2. * b - 1.

    @staticmethod
    def get_all_possible_words(k):
        """
            Return all possible code of length k
        """

        if k == 1:
            yield [0]
            yield [1]
        else:
            for elt in Transmitter.get_all_possible_words(k - 1):
                yield [0] + elt
                yield [1] + elt

    @staticmethod
    def calculate_distance(u1, u2):
        r = 0
        for i in range(len(u1)):
            r += (u1[i] - u2[i]) ** 2

        return r

    def apply_map(self, y_n, G):
        """

        :param y_n: Symbols such as len(y_n) = size(G, 1) / M
        :param G:
        :param transmitter:
        :return:
        """
        block_length = np.size(G, 0)

        v_min = float('inf')
        arg_min = -1

        for elt in Transmitter.get_all_possible_words(block_length):
            b_coded = np.array(elt).dot(G)
            x_n_estimated = self.transmitt(b_coded)

            d = np.sum((np.array(x_n_estimated) - np.array(y_n)) ** 2)

            if d < v_min:
                arg_min = elt
                v_min = d

        return arg_min


class Receiver:
    def receive(self, c):
        b = list(map(lambda x: 0 if x < 0 else 1, c))

        return b

    @staticmethod
    def calculate_ber(b_wanted, b_received):
        """
            Return the BER
        """
        assert ((len(b_wanted) == len(b_received)), 'Both input must have the same length')
        return float(np.sum(b_wanted == b_received)) / float(len(b_wanted))


class AWGNChannel:
    def process(self, c, EbN0dB):
        Pn = np.var(c) / 10. ** (EbN0dB / 10.)
        return (np.sqrt(Pn / 2.) * np.random.randn(len(c))) + c


if __name__ == '__main__':
    for elt in Transmitter.get_all_possible_words(3):
        print(elt)
