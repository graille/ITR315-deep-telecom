# -*- coding: utf-8 -*-

import numpy as np


def get_basic_channel_fnt(transmitter, channel, receiver, G=None):
    if G is None:
        G = np.array([[1]])

    factor = 10.0 * np.log10(np.size(G, 1) / np.size(G, 0))

    def channel_fnt(b, EbN0dB):
        EsN0dB = EbN0dB + factor

        c = transmitter.transmit(b)
        d = channel.process(c, EsN0dB)
        b_r = receiver.receive(d)

        return b_r, c, d

    return channel_fnt


def get_all_possible_words(k):
    """
        Return all possible code of length k
    """

    if k == 1:
        yield [0]
        yield [1]
    else:
        for elt in get_all_possible_words(k - 1):
            yield [0] + elt
            yield [1] + elt


def calculate_distance(u1, u2):
    return np.sum((np.array(u1) - np.array(u2)) ** 2)


def ber_performance(EbN0dBs, channel_function, L=1000, target_nb_errors=50):
    BER = np.zeros(len(EbN0dBs))

    for (i, EbN0dB) in enumerate(EbN0dBs):
        nb_errors = 0
        nb_elements = 0

        print("Start EbN0 %d/%d" % (i + 1, len(EbN0dBs)), EbN0dB)

        while nb_errors < target_nb_errors:
            b = np.random.randint(0, 2, L)
            b_r, _, _ = channel_function(b, EbN0dB)

            nb_errors += np.sum(b != b_r)

            nb_elements += L

        BER[i] = float(nb_errors) / float(nb_elements)

    return BER


if __name__ == '__main__':
    for elt in get_all_possible_words(3):
        print(elt)
