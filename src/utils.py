# -*- coding: utf-8 -*-
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sc_stats


def get_basic_channel_fct(transmitter, channel, receiver):
    def channel_fnt(b, EbN0dB):
        # Encoder
        c = transmitter.transmit(b)
        
        # Channel
        d = channel.process(c, EbN0dB)
        
        # Decoder
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


def ber_performance(EbN0dBs, channel_function, L=1000, target_nb_errors=50, **kwargs):
    BER = np.zeros(len(EbN0dBs))

    for (i, EbN0dB) in enumerate(EbN0dBs):
        nb_errors = 0
        nb_elements = 0

        print(f"Start EbN0 {i + 1}/{len(EbN0dBs)} [{np.round(EbN0dB,2)}] /{target_nb_errors} ", end='')
        t = time.time()

        while nb_errors < target_nb_errors:
            b = np.random.randint(0, 2, L)
            b_r, _, _ = channel_function(b, EbN0dB)

            old_nb_errors = nb_errors
            nb_errors += np.sum(b != b_r)

            # Print current number of errors
            if nb_errors > old_nb_errors:
                print(f'[{nb_errors}]', end='')

            nb_elements += L

        BER[i] = float(nb_errors) / float(nb_elements)
        print(f" | Ended in {np.round(time.time() - t, 2)} s")

    return BER


def get_fec_matrix(name):
    name = name.lower()
    if name == 'polar_8_16':
        return np.array(
            [[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
             [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
             [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
             [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
             [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        )
    elif name == 'hamming_7_4':
        return np.array(
            [[1, 0, 0, 0, 1, 1, 0],
             [0, 1, 0, 0, 0, 1, 1],
             [0, 0, 1, 0, 1, 0, 1],
             [0, 0, 0, 1, 1, 1, 1]]
        )
    else:
        return np.array([[1]])


def show_ber(modulation, EbN0dBs, BER):
    modulation = modulation.upper()

    # Get theoretical curve
    if modulation in ['BPSK']:
        EbN0 = np.power(10 * np.ones(len(EbN0dBs)), EbN0dBs / 10)
        pe = sc_stats.norm.sf(np.sqrt(2 * EbN0))

#         plt.plot(EbN0dBs, pe)  # plot the result

    plt.plot(EbN0dBs, BER, '-x')  # plot the result

#     plt.legend(['Theory', 'Simulation'])
    plt.xlabel('$\\frac{E_b}{N_0}$ in (dB)')
    plt.ylabel('$P_e$')
    plt.yscale('log')
    plt.grid(True)  # grid on


if __name__ == '__main__':
    for elt in get_all_possible_words(3):
        print(elt)
