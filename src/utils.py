# -*- coding: utf-8 -*-
import itertools
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sc_stats


def get_bps(modulation):
    if modulation.upper() in ['BPSK']:
        return 1
    else:
        raise Exception(f"Unknown modulation {modulation}")


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


def calculate_noise_power(EbN0dB, var_c, k, n, BPS):
    return (var_c / (10. ** (EbN0dB / 10.))) * (1 / BPS) * (n / k)


def get_all_possible_words(k):
    """
        Return all possible binary words of length k
    """

    return list(itertools.product([0, 1], repeat=k))


def calculate_distance(u1, u2):
    """
    :param u1: A word
    :param u2: Another word
    :return: The square of the hamming distance between u1 and u2
    """
    return np.sum((np.array(u1) - np.array(u2)) ** 2)


def ber_performance(EbN0dBs, channel_function, L=1000, target_nb_errors=50):
    assert (EbN0dBs != []) and (EbN0dBs is not None), "You must specify valid values for EbN0dBs"
    assert callable(channel_function), "The channel function must be callable"
    assert L > 10, "L must be greater than 10"
    assert target_nb_errors > 0, "The targeted number of error must be grater than 0"

    BER = np.zeros(len(EbN0dBs))

    for (i, EbN0dB) in enumerate(EbN0dBs):
        nb_errors = 0
        nb_elements = 0

        print(f"Start EbN0 {i + 1}/{len(EbN0dBs)} [{np.round(EbN0dB, 2)}] /{target_nb_errors} ", end='')
        T_nb_errors = target_nb_errors // 10

        t = time.time()

        while nb_errors < target_nb_errors:
            b = np.random.randint(0, 2, L)
            b_r, _, _ = channel_function(b, EbN0dB)

            nb_errors += np.sum(b != b_r)

            # Print current number of errors
            if nb_errors > T_nb_errors:
                print(f'[{100 * nb_errors / target_nb_errors} %]', end='')
                T_nb_errors += target_nb_errors // 10

            nb_elements += L

        BER[i] = float(nb_errors) / float(nb_elements)
        print(f" | Ended in {np.round(time.time() - t, 2)} s")

    return BER


def show_ber(modulation, EbN0dBs, BER, plot_theory=True):
    modulation = modulation.upper()

    # Get theoretical curve
    if plot_theory:
        if modulation in ['BPSK']:
            EbN0 = np.power(10 * np.ones(len(EbN0dBs)), EbN0dBs / 10)
            pe = sc_stats.norm.sf(np.sqrt(2 * EbN0))

            plt.plot(EbN0dBs, pe)  # plot the result

    plt.plot(EbN0dBs, BER, '-x')  # plot the result

    plt.xlabel('$\\frac{E_b}{N_0}$ in (dB)')
    plt.ylabel('$P_e$')
    plt.yscale('log')
    plt.grid(True)  # grid on


if __name__ == '__main__':
    for elt in get_all_possible_words(3):
        print(elt)
