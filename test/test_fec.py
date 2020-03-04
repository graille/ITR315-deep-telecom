# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from src.communications import Transmitter, AWGNChannel, Receiver, ReceiverMode
from src.utils import ber_performance, get_basic_channel_fct, show_ber

# Configuration
MODULATION = 'BPSK'
EbN0dBs = np.linspace(-15, 7, 10)

# Course
G = np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
              [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
              [1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
              [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

# # Hamming
# G = np.array([[1, 0, 0, 0, 1, 1, 0],
#               [0, 1, 0, 0, 0, 1, 1],
#               [0, 0, 1, 0, 1, 0, 1],
#               [0, 0, 0, 1, 1, 1, 1]])
#
# # Repetition
# G = np.ones((1, 5))

# Initialization
transmitter = Transmitter(MODULATION, G)
channel = AWGNChannel(MODULATION, G)
receiver = Receiver(MODULATION, G, ReceiverMode.MAP)

if __name__ == '__main__':
    BER = ber_performance(
        EbN0dBs,
        get_basic_channel_fct(transmitter, channel, receiver),
        np.size(G, 0) * 500,
        20
    )

    # Plot results
    plt.figure()
    show_ber(MODULATION, EbN0dBs, BER)
    plt.show()
