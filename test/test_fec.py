# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from src.communications import Transmitter, AWGNChannel, Receiver, ReceiverMode
from src.utils import ber_performance, get_basic_channel_fct, show_ber, get_fec_matrix

# Configuration
MODULATION = 'BPSK'
EbN0dBs = np.linspace(-50, 7, 20)
G = get_fec_matrix('POLAR_8_16')

# Initialization
transmitter = Transmitter(MODULATION, G)
channel = AWGNChannel(MODULATION, G)
receiver = Receiver(MODULATION, G, ReceiverMode.MAP)

if __name__ == '__main__':
    BER = ber_performance(
        EbN0dBs[::-1],
        get_basic_channel_fct(transmitter, channel, receiver),
        np.size(G, 0) * 1000,
        500
    )[::-1]

    # # Plot results
    # plt.figure()
    # show_ber(MODULATION, EbN0dBs, BER)
    # plt.show()

    np.savetxt('outputs/BER_G_MAP.csv', [
        np.array(EbN0dBs),
        np.array(BER)
    ], delimiter=',')
