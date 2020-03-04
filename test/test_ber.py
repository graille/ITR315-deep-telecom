# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sc_stats

from src.communications import Transmitter, AWGNChannel, Receiver, ReceiverMode
from src.utils import ber_performance, get_basic_channel_fnt

# Configuration
MODULATION = 'BPSK'
EbN0dBs = np.linspace(-15, 8, 10)

# Initialization
transmitter = Transmitter(MODULATION)
channel = AWGNChannel()
receiver = Receiver(MODULATION)

if __name__ == '__main__':
    BER = ber_performance(EbN0dBs, get_basic_channel_fnt(transmitter, channel, receiver), 1000, 100)

    # Get theoretical curve
    EbN0 = np.power(10 * np.ones(len(EbN0dBs)), EbN0dBs / 10)
    pe = sc_stats.norm.sf(np.sqrt(2 * EbN0))

    # Plot result
    plt.figure()

    plt.plot(EbN0dBs, pe)  # plot the result
    plt.plot(EbN0dBs, BER, '-x')  # plot the result
    plt.legend(['Theory', 'Simulation'])
    plt.xlabel('$\\frac{E_b}{N_0}$ in (dB)')
    plt.ylabel('$P_e$')
    plt.yscale('log')
    plt.grid(True)  # grid on
    plt.show()
