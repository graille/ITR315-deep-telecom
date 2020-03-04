# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from src.communications import Transmitter, AWGNChannel, Receiver
from src.utils import ber_performance, get_basic_channel_fct, show_ber

# Configuration
MODULATION = 'BPSK'
EbN0dBs = np.linspace(-15, 8, 10)

# Initialization
transmitter = Transmitter(MODULATION)
channel = AWGNChannel(MODULATION)
receiver = Receiver(MODULATION)

if __name__ == '__main__':
    BER = ber_performance(
        EbN0dBs,
        get_basic_channel_fct(transmitter, channel, receiver),
        1000,
        100
    )

    # Plot results
    plt.figure()
    show_ber(MODULATION, EbN0dBs, BER)
    plt.show()
