# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

from src.communications import Transmitter, AWGNChannel, Receiver
from src.utils import *

# Configuration
MODULATION = 'BPSK'
EbN0dBs = np.linspace(-20, 8, 20)

# Initialization
transmitter = Transmitter(MODULATION)
receiver = Receiver(MODULATION)

channel = AWGNChannel(get_bps(MODULATION), transmitter.block_length, transmitter.block_coded_length)

if __name__ == '__main__':
    BER = ber_performance(
        EbN0dBs,
        get_basic_channel_fct(transmitter, channel, receiver),
        1000,
        500
    )

    # Plot results
    plt.figure()
    show_ber(MODULATION, EbN0dBs, BER)
    plt.legend(['BPSK Theory', 'BPSK simulation'])
    plt.show()