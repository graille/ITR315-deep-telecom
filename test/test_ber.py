# -*- coding: utf-8 -*-

import sys
sys.path.append('../')

from src.communications import *
from src.utils import ber_performance
import matplotlib.pyplot as plt

import scipy.stats as sc_stats
import numpy as np

def channel(modulation, b, EbN0dB):
    transmitter = Transmitter()
    channel = AWGNChannel()    
    receiver = Receiver()
    
    c = transmitter.transmitt(b)
    d = channel.process(c, EbN0dB)
    b_r = receiver.receive(d)
    
    return b_r, c, d

EbN0dBs = np.linspace(-15, 10, 10)
BER = ber_performance('BPSK', EbN0dBs, channel, 1000, 100)

ebno = np.power(10*np.ones(len(EbN0dBs)), EbN0dBs/10) 
pe = sc_stats.norm.sf(np.sqrt(2*ebno))

plt.plot(EbN0dBs, pe) # plot the result
plt.plot(EbN0dBs, BER, '-x') # plot the result
plt.legend(['Theory', 'Simulation'])
plt.xlabel('$\\frac{E_b}{N_0}$ in (dB)') # xlabel
plt.ylabel('$P_e$') # ylabel
plt.yscale('log')
plt.grid(True) # grid on 