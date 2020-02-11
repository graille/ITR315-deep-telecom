# -*- coding: utf-8 -*-

import numpy as np


class Transmitter():
    def transmitt(self, b):
        # Map symbols
        return 2.*b - 1.;
        
class Receiver():
    def receive(self, c):
        b = list(map(lambda x: 0 if x < 0 else 1, c))
        
        return b

    @staticmethod        
    def calculate_ber(b_wanted, b_received):
        """
            Return the BER
        """
        assert(len(b_wanted) == len(b_received), 'Both input must have the same length')
        return float(np.sum(b_wanted == b_received)) / float(len(b_wanted))

class AWGNChannel:
    def process(self, c, EbN0dB):
        Pn = np.var(c) / 10.**(EbN0dB / 10.)
        return (np.sqrt(Pn / 2.) * np.random.randn(len(c))) + c
        
