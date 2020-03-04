# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from enum import Enum, auto
from multiprocessing import Pool

import numpy as np

from src.utils import calculate_distance, get_all_possible_words


class ChannelComponent:
    def __init__(self, modulation='BPSK', fec_matrix=None):
        if fec_matrix is None:
            fec_matrix = np.array([[1]])

        self.modulation = modulation.upper()
        self.fec_matrix = np.array(fec_matrix)

    @property
    def has_fec_matrix(self):
        return (np.size(self.fec_matrix, 0) > 1) or (np.size(self.fec_matrix, 1) > 1)

    @property
    def block_length(self):
        return np.size(self.fec_matrix, 0)

    @property
    def block_coded_length(self):
        return np.size(self.fec_matrix, 1)

    @property
    def BPS(self):
        rBPS = {
            "BPSK": 1
        }

        return rBPS[self.modulation]


class Transmitter(ChannelComponent):
    def transmit(self, b):
        b = np.array(b)

        # Apply error correction code matrix
        if self.has_fec_matrix:
            nb_blocks = len(b) // self.block_length
            b_t = np.zeros(nb_blocks * self.block_coded_length)

            for i in range(nb_blocks):
                b_l = b[i * self.block_length:(i + 1) * self.block_length]
                b_t[i * self.block_coded_length:(i + 1) * self.block_coded_length] = np.dot(b_l, self.fec_matrix) % 2
        else:
            b_t = b

        # Map symbols
        if self.modulation in ['BPSK']:
            return (2. * b_t) - 1.
        else:
            raise Exception(f"Unknown modulation {self.modulation}")


class ReceiverMode(Enum):
    CLASSIC = auto()
    MAP = auto()
    DEEP_LEARNING = auto()


class Receiver(ChannelComponent):
    def __init__(self, modulation='BPSK', fec_matrix=None, mode=ReceiverMode.MAP):
        super().__init__(modulation, fec_matrix)
        self.mode = mode

        # Pre-calculate coded elements for G
        transmitter = Transmitter(modulation, self.fec_matrix)

        self.block_elements = []
        self.block_coded_elements = []
        for elt in get_all_possible_words(self.block_length):
            self.block_elements.append(elt)
            self.block_coded_elements.append(transmitter.transmit(elt))

    def receive(self, y_n):
        y_n = np.array(y_n)

        if self.mode == ReceiverMode.CLASSIC:
            # If we are using a FEC matrix, we cannot demap directly, we need to decode the error correction code first
            if self.has_fec_matrix:
                raise Exception("You cannot decode directly by using a error correction code")

            # Otherwise, we can demap directly the bits
            if self.modulation in ['BPSK']:
                return np.array(list(map(lambda x: 0 if x < 0 else 1, y_n)))
            else:
                raise Exception(f"Unknown modulation {self.modulation}")
        elif self.mode == ReceiverMode.MAP:
            # If we didn't passed a FEC matrix, we don't need to use that method, a threshold detector is enough
            if not self.has_fec_matrix:
                print("Auto switch mode")
                self.mode = ReceiverMode.CLASSIC
                return self.receive(y_n)

            # Otherwise, we use the MAP detector
            nb_blocks = len(y_n) // self.block_coded_length
            b_r = np.zeros(nb_blocks * self.block_length)

            for i in range(nb_blocks):
                y_n_b = y_n[i * self.block_coded_length:(i + 1) * self.block_coded_length]

                # Apply MAP estimator
                distances = np.array(list(map(lambda x: calculate_distance(y_n_b, x), self.block_coded_elements)))
                b_r[i * self.block_length:(i + 1) * self.block_length] = self.block_elements[int(np.argmin(distances))]

            return b_r


class Channel(ChannelComponent, ABC):
    def __init__(self, modulation='BPSK', fec_matrix=None):
        super().__init__(modulation, fec_matrix)

    @abstractmethod
    def process(self, c, EsN0dB):
        pass


class AWGNChannel(Channel):
    def process(self, c, EbN0dB):
        fec_factor = (np.size(self.fec_matrix, 1) / np.size(self.fec_matrix, 0))

        Pn = (np.var(c) / (10. ** (EbN0dB / 10.))) * (1 / self.BPS) * fec_factor

        return (np.sqrt(Pn / 2.) * np.random.randn(len(c))) + np.array(c)
