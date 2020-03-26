# -*- coding: utf-8 -*-
from enum import Enum, auto

import numpy as np

from src.utils import calculate_distance, get_all_possible_words, calculate_noise_power


class NeuralNetworkType(Enum):
    ONE_HOT = auto()  # n inputs, 2**k outputs, used for classification
    INTEGER = auto()  # n inputs, k outputs, used for determination


class ChannelComponent:
    def __init__(self, modulation=None, fec_matrix=None):
        self.modulation = modulation.upper() if modulation is not None else None
        self.fec_matrix = np.array([[1]]) if fec_matrix is None else np.array(fec_matrix)

    @property
    def has_fec_matrix(self):
        """
        :return: True if a FEC matrix has been specified, False otherwise
        """
        return (np.size(self.fec_matrix, 0) > 1) or (np.size(self.fec_matrix, 1) > 1)

    @property
    def block_length(self):
        """
        :return: The length of non-coded bit blocks (k)
        """
        return np.size(self.fec_matrix, 0)

    @property
    def block_coded_length(self):
        """
        :return: The length of coded bit blocks (n)
        """
        return np.size(self.fec_matrix, 1)


# ----------------------------------------------------------------------------------------------------------------------
# Transmitters
# ----------------------------------------------------------------------------------------------------------------------

class Transmitter(ChannelComponent):
    def __init__(self, modulation=None, fec_matrix=None):
        super().__init__(modulation, fec_matrix)

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
            c = (2. * b_t) - 1.
        else:
            raise Exception(f"Unknown modulation {self.modulation}")

        return c


class NetworkTransmitter(ChannelComponent):
    """
        Receiver channel element using Neural Network for decoding
    """

    def __init__(self, block_length, block_coded_length, network_model=None, network_type=None):
        """
        :param block_length: Length of blocks before encoding (k)
        :param block_coded_length: Length of blocks after encoding (n)
        :param network_model: The neural network Keras model
        :param network_type: The neural network type
        """
        super().__init__(None, np.zeros((block_length, block_coded_length)))

        assert (network_model is not None), "You must specify a Network model"
        assert (network_type is not None), "You must specify a network type if you want to use it"

        self.network_model = network_model
        self.network_type = network_type

    def transmit(self, b):
        """
        :param y_n: Input symbols
        :return: Output bits
        """
        if self.network_type == NeuralNetworkType.ONE_HOT:
            raise Exception("One-hot network not supported for transmitter")
        elif self.network_type == NeuralNetworkType.INTEGER:
            c = self.network_model.predict(np.array(np.split(b, len(b) // self.block_length)))
            return c.flatten()
        else:
            raise Exception(f"Unsupported network type {self.network_type}")


# ----------------------------------------------------------------------------------------------------------------------
# Receivers
# ----------------------------------------------------------------------------------------------------------------------

class Receiver(ChannelComponent):
    """
        Basic receiver, using a FEC Matrix and a specified modulation
    """

    class ReceiverMode(Enum):
        CLASSIC = auto()
        MAP = auto()

    def __init__(self, modulation=None, fec_matrix=None):
        super().__init__(modulation, fec_matrix)
        self.mode = self.ReceiverMode.MAP

        # Pre-calculate coded elements for G
        transmitter = Transmitter(modulation, self.fec_matrix)

        block_elements = []
        block_coded_elements = []
        for elt in get_all_possible_words(self.block_length):
            block_elements.append(elt)
            block_coded_elements.append(transmitter.transmit(elt))

        self.block_elements = np.array(block_elements)
        self.block_coded_elements = np.array(block_coded_elements)

    def receive(self, y_n):
        y_n = np.array(y_n)

        if self.mode == self.ReceiverMode.CLASSIC:
            # Classic mode, using a simple Threshold
            # For non-coded bits, this method do the same thing as MAP, but faster

            # If we are using a FEC matrix, we cannot demap directly, we need to decode the error correction code first
            if self.has_fec_matrix:
                raise Exception("You cannot decode directly by using an error correction code, use MAP mode instead")

            # Otherwise, we can demap directly the symbols
            if self.modulation in ['BPSK']:
                return np.array(list(map(lambda x: 0 if x < 0 else 1, y_n)))
            else:
                raise Exception(f"Unknown modulation {self.modulation}")
        elif self.mode == self.ReceiverMode.MAP:
            # If we didn't passed a FEC matrix, we don't need to use that method, a threshold detector is enough
            if not self.has_fec_matrix:
                print("No FEC matrix specified, auto switch receiver mode to CLASSIC")
                self.mode = self.ReceiverMode.CLASSIC
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


class NetworkReceiver(ChannelComponent):
    """
        Receiver channel element using Neural Network for decoding
    """

    def __init__(self, block_length, block_coded_length, network_model=None, network_type=None):
        """
        :param block_length: Length of blocks before encoding (k)
        :param block_coded_length: Length of blocks after encoding (n)
        :param network_model: The neural network Keras model
        :param network_type: The neural network type
        """
        super().__init__(None, np.zeros((block_length, block_coded_length)))

        assert (network_model is not None), "You must specify a Network model"
        assert (network_type is not None), "You must specify a network type if you want to use it"

        self.network_model = network_model
        self.network_type = network_type

        self.block_elements = np.array(list(get_all_possible_words(self.block_length)))

    def receive(self, y_n):
        """
        :param y_n: Input symbols
        :return: Output bits
        """
        if self.network_type == NeuralNetworkType.ONE_HOT:
            b_r = self.network_model.predict(np.array(np.split(y_n, len(y_n) // self.block_coded_length)))
            b_r = np.array(list(map(lambda x: self.block_elements[np.argmax(x)], b_r)))
            return b_r.flatten()
        elif self.network_type == NeuralNetworkType.INTEGER:
            b_r = self.network_model.predict(np.array(np.split(y_n, len(y_n) // self.block_coded_length)))
            b_r = b_r.flatten()
            return np.round(b_r)
        else:
            raise Exception(f"Unsupported network type {self.network_type}")


# ----------------------------------------------------------------------------------------------------------------------
# Channels
# ----------------------------------------------------------------------------------------------------------------------

class AWGNChannel:
    """
        Additive white gaussian noise channel
    """

    def __init__(self, BPS, block_length, block_coded_length):
        """
        :param BPS: Number of bits per symbol
        :param block_length: Length of blocks before encoding (k)
        :param block_coded_length: Length of blocks after encoding (n)
        """
        self.n = block_coded_length
        self.k = block_length
        self.BPS = BPS

    def process(self, c, EbN0dB):
        Pn = calculate_noise_power(EbN0dB, np.var(c), self.k, self.n, self.BPS)
        return np.array((np.sqrt(Pn / 2.) * np.random.randn(len(c))) + np.array(c))
