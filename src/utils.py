# -*- coding: utf-8 -*-

import numpy as np

def ber_performance(modulation, EbN0dBs, channel_function, L=1000, target_nb_errors=50):
    BER = np.zeros(len(EbN0dBs))    
    
    for (i, EbN0dB) in enumerate(EbN0dBs):
        nb_errors = 0
        nb_elements = 0
        
        print("Start EbN0 %d/%d"%(i,len(EbN0dBs)), EbN0dB)
        
        while (nb_errors < target_nb_errors):
            b = np.random.randint(0, 2, L)
            b_r, _, _ = channel_function(modulation, b, EbN0dB)
            
            nb_errors += np.sum(b != b_r)
            
            #print('Nb elemetns | ', nb_elements, ' | Nb errors | ', EbN0dB, ' | ', nb_errors)/
            nb_elements += L
            
        BER[i] = float(nb_errors) / float(nb_elements)
        
    return BER