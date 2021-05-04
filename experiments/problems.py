#!/usr/bin/env python3
import numpy as np
from pyitlib import discrete_random_variable as drv

class fitness():

    def __init__(self, c):
        self.classifier = c

    def f(self, chromosome):
        num_genes = len(chromosome[0]) # columns
        y=[]
        #predict
        for i in range(len(chromosome)):
            y.append(self.classifier.predict([chromosome[i]])[0])
        
        #compute mutual array using y
        M=[]
        ch = np.array(chromosome)
        
        for i in range(num_genes):
            # int type problem!!
            M.append(drv.information_mutual(np.array(y), ch[:,i], cartesian_product=True))

        den = 0
        num = 0
        threshold = sum(M) * 0.3
        M.sort(reverse=True)

        den = 0
        while num < threshold:
            num=num+M[den]
            den=den+1

        return num/den