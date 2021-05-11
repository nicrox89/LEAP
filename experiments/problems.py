#!/usr/bin/env python3
import numpy as np
from pyitlib import discrete_random_variable as drv
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression

class fitness():

    def __init__(self, c):
        self.classifier = c
        self.stat = []

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
            M.append(drv.information_mutual(ch[:,i],np.array(y),cartesian_product=True))
            #M.append(mutual_info_score(ch[:,i],np.array(y),contingency=None))
            #M.append(normalized_mutual_info_score(ch[:,i],np.array(y),average_method='arithmetic'))
            #M.append(mutual_info_regression(ch[:,i],np.array(y),discrete_features='auto'))

        den = 0
        num = 0
        res = 0
        threshold = sum(M) * 0.7
        M.sort(reverse=True)

        #den = 0
        while num < threshold:
            num=num+M[den]
            den=den+1
        if den!=0:
            res = num/den
        self.stat.append([res,den])
            #print("MI:",res)
            #print("num features:",den)
        return res

    def getStat(self):
        return self.stat

    def setStat(self):
        self.stat = []