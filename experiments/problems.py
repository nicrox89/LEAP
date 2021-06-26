#!/usr/bin/env python3
import numpy as np
import math
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
        #len(chromosome) = 30 (observations)
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
        
        #------v2.0
        #dictionary Variables-MI
        Var = ["age","education.num","marital.status","race","sex","capital.gain","capital.loss","hours.per.week"]

        d = {"".join(Var[0]):M[0]}
        #print(d)

        for i in range(len(Var)):
            d["".join(Var[i])] = M[i]
        #print(d)

        sort_orders = sorted(d.items(), key=lambda x: x[1], reverse=True)
        #print(sort_orders)

        best = []
        #------
        den = 0
        num = 0
        res = 0
        summary = 0
        summary = sum(M)
        threshold = summary * 0.5
        
        M.sort(reverse=True)
        #print(M)

        #den = 0
        while num < threshold:
            num=num+M[den]
            den=den+1
        if den!=0:
            #res = num/den
            #den_n = (den-1) / (8 - 1)
            #res = num-den_n
            res = num*math.exp(-den)
        #------v2.0
        for i in range(den):
            best.append(sort_orders[i])
        self.stat.append([res,den,best])
        #------
        #self.stat.append([res,den])
            #print("MI:",res)
            #print("num features:",den)
        return res

    def getStat(self):
        return self.stat

    def setStat(self):
        self.stat = []