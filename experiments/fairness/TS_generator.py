import random
import numpy as np
import zlib
import math
import pickle

from functools import reduce
from operator import mul
from fractions import Fraction
from math import sqrt
import pandas

import time

import sys

#sys.stdout = open('experiments/fairness/TS-GEN.out', 'w')


# --- FUNCTIONS ---

def decode_columns(arr, bounds):
    d_arr = []
    for i in range(len(arr)):
        _min, _max = bounds[i]
        value = arr[i]
        if (_max - _min) == 1:
            d_arr.append(round(value))
        else:
            d_arr.append(round(value*(_max-_min)+_min))
    return d_arr


def maximize_y(Y):
    
    
    y_len_max = 0
    y_len_min = math.inf
    y_max = []
    y_min = []

    for i in range(len(Y)):
        y = np.delete(Y, i, axis=0)
        y_C = ""
        for sub_el in y:
            y_C += str(zlib.compress(str(sub_el).encode()))
        if len(y_C)>y_len_max:
            y_len_max = len(y_C)
            y_max_el = Y[i]
            y_max = str(zlib.compress(str(y_max_el).encode()))
            y_max_index = i
    Y_C = ""           
    for sub_el_2 in Y:
        Y_C += str(zlib.compress(str(sub_el_2).encode()))
        sub_el_2_C = str(zlib.compress(str(sub_el_2).encode()))
        if len(sub_el_2_C)<y_len_min:
                y_len_min = len(sub_el_2_C)
                y_min = sub_el_2_C

    Y_k = np.delete(Y, y_max_index, axis=0)

    Y_k_C = ""
    for sub_el_3 in Y_k:
        Y_k_C += str(zlib.compress(str(sub_el_3).encode()))

    #print(len(Y_C),len(y_min),len(Y_k_C))
    NCD.append( ((len(Y_C) - len(y_min))/len(y_max) , Y_k) )


    if len(Y_k) > 2:
        maximize_y(Y_k)

    return NCD


def nCk(n,k):
    return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )

def uniformComparison(oriDist,eps2=0.1):
    expected_size = None
    isuniform = None

    if not(oriDist):
        print("Empty")
        return
    dist = pandas.DataFrame(oriDist)
    domain=np.unique(dist,axis=0)
    #The numer of samples come from lemma 5 of collision-based testers are optimal for uniformity and closeness
    expected= 6*sqrt(len(domain))/eps2
    if(len(oriDist) < expected):
        expected_size = str(expected)
    s = 0
    for i in range(len(oriDist)):
        s = s+ oriDist[(i+1):len(oriDist)].count(oriDist[i])
    t= nCk(len(oriDist),2)*(1+3/4*eps2)/len(domain)
    if(s>t):
        isuniform = False
    else:
        isuniform = True

    return (isuniform, expected_size)

# --------------------------------


NCD = []

TS_size = 50
num_genes = 5
bounds = [(18,50),(0,1),(0,1),(0,3),(10,50)]



def generate_TC(CNT):
    #t = time.time()

    TSD = []
    CNT += 1

    candidates_lst = []
    candidates = np.random.uniform(0,1,(50,num_genes))

    for i in range (len(candidates)):
        candidates_lst.append(decode_columns(candidates[i],bounds))
        candidates_lst[i]=np.array(candidates_lst[i])

    candidateTCs = candidates_lst
    for i in range(len(candidateTCs)):
        candidateTCs[i]=list(candidateTCs[i])


    for c in candidateTCs:
        NCD = []
        TS_temp = [i for i in TS_init]
        TS_temp.append(c)
        
        uniformity, size = uniformComparison(TS_temp,eps2=0.1)
      
        NCD = maximize_y(np.array(TS_temp))
        if uniformity:
            TSD.append((max(NCD,key=lambda item:item[0]),c, size))
        

    best_TSD, best_c, size = max(TSD,key=lambda item:item[0][0])


    if best_TSD[0] > TSD_init[0]:
        TS_init.append(best_c)

    size = round(float(size))

    # elapsed = time.time()-t
    # print(elapsed)

    if (len(TS_init) < size):
        print(len(TS_init),size)
        print("less")
        print(TS_init)
        if CNT < 200:
            generate_TC(CNT)
    else:
        print(size)
        print("more")
        print(TS_init)
    


# --- INITIAL TS RND GENERATION ---

TS1 = []
TS = np.random.uniform(0,1,(5,num_genes))

for i in range (len(TS)):
    TS1.append(decode_columns(TS[i],bounds))
    TS1[i]=np.array(TS1[i])

TS_init = TS1
for i in range(len(TS_init)):
    TS_init[i]=list(TS_init[i])


NCD = []
NCD = maximize_y(np.array(TS_init))

TSD_init = max(NCD,key=lambda item:item[0])

# ----------------------------------

NCD = []
generate_TC(0)


#sys.stdout.close()


