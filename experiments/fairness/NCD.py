import numpy as np
import zlib
import math
import pickle

NCD = []

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

    print(len(Y_C),len(y_min),len(Y_k_C))
    #NCD.append( ((len(Y_C) - len(y_min))/len(Y_k_C) , Y_k) )
    NCD.append( ((len(Y_C) - len(y_min))/len(y_max) , Y_k) )

    if len(Y_k) > 2:
        maximize_y(Y_k)


#Ts = pickle.load(open("./experiments/fairness/variableStoringFile.dat", "rb"))
#Ts = [[1,2,3],[1,2,33],[1,2,3],[1,2,3],[1,2,3]]
#Ts = [[1,2,3],[1,2,33],[1,222,3],[1,2,3],[1,2,3]]
#Ts = [[1,2,3],[1,2,33],[1,222,3],[17,2,3],[1,2,3]]
#Ts = [[1,2,3],[1,2,33],[1,222,3],[17,2,3],[1,2,343]]
#Ts = [[1,2,3],[1,2,33],[1,222,3],[44,2,0]]
#Ts = [[1,2,3],[1,2,33],[1,222,3],[1,2,3]]
#Ts = [[1,2,3],[1,2,33],[1,2,3],[1,2,3]]
#[1,2,3],[1,2,33],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3],[1,2,3]]

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


TS_size = 50
num_genes = 5
bounds = [(18,50),(0,1),(0,1),(0,3),(10,50)]

candidates_lst = []
candidates = np.random.uniform(0,1,(TS_size,num_genes))

for i in range (len(candidates)):
    candidates_lst.append(decode_columns(candidates[i],bounds))
    candidates_lst[i]=np.array(candidates_lst[i])

candidateTCs = candidates_lst
for i in range(len(candidateTCs)):
    candidateTCs[i]=list(candidateTCs[i])

#maximize_y(np.array(Ts[:10]))
#maximize_y(np.array(Ts))
maximize_y(np.array(candidateTCs))

max_Ts = max(NCD,key=lambda item:item[0])


from scipy.stats import entropy
import pandas as pd

data = [1,1,2,2]
data = [[1,1],[1,1]]

pd_series = pd.Series(data)
counts = pd_series.value_counts()
entropy = entropy(counts)

    