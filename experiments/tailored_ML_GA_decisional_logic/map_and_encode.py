import numpy as np
import math
import copy
from pyitlib import discrete_random_variable as drv


def encode_columns(arr, bounds):
    encoded_arr = []
    for i in range(len(arr[0,])):
        a,b=bounds[i]
        possible_values = b-a
        if(possible_values>1): # if it is not binary, encode!
            for j in range(a,b+1):
                encoded_arr.append(np.array([1 if x==j else 0 for x in arr[:,i]]))
        else:
            encoded_arr.append(arr[:,i])
        
    return np.array(encoded_arr).T


def encode_columns_splits(arr, bounds, num_splits):
    encoded_arr = []
    splits = []
    count = 0

    for i in range(len(arr[0,])): # for each variable
        a,b=bounds[i]
        possible_values = b-a
        
        spl = []

        if(possible_values>1): # if it is not binary, encode!

            possible_values = possible_values +1 if a == 0 else possible_values
            borders = math.floor(possible_values / num_splits[i])

            for k in range(num_splits[i]):
                start = (k*borders)+a
                
                if k == num_splits[i] -1:
                    end = b
                else:
                    end = start + borders -1
                    
                spl.append((start,end))
                splits.append({"group": i, "index": count, "borders": (start,end), "taken": False})
                count = count + 1
            #spl.append((0,a-1))
            #spl = [(((k+1)*borders)+a,((k+1)*borders)+a+borders) for k in range(num_splits[i]-1)]
            #spl.append((((borders*num_splits[i])+1),b))
            
            for j in range(len(spl)):
                encoded_arr.append(np.array([1 if x in range(spl[j][0], spl[j][1]) else 0 for x in arr[:,i]]))
        else:
            spl.append((0,1))
            splits.append({"group": i, "index": count, "borders": (0,1), "taken": False})
            count = count + 1
            encoded_arr.append(arr[:,i])
        
        #splits.append(spl)
        
    return (np.array(encoded_arr).T, splits)



def cause(parents, decisional):
    partitions = parents[0].genome[-1]
    test = parents[0].genome[:-1]
    pred = [decisional([list(x)]) for x in test]

    

    test_inv = copy.deepcopy(test)
    for i in range(len(test_inv)):
        if test_inv[i][1] == 0:
            test_inv[i][1] = 1
        elif test_inv[i][1] == 1:
            test_inv[i][1] = 0

    for i in range(len(test_inv)):
        if test_inv[i][2] == 0:
            test_inv[i][2] = 1
        elif test_inv[i][2] == 1:
            test_inv[i][2] = 0

    # for i in range(len(test_inv)):
    #    if test_inv[i][4] > 30:
    #        test_inv[i][4] = 20
    #    elif test_inv[i][4] <= 30:
    #        test_inv[i][4] = 40

    for i in range(len(test_inv)):
        if test_inv[i][3] > 1:
            test_inv[i][3] = 0
        elif test_inv[i][3] <= 1:
            test_inv[i][3] = 2

    #test_inv = [1 if x[1] == 0 else (0 if x[1] == 1 else None) for x in test]
    pred_inv = [decisional([list(x)]) for x in test_inv]

    score =sum([1 if pred[i] != pred_inv[i] else 0  for i in range(len(pred))])/len(pred)

    return score




def test():
    features = ["age","gender","marital_status","education","lift_heavy_weight"]
    bounds = [(18,50),(0,1),(0,1),(0,3),(10,50)]
    test_arr = np.array([[20,1,0,3,30],[30,0,0,2,20],[45,1,1,1,25],[50,1,1,3,15]])
    splits = [3,0,0,2,3]

    tst_0 = encode_columns_splits(test_arr, bounds, splits)


    tst = encode_columns(test_arr, bounds)
    arr_e1 = [list(test_arr[:,i]) for i in range(len(test_arr[0]))]
    e1 = drv.entropy_joint(arr_e1)

    arr_e2 = [list(tst[:,i]) for i in range(len(tst[0]))]
    e2 = drv.entropy_joint(arr_e2)
    print(tst)
