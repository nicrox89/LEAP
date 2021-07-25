import numpy as np
from pyitlib import discrete_random_variable as drv

features = ["age","gender","marital_status","education","lift_heavy_weight"]
bounds = [(18,50),(0,1),(0,1),(0,3),(10,50)]
test_arr = np.array([[20,1,0,3,30],[30,0,0,2,20],[45,1,1,1,25],[50,1,1,3,15]])

def encode_columns(arr, bounds):
    encoded_arr = []
    for i in range(len(arr[0,])):
        a,b=bounds[i]
        possible_values = b-a
        if(possible_values>1): # it is not binary, endcode!
            for j in range(a,b+1):
                encoded_arr.append(np.array([1 if x==j else 0 for x in arr[:,i]]))
        else:
            encoded_arr.append(arr[:,i])
        
    return np.array(encoded_arr).T


tst = encode_columns(test_arr, bounds)
arr_e1 = [list(test_arr[:,i]) for i in range(len(test_arr[0]))]
e1 = drv.entropy_joint(arr_e1)

arr_e2 = [list(tst[:,i]) for i in range(len(tst[0]))]
e2 = drv.entropy_joint(arr_e2)
print(tst)