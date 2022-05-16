# --- TEST SUITE GENERATION random()) ---

import numpy as np
import pickle

#Save the variable
 


TS_size = 100
TestSuite0 = []
bounds = [(18,50),(0,1),(0,1),(0,3),(10,50)]



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


TS = [[(np.random.random()) for i in range(5)]for j in range(TS_size)]

for i in range (len(TS)):
    TestSuite0.append(decode_columns(TS[i],bounds))
    TestSuite0[i]=np.array(TestSuite0[i])

TestSuite_Zero = TestSuite0
for i in range(len(TestSuite_Zero)):
    TestSuite_Zero[i]=list(TestSuite_Zero[i])

s=0
T = []
for i in range(len(TestSuite_Zero)):
    s = s+ TestSuite_Zero[(i+1):len(TestSuite_Zero)].count(TestSuite_Zero[i])
    s_i = TestSuite_Zero[(i+1):len(TestSuite_Zero)].count(TestSuite_Zero[i])
    if s_i == 0:
        T.append(i)
    else:
        T.append(i)
        T.append(i)

domain = np.unique(T,axis=0)

print()
print("TS SIZE: ")
print(TS_size)
print("RANDOM")
print("Duplicates: ")
print(s)

pickle.dump(TestSuite_Zero, open("./experiments/fairness/variableStoringFile.dat", "wb"), protocol=2)
