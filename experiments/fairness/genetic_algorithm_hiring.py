import sys
import random
from pyitlib import discrete_random_variable as drv
import numpy as np

from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from leap_ec.int_rep.ops import mutate_randint, individual_mutate_randint

from scipy import spatial

from problemMultiCMI import fitness
from individual import Individual
from leap_ec import util
from leap_ec.context import context
from leap_ec import ops
from toolz import pipe
from leap_ec import probe
from typing import Iterator, List
from copy import copy
from copy import deepcopy
from sklearn.metrics.pairwise import manhattan_distances, euclidean_distances
from numpy.linalg import norm
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
#from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from functools import reduce
from operator import mul
from fractions import Fraction
from math import sqrt
import pandas

import zlib
from numpy import linalg as LA
import copy

from textdistance import entropy_ncd

def decide(applicant):
    gender = 1
    if applicant[gender] == 1:
        return 0
    else:
        return 1


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

def decode_columns_Y(arr, bounds):
    d_arr = []
    for i in range(len(arr)):
        _min, _max = bounds[i]
        value = arr[i]
        if (_max - _min) == 1:
            d_arr.append(round(value))
        else:
            d_arr.append(round(value*(_max-_min)+_min))
        
    d_arr.append(decide(np.array(d_arr)))
    return d_arr


#female=1 male=0
def decide1(applicant):
    gender = 1
    if applicant[0][gender] == 1:
        return 0
    else:
        return 1

def decide2(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if  applicant[0][heavy_weight] >= 30:
        if applicant[0][gender] == 1:
            r = random.randrange(0, 1)
            if r > 0:
                return 1
            else:
                return 0
        else:
            return 1
    else:
        return 0

def decide3(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if  applicant[0][heavy_weight] >= 41:
        if applicant[0][gender] == 1:
            return 1
        else:
            return 1
    elif applicant[0][heavy_weight] < 41:
        if applicant[0][age] <= 31:
            return 1
        else:
            return 0

def decide4(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if  applicant[0][heavy_weight] >= 21:
        if applicant[0][gender] == 1:
            return 1
        else:
            return 0
    elif applicant[0][heavy_weight] < 21:
        if applicant[0][gender] == 1:
            return 1
        else:
            return 0


#--- GENETIC OPERATORS ---

def tournament_selection(population, k=2):
#randomly choose k individuals (2 at a time) and select the best one
    choices = random.choices(population, k=k)
    if choices[0].fitness > choices[1].fitness:
        best = choices[0]
    else:
        best = choices[1]
    
    return best

# crossover two parents to create two children
def crossover(ind1, ind2, r_cross):
	# children are copies of parents by default
    if len(ind1.genome) != len(ind2.genome):
        # TODO what about variable length genomes?
        raise RuntimeError(
            'genomes must be same length for uniform crossover')

    ind_A = np.array(ind1.genome) #
    ind_B = np.array(ind2.genome) #
    ind_TMP = copy(ind_A) #

    for i in range(len(ind1.genome)):
        if random.random() < r_cross:
            ind_TMP[i] = ind_B[i] #
            ind_B[i] = ind_A[i] #
            ind_A[i] = ind_TMP[i] #
            #ind1.genome[:,i], ind2.genome[:,i] = ind2.genome[:,i], ind1.genome[:,i]

    ind1.genome = list(ind_A) #
    ind2.genome = list(ind_B) #
    return [ind1, ind2]


def mutation(individual,m_rate):
    #print("before")
    #print(individual.genome)
    #index_1 = random.randrange(len(individual.genome))

    for i in range(len(individual.genome)):
        if random.random() < m_rate:
            individual.genome[i] = 1 if individual.genome[i] == 0 else 0

    #print("after")
    #print(individual.genome)
    return individual

#---


# ----------------------------------------------------------------
# ----------------------------- main -----------------------------
# ----------------------------------------------------------------

var = ["age","gender","marital_status","education","lift_heavy_weight"]

#FITNESS FUNCTION

p = fitness()

model = decide1

result = []
y = []
TestSuite = []
TestSuite0 = []
TestSuite1 = []

#number of Test Cases (1 matrix)
TS_size = 10000 #100000

#number of individuals/instances = partitions
pop_size = 10

#number of features
#num_genes = len(var)
num_genes = 2
#number generations
num_generations = 30

features = ["age","gender","marital_status","education","lift_heavy_weight"]
#bounds = [(18,50),(0,1),(0,1),(0,3),(10,50)]
bounds = [(8,17),(8,17)]
binary_bounds = [(0,1),(0,1),(0,1),(0,1),(0,1)]


#create Test Suite
# features_inst = [[]for j in range(num_genes)]
# TestSuite = []

# for k in range(num_genes):
#     features_inst[k] = [random.randint(bounds[k][0],bounds[k][1]) for i in range(TS_size)]

# temp = np.array(features_inst)

# for i in range(TS_size):
#     TestSuite.append(temp[:,i])

# print(TestSuite)

#ART


import math
import numpy as np

def min_column(x):
    return [min(x[:,i]) for i in range(len(x[0]))]
    
def max_column(x):
    return [max(x[:,i]) for i in range(len(x[0]))]

def normalise(x, min_x ,max_x):
    return [[((x[i,j] - min_x[j]) / (max_x[j] - min_x[j])) for j in range(len(x[0]))] for i in range(len(x))]

def distanceCos(s1, s2):
    ab = sum(np.multiply(s1, s2))
    a_pwr = math.sqrt(sum(np.power(s1, 2)))
    b_pwr = math.sqrt(sum(np.power(s2, 2)))
    return ab / (a_pwr * b_pwr)

def distance2(s1, s2):
    d = distance.euclidean(s1, s2)
    return d


currentTS = []
checkTestSuite = []
num_candidates = 10

#checkList = np.random.uniform(0,1,(TS_size,num_genes))

# generate check list
def generate_cl(size,genes):
    check_ts = []
    random.seed(10)
    cl = np.random.uniform(0,1,(size,genes))
    for i in range (len(cl)):
        check_ts.append(decode_columns(cl[i],bounds))
        check_ts[i]=np.array(check_ts[i])
    return check_ts

def generate_cl_Y(size,genes):
    check_ts = []
    cl = np.random.uniform(0,1,(size,genes))
    for i in range (len(cl)):
        check_ts.append(decode_columns_Y(cl[i],bounds))
        check_ts[i]=np.array(check_ts[i])
    return check_ts

def compute_distance(ts: List, TS_size):
    import zlib
    num_genes = 2
    candidates = generate_cl(num_candidates, num_genes)
    # mds = MDS(random_state=0)
    # candidates2 = mds.fit_transform(candidates)
    #plt.scatter(np.array(candidates2)[:,0], np.array(candidates2)[:,1])
    #plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
    #plt.show()
    distances = []
    elements = []
    for c in candidates:
        d_min = math.inf
        elem = None
        for t in ts:
            # EUCLIDEAN
            d = distance2(c, t)
            # NCD
            # c = str(c).encode('utf-8')
            # t = str(t).encode('utf-8')
            # conc = c + t
            # d = (len(zlib.compress(conc)) - min(len(zlib.compress(c)), len(zlib.compress(t)))) / max(len(zlib.compress(c)), len(zlib.compress(t)))
            if d < d_min:
                d_min = d
                elem = c
        distances.append(d_min)
        elements.append(elem)

    best = max(distances)
    ts.append(elements[distances.index(best)])
    #mds = MDS(random_state=0)
    #ts2 = mds.fit_transform(ts)
    #plt.scatter(np.array(ts2)[:,0], np.array(ts2)[:,1], color='red')
    #plt.show()
    #if len(ts) < TS_size:
    #compute_distance(currentTS, TS_size)

def compute_distance_NCD(ts: List, TS_size):
    import zlib
    num_genes = 5
    candidates = generate_cl(num_candidates, num_genes)
    # mds = MDS(random_state=0)
    # candidates2 = mds.fit_transform(candidates)
    
    #plt.scatter(np.array(candidates2)[:,0], np.array(candidates2)[:,1])
    #plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
    #plt.show()
    distances = []
    elements = []
    for c in candidates:
        d_min = math.inf
        elem = None
        for t in ts:
            # EUCLIDEAN
            d = NCD(c, t)
            # NCD
            # c = str(c).encode('utf-8')
            # t = str(t).encode('utf-8')
            # conc = c + t
            # d = (len(zlib.compress(conc)) - min(len(zlib.compress(c)), len(zlib.compress(t)))) / max(len(zlib.compress(c)), len(zlib.compress(t)))
            if d < d_min:
                d_min = d
                elem = c
        distances.append(d_min)
        elements.append(elem)

    best = max(distances)
    ts.append(elements[distances.index(best)])
    #mds = MDS(random_state=0)
    #ts2 = mds.fit_transform(ts)
    #plt.scatter(np.array(ts2)[:,0], np.array(ts2)[:,1], color='red')
    #plt.show()
    #if len(ts) < TS_size:
    #    compute_distance(currentTS, TS_size)


def compute_distance_EntropyNCD(ts: List, TS_size):
    import zlib
    num_genes = 5
    candidates = generate_cl(num_candidates, num_genes)
    # mds = MDS(random_state=0)
    # candidates2 = mds.fit_transform(candidates)
    
    #plt.scatter(np.array(candidates2)[:,0], np.array(candidates2)[:,1])
    #plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
    #plt.show()
    distances = []
    elements = []
    for c in candidates:
        d_min = math.inf
        elem = None
        for t in ts:
            # EUCLIDEAN
            d = EntropyNCD(c, t)
            # NCD
            # c = str(c).encode('utf-8')
            # t = str(t).encode('utf-8')
            # conc = c + t
            # d = (len(zlib.compress(conc)) - min(len(zlib.compress(c)), len(zlib.compress(t)))) / max(len(zlib.compress(c)), len(zlib.compress(t)))
            if d < d_min:
                d_min = d
                elem = c
        distances.append(d_min)
        elements.append(elem)

    best = max(distances)
    ts.append(elements[distances.index(best)])
    #mds = MDS(random_state=0)
    #ts2 = mds.fit_transform(ts)
    #plt.scatter(np.array(ts2)[:,0], np.array(ts2)[:,1], color='red')
    #plt.show()
    #if len(ts) < TS_size:
    #    compute_distance(currentTS, TS_size)


def compute_distance_Y(ts: List, TS_size):
    import zlib
    num_genes = 5
    candidates = generate_cl_Y(num_candidates, num_genes)
    mds = MDS(random_state=0)
    candidates2 = mds.fit_transform(candidates)
    
    #plt.scatter(np.array(candidates2)[:,0], np.array(candidates2)[:,1])
    #plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
    #plt.show()
    distances = []
    elements = []
    for c in candidates:
        d_min = math.inf
        elem = None
        for t in ts:
            # EUCLIDEAN
            d = distance2(c, t)
            # NCD
            # c = str(c).encode('utf-8')
            # t = str(t).encode('utf-8')
            # conc = c + t
            # d = (len(zlib.compress(conc)) - min(len(zlib.compress(c)), len(zlib.compress(t)))) / max(len(zlib.compress(c)), len(zlib.compress(t)))
            if d < d_min:
                d_min = d
                elem = c
        distances.append(d_min)
        elements.append(elem)

    best = max(distances)
    ts.append(elements[distances.index(best)])
    #mds = MDS(random_state=0)
    #ts2 = mds.fit_transform(ts)
    #plt.scatter(np.array(ts2)[:,0], np.array(ts2)[:,1], color='red')
    #plt.show()
    #if len(ts) < TS_size:
    #    compute_distance(currentTS, TS_size)

def NCD(c,t):
    c = str(c).encode('utf-8')
    t = str(t).encode('utf-8')
    conc = c + t
    d = (len(zlib.compress(conc)) - min(len(zlib.compress(c)), len(zlib.compress(t)))) / max(len(zlib.compress(c)), len(zlib.compress(t)))
    return d

def EntropyNCD(c,t):
    c = str(c)
    t = str(t)
    d = entropy_ncd(c,t)
    return d


def plotting3D():
    from mpl_toolkits import mplot3d

    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    plt.show()

    ax.plot3D(np.array(currentTS)[:,0], np.array(currentTS)[:,3], np.array(currentTS)[:,4],'o')

def plotting2D():
    from mpl_toolkits import mplot3d

    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    plt.show()

    ax.plot3D(np.array(currentTS)[:,0], np.array(currentTS)[:,1], np.array(currentTS)[:,2],'o')


#--- L2-NORM COLLISION UNIFORMITY TEST ---

def nCk(n,k):
    return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )

def uniformComparison(oriDist,eps2=0.1):
    if not(oriDist):
        print("Empty")
        return
    dist = pandas.DataFrame(oriDist)
    domain = np.unique(dist,axis=0)
    #The numer of samples come from lemma 5 of collision-based testers are optimal for uniformity and closeness
    expected = 6*sqrt(len(domain))/eps2
    if(len(oriDist) < expected):
        print("You need " + str(expected) + " samples")
    else:
        print("The domain lenght is: "+str(len(domain)))
    s = 0
    for i in range(len(oriDist)):
        s = s+ oriDist[(i+1):len(oriDist)].count(oriDist[i])
    t= round(nCk(len(oriDist),2)*(1+3/4*eps2)/len(domain))
    if(s>t):
        print("False")
    else:
        print("True")

#uniformComparison([[1,1,1],[1,1,1]],eps2=0.1)

#---


def dist(ts: List, TS_size):
    import zlib
    num_genes = 5
    candidates = generate_cl(num_candidates, num_genes)

    distances = []
    elements = []
    for c in candidates:
        d_max = 0
        elem = None
        for t in ts:
            d = NCD(c, t)
            if d > d_max:
                d_max = d
                elem = c
        distances.append(d_max)
        elements.append(elem)

    best = max(distances)
    ts.append(elements[distances.index(best)])




# --- TEST SUITE GENERATION random()) ---

import seaborn as sns; sns.set()

TS_size = 50

TS = [[(np.random.random()) for i in range(2)]for j in range(TS_size)]

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
print("UNIFORM: ")
print(uniformComparison(TestSuite_Zero,eps2=0.1))

# --- plot
mds = MDS(random_state=0)
currentTS2 = mds.fit_transform(TestSuite_Zero)
plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
plt.title("RND")
plt.show()
# ---

# fig, ax = plt.subplots()
# sns.distplot(T, bins=len(T), color="g", ax=ax)
# plt.xticks(domain)
# plt.show()

#import pickle

#Save the variable
#pickle.dump(TestSuite_Zero, open("./experiments/fairness/variableStoringFile.dat", "wb"))
#Use saved variable
#T = pickle.load(open("./experiments/fairness/variableStoringFile.dat", "rb"))


# --- 



# --- TEST SUITE GENERATION random.uniform() ---

TS_size = 100

TS = np.random.uniform(0,1,(TS_size,num_genes))

for i in range (len(TS)):
    TestSuite1.append(decode_columns(TS[i],bounds))
    TestSuite1[i]=np.array(TestSuite1[i])

TestSuite_One = TestSuite1
for i in range(len(TestSuite_One)):
    TestSuite_One[i]=list(TestSuite_One[i])

mds = MDS(random_state=0)
currentTS2 = mds.fit_transform(TestSuite_One)

plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
plt.title("RND.UNIFORM()")
plt.show()

s=0
T = []
for i in range(len(TestSuite_One)):
    s = s+ TestSuite_One[(i+1):len(TestSuite_One)].count(TestSuite_One[i])
    s_i = TestSuite_One[(i+1):len(TestSuite_One)].count(TestSuite_One[i])
    if s_i == 0:
        T.append(i)
    else:
        T.append(i)
        T.append(i)

domain = np.unique(T,axis=0)

print()
print("TS SIZE: ")
print(TS_size)
print("RANDOM UNIFORM")
print("Duplicates: ")
print(s)
print("UNIFORM: ")
print(uniformComparison(TestSuite_One,eps2=0.1))


# fig, ax = plt.subplots()
# sns.distplot(t[:,5], bins=2, color="g", ax=ax)
#plt.xticks(domain)
# plt.show()
# fig, ax = plt.subplots()
# sns.distplot(T, bins=len(T), color="g", ax=ax)
# plt.xticks(domain)
# plt.show()

# --- 


# --- TEST SUITE GENERATION random.uniform() + OUTPUT ---

TS_size = 100
TestSuite_One = []
TestSuite1 = []

TS = np.random.uniform(0,1,(TS_size,num_genes))

for i in range (len(TS)):
    TestSuite1.append(decode_columns_Y(TS[i],bounds))
    TestSuite1[i]=np.array(TestSuite1[i])

TestSuite_One = TestSuite1
for i in range(len(TestSuite_One)):
    TestSuite_One[i]=list(TestSuite_One[i])

# mds = MDS(random_state=0)
# currentTS2 = mds.fit_transform(TestSuite_One)

# plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
# plt.show()

s=0
T = []
for i in range(len(TestSuite_One)):
    s = s+ TestSuite_One[(i+1):len(TestSuite_One)].count(TestSuite_One[i])
    s_i = TestSuite_One[(i+1):len(TestSuite_One)].count(TestSuite_One[i])
    if s_i == 0:
        T.append(i)
    else:
        T.append(i)
        T.append(i)

domain = np.unique(T,axis=0)

print()
print("TS SIZE: ")
print(TS_size)
print("RANDOM UNIFORM")
print("Duplicates: ")
print(s)
print("UNIFORM: ")
print(uniformComparison(TestSuite_One,eps2=0.1))

t = np.array(TestSuite_One)

zero=0
for i in range(len(t[:,2])):
    if t[:,2][i]==0:
        zero=zero+1

print("zero")
print(zero)
# fig, ax = plt.subplots()
# sns.distplot(T, bins=len(T), color="g", ax=ax)
# plt.xticks(domain)
# plt.show()
# fig, ax = plt.subplots()
# sns.distplot(TestSuite_One[5], bins=2, color="g", ax=ax)
# plt.xticks(domain)
# plt.show()
# --- 



# --- TEST SUITE GENERATION ADAPTIVE RANDOM TESTING + EUCLIDEAN DISTANCE - CLASSIC ---

# initialise test suite
currentTS = generate_cl(1, num_genes)

# mds = MDS(random_state=0)
# currentTS2 = mds.fit_transform(currentTS)
#plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
#plt.show()

TS_size = 500

for i in range(TS_size):
    compute_distance(currentTS, TS_size)
    print(i)

TestSuite_Two = currentTS
for i in range(len(TestSuite_Two)):
    TestSuite_Two[i]=list(TestSuite_Two[i])

s=0
T = []
for i in range(len(TestSuite_Two)):
    s = s+ TestSuite_Two[(i+1):len(TestSuite_Two)].count(TestSuite_Two[i])
    s_i = TestSuite_Two[(i+1):len(TestSuite_Two)].count(TestSuite_Two[i])
    if s_i == 0:
        T.append(i)
    else:
        T.append(i)
        T.append(i)

domain = np.unique(T,axis=0)

print()
print("TS SIZE: ")
print(TS_size)
print("ART EUCLIDEAN DISTANCE")
print("Duplicates: ")
print(s)
print("UNIFORM: ")
print(uniformComparison(TestSuite_Two,eps2=0.1))

#plotting3D()

mds = MDS(random_state=0)
currentTS2 = mds.fit_transform(TestSuite_Two)

plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
plt.title("ART-EUCLIDEAN")
plt.show()

# ---
#plt.text(points[8][0],points[8][1],str(round(distances[8], 2)))


# --- TEST SUITE GENERATION ADAPTIVE RANDOM TESTING + NCD DISTANCE - CLASSIC ---

# initialise test suite
currentTS = generate_cl(1, num_genes)

# mds = MDS(random_state=0)
# currentTS2 = mds.fit_transform(currentTS)
#plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
#plt.show()

TS_size = 50

for i in range(TS_size):
    compute_distance_NCD(currentTS, TS_size)
    #compute_distance_EntropyNCD(currentTS, TS_size)
    print(i)

TestSuite_NCD = currentTS
for i in range(len(TestSuite_NCD)):
    TestSuite_NCD[i]=list(TestSuite_NCD[i])

s=0
T = []
for i in range(len(TestSuite_NCD)):
    s = s+ TestSuite_NCD[(i+1):len(TestSuite_NCD)].count(TestSuite_NCD[i])
    s_i = TestSuite_NCD[(i+1):len(TestSuite_NCD)].count(TestSuite_NCD[i])
    if s_i == 0:
        T.append(i)
    else:
        T.append(i)
        T.append(i)

domain = np.unique(T,axis=0)

print()
print("TS SIZE: ")
print(TS_size)
print("ART NCD DISTANCE")
print("Duplicates: ")
print(s)
print("UNIFORM: ")
print(uniformComparison(TestSuite_NCD,eps2=0.1))

#plotting3D()

mds = MDS(random_state=0)
currentTS2 = mds.fit_transform(TestSuite_NCD)

plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
plt.title("ART-NCD")
#plt.title("ART-NCD-Entropy")
plt.show()


# --- TEST SUITE GENERATION ADAPTIVE RANDOM TESTING + ENTROPY NCD DISTANCE - CLASSIC ---

# initialise test suite
currentTS = generate_cl(1, num_genes)

# mds = MDS(random_state=0)
# currentTS2 = mds.fit_transform(currentTS)
#plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
#plt.show()

TS_size = 50

for i in range(TS_size):
    compute_distance_EntropyNCD(currentTS, TS_size)
    print(i)

TestSuite_NCD = currentTS
for i in range(len(TestSuite_NCD)):
    TestSuite_NCD[i]=list(TestSuite_NCD[i])

s=0
T = []
for i in range(len(TestSuite_NCD)):
    s = s+ TestSuite_NCD[(i+1):len(TestSuite_NCD)].count(TestSuite_NCD[i])
    s_i = TestSuite_NCD[(i+1):len(TestSuite_NCD)].count(TestSuite_NCD[i])
    if s_i == 0:
        T.append(i)
    else:
        T.append(i)
        T.append(i)

domain = np.unique(T,axis=0)

print()
print("TS SIZE: ")
print(TS_size)
print("ART ENTROPY NCD DISTANCE")
print("Duplicates: ")
print(s)
print("UNIFORM: ")
print(uniformComparison(TestSuite_NCD,eps2=0.1))

#plotting3D()

mds = MDS(random_state=0)
currentTS2 = mds.fit_transform(TestSuite_NCD)

plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
plt.title("ART-NCD-Entropy")
plt.show()



# --- TEST SUITE GENERATION ADAPTIVE RANDOM TESTING + EUCLIDEAN DISTANCE - CLASSIC + OUTPUT ---

# initialise test suite
currentTS = generate_cl_Y(1, num_genes)

mds = MDS(random_state=0)
currentTS2 = mds.fit_transform(currentTS)

#plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
#plt.show()
TS_size = 49

for i in range(TS_size):
    compute_distance_Y(currentTS, TS_size)
    print(i)

TestSuite_Two = currentTS
for i in range(len(TestSuite_Two)):
    TestSuite_Two[i]=list(TestSuite_Two[i])

s=0
T = []
for i in range(len(TestSuite_Two)):
    s = s+ TestSuite_Two[(i+1):len(TestSuite_Two)].count(TestSuite_Two[i])
    s_i = TestSuite_Two[(i+1):len(TestSuite_Two)].count(TestSuite_Two[i])
    if s_i == 0:
        T.append(i)
    else:
        T.append(i)
        T.append(i)

domain = np.unique(T,axis=0)

print()
print("TS SIZE: ")
print(TS_size)
print("ART EUCLIDEAN DISTANCE")
print("Duplicates: ")
print(s)
print("UNIFORM: ")
print(uniformComparison(TestSuite_Two,eps2=0.1))

t = np.array(TestSuite_Two)

zero=0
for i in range(len(t[:,5])):
    if t[:,5][i]==0:
        zero=zero+1

print("zero")
print(zero)

#plotting3D()

# --- 


# --- TEST SUITE GENERATION ADAPTIVE RANDOM TESTING + EUCLIDEAN DISTANCE - ENANCHED ---

# generate random dataset for k-means algorithm training
TS = np.random.uniform(0,1,(10,2))

for i in range (len(TS)):
    TestSuite.append(decode_columns(TS[i],bounds))
    TestSuite[i]=np.array(TestSuite[i])

from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic dataset with 8 random clusters
#X, y = make_blobs(n_samples=1000, n_features=12, centers=8, random_state=42)

# Instantiate the clustering model and visualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))

from sklearn import preprocessing
normalized = preprocessing.normalize(TestSuite)
visualizer.fit(normalized)        # Fit the data to the visualizer
visualizer.show()
n_clusters = int(visualizer.elbow_value_)

size = [64 for i in range(10)]
# fig = plt.figure(4, (10,4))
# ax = fig.add_subplot(2,2,1, projection='3d')
# ax.set_xlabel(features[0])
# ax.set_ylabel(features[1])
# ax.set_zlabel(features[2])
# plt.scatter(np.array(TestSuite)[:,0], np.array(TestSuite)[:,1], zs=np.array(TestSuite)[:,2], s=size)
# plt.title('Original Points')

# transformation num_genesD to 2D
mds = MDS(random_state=0)
X_transform = mds.fit_transform(TestSuite)


# ax = fig.add_subplot(2,2,2)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.plot(np.array(X_transform)[:,0], np.array(X_transform)[:,1],'o')
# plt.title('Transformed Points')
# fig.subplots_adjust(wspace=.4, hspace=0.5)


# k-means algorithm 

# from yellowbrick.cluster import KElbowVisualizer
# model = KMeans()
# visualizer = KElbowVisualizer(model, k=(1,12)).fit(df)
# visualizer.show()


kmeans = KMeans(n_clusters)
kmeans.fit(TestSuite)
identified_clusters = kmeans.fit_predict(TestSuite)
data_with_clusters = TestSuite.copy()
clusters = identified_clusters 
kmeans.inertia_
centroids = kmeans.cluster_centers_
kmeans.n_iter_

centroids = np.round(centroids,0).astype(int)
# ax = fig.add_subplot(2,2,3, projection='3d')
# ax.scatter(np.array(TestSuite)[:,0],np.array(TestSuite)[:,1],np.array(TestSuite)[:,2], c=list(clusters),s=size, cmap='rainbow')

#transformed points have same clusters of original poins
# kmeans = KMeans(n_clusters)
# kmeans.fit(X_transform)
# identified_clusters = kmeans.fit_predict(X_transform)
# data_with_clusters = X_transform.copy()
# clustersT = identified_clusters 
# kmeans.inertia_
# centroids2 = kmeans.cluster_centers_
# kmeans.cluster_centers_
# kmeans.n_iter_


# ax = fig.add_subplot(2,2,4)
# plt.scatter(X_transform[:,0],X_transform[:,1],c=list(clustersT),cmap='rainbow')
# plt.scatter(centroids2[:,0],centroids2[:,1],s=10,color='black')
#plt.show()
#plt.savefig("out.png")


invertedList = [[]for i in range(len(centroids))]
d_min = math.inf
TS_len = 0
TS_def = []
TS_size = 10000


    # STEP 1
    # generate first Test Case


TC1 = np.random.uniform(0,1,(1,2))
for i in range (len(TC1)):
    c = (decode_columns(TC1[i],bounds))
    #c = np.array(c)

for i in range(len(centroids)):
    d_c = distance2(c, centroids[i])
    if d_c < d_min:
        d_min = d_c
        NNcluster_ID = i

#c = list(c)
invertedList[NNcluster_ID].append(c)
TS_def.append(c)
TS_len = 1

while TS_len < TS_size:
    # STEP 2
    # generate set of candidate Test Cases
    C = []
    d_C_coarse = []
    d_C_fine = []
    NNclusters_ID = []
    TCs = np.random.uniform(0,1,(50,5))

    for i in range (len(TCs)):
        C.append(decode_columns(TCs[i],bounds))
        C[i]=np.array(C[i])
    
    # mds = MDS(random_state=0)
    # X_transformC = mds.fit_transform(C)
    #plt.scatter(np.array(X_transformC)[:,0], np.array(X_transformC)[:,1], s=50, marker="^", color='grey')


    # STEP 3
    # find NNcluster for each TC
    for i in range(len(C)):
        d_min = math.inf
        for j in range(len(centroids)):
            d_c = distance2(C[i], centroids[j])
            if d_c < d_min:
                d_min = d_c
                NNcluster_ID = j
        d_C_coarse.append(d_min)
        NNclusters_ID.append(NNcluster_ID)

    # STEP 4
    # find NN for each TC within NNcluster
    # each candidate test input is compared with the executed test inputs stored in the inverted list of the NN centroid
    newC = np.copy(C)
    newNNclusters_ID = np.copy(NNclusters_ID)
    removed_el_from_newC = []
    for i in range(len(C)):
        d_min = math.inf
        if len(invertedList[NNclusters_ID[i]]) > 0:
            for inverted_list_el in invertedList[NNclusters_ID[i]]:
                d_c = distance2(C[i], inverted_list_el)
                if d_c < d_min:
                    d_min = d_c
            d_C_fine.append(d_min)
        else:
            invertedList[NNclusters_ID[i]].append(list(C[i]))
            TS_def.append(list(C[i]))
            removed_el_from_newC.append(i)
            # mds = MDS(random_state=0)
            # X_transformD = mds.fit_transform(TS_def)
            #plt.scatter(np.array(X_transformD)[:,0], np.array(X_transformD)[:,1], s=50, marker="^", color='black')

            TS_len = TS_len + 1
            

    # STEP 5
    # find max among all TCs NN 
    # mds = MDS(random_state=0)
    # X_transformD1 = mds.fit_transform(TS_def)
    # plt.scatter(np.array(X_transformD1)[:,0], np.array(X_transformD1)[:,1], s=50, color='black')


    newC = np.delete(newC,removed_el_from_newC,axis=0)
    newNNclusters_ID = np.delete(newNNclusters_ID,removed_el_from_newC,axis=0)

    best_tc_id = d_C_fine.index(max(d_C_fine))
    best_tc = newC[best_tc_id]
    invertedList[newNNclusters_ID[best_tc_id]].append(list(best_tc))
    TS_def.append(list(best_tc))
    # mds = MDS(random_state=0)
    # X_transformD = mds.fit_transform([list(best_tc)])
    # plt.scatter(np.array(X_transformD)[:,0], np.array(X_transformD)[:,1], s=50, marker="^", color='black')

    TS_len = TS_len + 1
    print(TS_len)

TestSuite_Three = TS_def

X_transformD = mds.fit_transform(TestSuite_Three)
plt.scatter(np.array(X_transformD)[:,0], np.array(X_transformD)[:,1], color='black')
plt.title("ART-ENHANCHED-EUCLIDEAN")
plt.show()

s=0
T = []
for i in range(len(TestSuite_Three)):
    s = s+ TestSuite_Three[(i+1):len(TestSuite_Three)].count(TestSuite_Three[i])
    s_i = TestSuite_Three[(i+1):len(TestSuite_Three)].count(TestSuite_Three[i])
    if s_i == 0:
        T.append(i)
    else:
        T.append(i)
        T.append(i)

domain = np.unique(T,axis=0)

print()
print("TS SIZE: ")
print(TS_size)
print("ART EUCLIDEAN DISTANCE - ENHANCHED")
print("Duplicates: ")
print(s)
print("UNIFORM: ")
print(uniformComparison(TestSuite_Three,eps2=0.1))


# X_transformD = mds.fit_transform(TestSuite_Three)
# plt.scatter(np.array(X_transformD)[:,0], np.array(X_transformD)[:,1], s=50, marker="^", color='black')
# plt.plot()

# fig, ax = plt.subplots()
# sns.distplot(T, bins=len(T), color="g", ax=ax)
# plt.xticks(domain)
# plt.show()


# --- TEST SUITE GENERATION NCD DISTANCE - CLASSIC ---

# initialise test suite
currentTS = generate_cl(1, num_genes)

# mds = MDS(random_state=0)
# currentTS2 = mds.fit_transform(currentTS)
#plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
#plt.show()

TS_size = 50

for i in range(TS_size):
    dist(currentTS, TS_size)
    print(i)

TestSuite_NCD = currentTS
for i in range(len(TestSuite_NCD)):
    TestSuite_NCD[i]=list(TestSuite_NCD[i])

s=0
T = []
for i in range(len(TestSuite_NCD)):
    s = s+ TestSuite_NCD[(i+1):len(TestSuite_NCD)].count(TestSuite_NCD[i])
    s_i = TestSuite_NCD[(i+1):len(TestSuite_NCD)].count(TestSuite_NCD[i])
    if s_i == 0:
        T.append(i)
    else:
        T.append(i)
        T.append(i)

domain = np.unique(T,axis=0)

print()
print("TS SIZE: ")
print(TS_size)
print("NCD DISTANCE")
print("Duplicates: ")
print(s)
print("UNIFORM: ")
print(uniformComparison(TestSuite_NCD,eps2=0.1))

#plotting3D()

mds = MDS(random_state=0)
currentTS2 = mds.fit_transform(TestSuite_NCD)

plt.scatter(np.array(currentTS2)[:,0], np.array(currentTS2)[:,1], color='black')
plt.title("NCD")
plt.show()









# --- TEST UNIFORMITY COLLISION L2-NORM ---

import pandas

def nCk(n,k):
    return int( reduce(mul, (Fraction(n-i, i+1) for i in range(k)), 1) )

def uniformComparison(oriDist,eps2=0.1):
    if not(oriDist):
        print("Empty")
        return
    dist = pandas.DataFrame(oriDist)
    domain=np.unique(dist,axis=0)
    #The numer of samples come from lemma 5 of collision-based testers are optimal for uniformity and closeness
    expected= 6*sqrt(len(domain))/eps2
    if(len(oriDist) < expected):
        print("You need " + str(expected) + " samples")
    else:
        print("The domain lenght is: "+str(len(domain)))
    s = 0
    for i in range(len(oriDist)):
        s = s+ oriDist[(i+1):len(oriDist)].count(oriDist[i])
    t= nCk(len(oriDist),2)*(1+3/4*eps2)/len(domain)
    if(s>t):
        print("Fail")
    else:
        print("Done")

# --- 



#save interactive rotating 3d plot
# import matplotlib.pyplot as plt 
# import numpy as np 

# plt.rcParams["figure.figsize"] = [7.00, 3.50] 
# plt.rcParams["figure.autolayout"] = True 
# fig = plt.figure() 
# ax = fig.add_subplot(111, projection='3d') 
# u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j] 
# x = np.cos(u) * np.sin(v) 
# y = np.sin(u) * np.sin(v) 
# z = np.cos(v) 
# ax.plot_wireframe(x, y, z, color="red") 
# ax.set_title("Sphere") 
# plt.savefig("test.pdf") 
# plt.show()


# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.set_xlabel(features[0])
# ax.set_ylabel(features[1])
# ax.set_zlabel(features[2])
# ax.plot(np.array(TestSuite)[:,0], np.array(TestSuite)[:,1], np.array(TestSuite)[:,2],'o', c=colors)
# plt.show(


# ----


# ---- TEST TRANSFORMATION 3D - 2D ---- 
# colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'grey', 'olive','pink']

# size = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
# fig = plt.figure(2, (10,4))
# ax = fig.add_subplot(121, projection='3d')
# ax.set_xlabel(features[0])
# ax.set_ylabel(features[1])
# ax.set_zlabel(features[2])
# plt.scatter(np.array(TestSuite)[:,0], np.array(TestSuite)[:,1], zs=np.array(TestSuite)[:,2], s=size, c=colors)
# plt.title('Original Points')

# mds = MDS(random_state=0)
# X_transform = mds.fit_transform(TestSuite)
# #print(X_transform)

# ax = fig.add_subplot(122)
# plt.xlabel("X")
# plt.ylabel("Y")
# #plt.plot(np.array(X_transform)[:,0], np.array(X_transform)[:,1],'o')
# plt.scatter(np.array(X_transform)[:,0], np.array(X_transform)[:,1], s=size, c=colors)
# plt.title('Transformed Points')
# fig.subplots_adjust(wspace=.4, hspace=0.5)
# plt.show()

# --- 


# --- TEST SUITE GENERATION RANDOM.UNIFORM ---

# TS = np.random.uniform(0,1,(TS_size,num_genes))

# for i in range (len(TS)):
#     TestSuite.append(decode_columns(TS[i],bounds))
#     TestSuite[i]=np.array(TestSuite[i])
#     #TestSuite = TestSuite.astype(int)

# --- 

for element in TestSuite: 
    y.append(model([element])) 


#dist_euclid = euclidean_distances(TestSuite)
#n = norm(TestSuite[0]-TestSuite[1])


# ----------------------------- GA -----------------------------

individual = Individual(None, TestSuite, y, p)
population = individual.init_population(pop_size, num_genes)

for ch in population:
    ch.get_fitness(True)

best_parent_fitness = 0


for i in range (pop_size):
    if population[i].fitness > best_parent_fitness:
        best_parent_fitness = population[i].fitness
        best_parent = deepcopy(population[i])

print("GENERATION 0 - BEST INDIVIDUAL:")
print(best_parent.genome)
print(best_parent.fitness)
print()

print("---INITIAL POPULATION---")
for i in range (pop_size):
    print("Individual:",i)
    print(population[i].genome)
    print(population[i].fitness)
    print()



# ----------------------------- EVO -----------------------------

for gen in range(num_generations):

    parents = []
    for _ in range(pop_size-1):
        parents.append(tournament_selection(population))

    offspring = list()

    for i in range(0, pop_size-1, 2):
        # get selected parents in pairs
        if i == pop_size-2:
            p1, p2 = parents[i], parents[0]
        else:
            p1, p2 = parents[i], parents[i+1]
        # crossover and mutation
        for c in crossover(p1, p2, 0.2):
            # store for next generation
            cc=mutation(c,0.2)
            cc.get_fitness(False)
            offspring.append(cc)

    offspring[-1] = best_parent
    population = offspring

    best_parent_fitness = 0

    for i in range (pop_size):
        if population[i].fitness > best_parent_fitness:
            best_parent_fitness = population[i].fitness
            best_parent = deepcopy(population[i])

    print("GENERATION - ", gen, "BEST INDIVIDUAL:")
    print(best_parent.genome)
    print(best_parent.fitness)
    print()

print("---FINAL POPULATION---")
for i in range (pop_size):
    print("Individual:",i)
    print(population[i].genome)
    print(population[i].fitness)
    print()
