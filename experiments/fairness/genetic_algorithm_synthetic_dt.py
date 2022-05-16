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
import itertools
import math
from sklearn import manifold


import pickle


#LAST ONE WORKING ON HIRING SYNTHETIC DATA

def decide(applicant):
    gender = 1
    if applicant[gender] == 1:
        return 0
    else:
        return 1

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
    ind_TMP = np.copy(ind_A) #

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

model = decide4

result = []
y = []
TestSuite = []

#number of Test Cases (1 matrix)
TS_size = 40000 #100000

#number of individuals/instances = partitions
pop_size = 10

#number of features
num_genes = len(var)

#number generations
num_generations = 30

features = ["age","gender","marital_status","education","lift_heavy_weight"]
bounds = [(18,50),(0,1),(0,1),(0,3),(10,50)]
binary_bounds = [(0,1),(0,1),(0,1),(0,1),(0,1)]

freq = [[] for i in range(num_genes)]  
uq = [[] for i in range(num_genes)]  
prob = [[] for i in range(num_genes)]  


TestSuite = pickle.load(open("./experiments/fairness/art_euclidean_1000.dat", "rb"))


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

TestSuite = [list(TestSuite[i]) for i in range(len(TestSuite))]

s=0
T = []
for i in range(len(TestSuite)):
    s = s+ TestSuite[(i+1):len(TestSuite)].count(TestSuite[i])
    s_i = TestSuite[(i+1):len(TestSuite)].count(TestSuite[i])
    if s_i == 0:
        T.append(i)
    else:
        T.append(i)
        T.append(i)

domain = np.unique(T,axis=0)

print()
print("TS SIZE: ")
print(40000)
print("NCD DISTANCE")
print("Duplicates: ")
print(s)
print("UNIFORM: ")
print(uniformComparison(TestSuite,eps2=0.1))


# mds = MDS(random_state=0)
# biDTS = mds.fit_transform(TestSuite)

# d2 = euclidean_distances(biDTS, biDTS)
# seed = np.random.RandomState(seed=3)

# mds = manifold.MDS(n_components=3, metric=True, max_iter=3000, eps=1e-9, random_state=seed,
#                    dissimilarity="precomputed", n_jobs=1)
# embed3d = mds.fit(d2).embedding_


# fig = plt.figure(figsize=(13,5))
# subpl2 = fig.add_subplot(121)
# subpl2.scatter(biDTS[:, 0], biDTS[:, 1],s=10)
# subpl = fig.add_subplot(122,projection='3d')
# subpl.scatter(embed3d[:, 0], embed3d[:, 1], embed3d[:, 2],s=10)
# plt.title("ART EUCLIDEAN DISTANCE")
# plt.show()




#--- rnd 

# def decode_columns(arr, bounds):
#     d_arr = []
#     for i in range(len(arr)):
#         _min, _max = bounds[i]
#         value = arr[i]
#         if (_max - _min) == 1:
#             d_arr.append(round(value))
#         else:
#             d_arr.append(round(value*(_max-_min)+_min))
#     return d_arr


# TS_size = 40000

# TestSuite1 = []

# TS = np.random.uniform(0,1,(TS_size,num_genes))

# for i in range (len(TS)):
#     TestSuite1.append(decode_columns(TS[i],bounds))
#     TestSuite1[i]=np.array(TestSuite1[i])

# TestSuite = TestSuite1
# for i in range(len(TestSuite)):
#     TestSuite[i]=list(TestSuite[i])

#--- 

# calculate probability distribution of TestSuite
var_values = [[x[i] for x in TestSuite] for i in range(num_genes)]

for i in range(num_genes):
    freq[i] = dict()
    uq[i] = np.unique(var_values[i])
    freq[i] = dict.fromkeys(uq[i], 0)

    for e in var_values[i]:
        freq[i][e]+=1

for i in range(num_genes):
    for j in range(len(uq[i])):
        prob[i].append(list(freq[i].values())[j]/len(var_values[i]))


# calculate joint probability distribution of two/three variables
X = [0,0,1,0,1,1]
#Y = [27,18,32,33,40,20]
Z = [20,20,20,20,20,21]

X_uq = np.unique(X)
#Y_uq = np.unique(Y)
Z_uq = np.unique(Z)

XY_uq = []
XY_pairs = []
XY_uq.append(X_uq)
#XY_uq.append(Y_uq)
XY_uq.append(Z_uq)



XY_pairs_d = dict()
for i in range(len(X)):
    #XY_pairs[(X[i],Y[i],Z[i])] = 0
    XY_pairs_d[(X[i],Z[i])] = 0
    XY_pairs.append((X[i],Z[i]))

XY_cart = []

for element in itertools.product(*XY_uq):
    XY_cart.append(element)

# for key in XY_cart.keys():
#     XY_cart[key] = XY_pairs.count(key)
#     XY_cart[key] = XY_cart[key]/len(X)

# XY_pairs_list = XY_pairs_d.keys()
# XY_pairs_list = [key for key in XY_pairs_list]

XY_pairs_list = XY_pairs

for element in XY_cart:
    if element in XY_pairs:
        XY_pairs_d[element] = XY_pairs_list.count(element)
        XY_pairs_d[element] = XY_pairs_d[element]/len(X)

joint_prob = XY_pairs_d.values()

#sum(XY_cart.values())

joint_prob = [value for value in joint_prob]

joint_entropy = -(sum([(joint_prob[i]*math.log(joint_prob[i],2)) for i in range(len(joint_prob))]))

p_Z = [5/6,1/6]




# fig, ax = plt.subplots()
# plt.bar(uq[0],prob[0])
# plt.show()

# --- TEST SUITE GENERATION random.uniform() ---

# TS_size = 100000

# TS = np.random.uniform(0,1,(TS_size,num_genes))

# for i in range (len(TS)):
#     TestSuite.append(decode_columns(TS[i],bounds))
#     TestSuite[i]=np.array(TestSuite[i])

# TestSuite = TestSuite
# for i in range(len(TestSuite)):
#     TestSuite[i]=list(TestSuite[i])


#-----

# biDTS = pickle.load(open("./experiments/fairness/art_euclidean_10000_2D.dat", "rb"))

# d2 = euclidean_distances(biDTS, biDTS)

# from sklearn import manifold
# seed = np.random.RandomState(seed=3)

# mds = manifold.MDS(n_components=3, metric=True, max_iter=3000, eps=1e-9, random_state=seed,
#                    dissimilarity="precomputed", n_jobs=1)
# embed3d = mds.fit(d2).embedding_

# fig = plt.figure(figsize=(5*3,4.5))
# subpl2 = fig.add_subplot(133)
# subpl2.scatter(biDTS[:, 0], biDTS[:, 1],s=20)
# subpl = fig.add_subplot(132,projection='3d')
# subpl.scatter(embed3d[:, 0], embed3d[:, 1], embed3d[:, 2],s=20)
# plt.show()


# mds = MDS(random_state=0)
# biDTS = mds.fit_transform(TestSuite)

# plt.scatter(np.array(biDTS)[:,0], np.array(biDTS)[:,1], size=15, color='black')
# plt.title("RND.UNIFORM()")
# plt.show()

#-----

for element in TestSuite: 
    y.append(model([element])) 
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
