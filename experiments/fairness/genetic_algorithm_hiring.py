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






# ----------------------------------------------------------------
# ----------------------------- main -----------------------------
# ----------------------------------------------------------------

var = ["age","gender","marital_status","education","lift_heavy_weight"]

#FITNESS FUNCTION

p = fitness()

model = decide3


result = []
y = []
TestSuite = []

#number of Test Cases (1 matrix)
TS_size = 10 #100000

#number of individuals/instances = partitions
pop_size = 10

#number of features
num_genes = len(var)

#number generations
num_generations = 10

features = ["age","gender","marital_status","education","lift_heavy_weight"]
bounds = [(18,50),(0,1),(0,1),(0,3),(10,50)]
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

def distance(s1, s2):
    ab = sum(np.multiply(s1, s2))
    a_pwr = math.sqrt(sum(np.power(s1, 2)))
    b_pwr = math.sqrt(sum(np.power(s2, 2)))
    return ab / (a_pwr * b_pwr)



TS_size = 100

currentTS = []
checkTestSuite = []

#checkList = np.random.uniform(0,1,(TS_size,num_genes))


# generate check list
def generate_cl(size,genes):
    check_ts = []
    cl = np.random.uniform(0,1,(size,genes))
    for i in range (len(cl)):
        check_ts.append(decode_columns(cl[i],bounds))
        check_ts[i]=np.array(check_ts[i])
    return check_ts

# initialise test suite
currentTS = generate_cl(1, num_genes)


def compute_distance(ts: List, TS_size):
    candidates = generate_cl(10, num_genes)
    distances = []
    elements = []
    for t in ts:
        d_min = 2 # math.inf
        elem = None
        for c in candidates:
            d = distance(c, t)
            if d < d_min:
                d_min = d
                elem = c
        distances.append(d_min)
        elements.append(elem)

    best = max(distances)
    ts.append(elements[distances.index(best)])
    if len(ts) < TS_size:
        compute_distance(currentTS, TS_size)

compute_distance(currentTS, TS_size)


'''
for plotting

from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection="3d")

plt.show()

ax.plot3D(np.array(currentTS)[:,0], np.array(currentTS)[:,1], np.array(currentTS)[:,2],'o')

'''





#------
TS = np.random.uniform(0,1,(TS_size,num_genes))

for i in range (len(TS)):
    TestSuite.append(decode_columns(TS[i],bounds))
    TestSuite[i]=np.array(TestSuite[i])
    #TestSuite = TestSuite.astype(int)

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
