import random
import string

import sys
import random
from pyitlib import discrete_random_variable as drv
import numpy as np
from individual import Individual
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from leap_ec.int_rep.ops import mutate_randint, individual_mutate_randint
from scipy import spatial
from problemMultiCMItest import fitness
from one_hot_encode import encode_columns_splits
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
import numpy as np
import matplotlib.pyplot as plt
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
import pandas
import pickle


# var = ["username","password"]
# features = ["username","password"]



bounds = [(8,17),(8,17)]

result = []
y = []
TestSuite = []

#number of Test Cases (1 matrix)
TS_size = 200 #100000

#number of individuals/instances = partitions
pop_size = 10

#number of features


#number generations
num_generations = 50





def decideSecret(user):
    output = 0
    confidential = 1
    if user[0][confidential] == 10:
        output = 1
    else:
        output = 0
    #output = 1
    return output




def init(length):
    TS = create_string_vector(TS_size,bounds)
    return TS
        
def create_string_vector(size, bounds):
    return [[string_generator(random.randint(min_, max_)) for min_, max_ in bounds] for i in range(size)]

def string_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

TestSuite = init(TS_size)
print()

var = []

# for i in range(len(TestSuite)):
#     var.append("U_" + str(len(TestSuite[i][0])))

# for i in range(len(TestSuite)):
#     var.append("P_" + str(len(TestSuite[i][1])))

# var = np.unique(var,axis=0)

# features = var

# num_genes = len(var)


#TestSuite = pickle.load(open("./experiments/info_flow/art_euclidean_psw_1000.dat", "rb"))


# TO RESTORE
data = [[]for j in range(len(TestSuite))]

for i in range(len(data)):
    data[i].append(len(TestSuite[i][0]))
    data[i].append(len(TestSuite[i][1]))


splits = [3,3]
types = ['U','P']
names=[]

# TO RESTORE
TestSuite = np.array(data)
TestSuite = np.array(TestSuite)

TestSuite2 = encode_columns_splits(TestSuite, bounds, splits)


for s in range(len(splits)):
    names.append([types[s] for _ in range(splits[s])])

names = list(np.array(names).flatten())
var = []
for j in range(len(TestSuite2[1])):
    a,b=TestSuite2[1][j]['borders']
    var.append(names[j]+str(a)+'-'+str(b))


features = var

num_genes = len(var)

print()
# from sklearn import preprocessing
# df = pd.DataFrame(data)
# enc = preprocessing.OneHotEncoder()
# enc.fit(df)
# onehotdata = enc.transform(df).toarray()

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


#FITNESS FUNCTION

p = fitness()

model = decideSecret




#-----

for element in TestSuite: 
    y.append(model([element])) 
# ----------------------------- GA -----------------------------

TestSuite2 = TestSuite2[0]
#was TestSuite
individual = Individual(None, TestSuite2, y, p, var)
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
