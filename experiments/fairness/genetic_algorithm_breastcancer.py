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


#680
dataset = pd.read_csv("experiments/data/breast-cancer-red.csv")

# Check for Null Data
dataset.isnull().sum()

# Replace All Null Data in NaN
dataset = dataset.fillna(np.nan)

# Peek at data
print(dataset.head(2))

#feature_names=['Sex_Code_Text','Ethnic_Code_Text','Language','LegalStatus','CustodyStatus','MaritalStatus','RecSupervisionLevel','Age','ScoreText']


# Remove id column
dataset.drop(labels=["Sample code number"], axis = 1, inplace = True)


# Drop the NaN rows now 
dataset.dropna(how='any',inplace=True)

# Split-out Validation Dataset and Create Test Variables
array = dataset.values
var = dataset.columns 

X = array[:,0:9]



# for i in range (len(dataset)):
#     X[:,5][i] = int(X[:,5][i])

y = array[:,9]
print('Split Data: X')
#print(X)
print('Split Data: Y')
#print(y)


var = np.array(var)
features = var[0:-1]


result = []
TestSuite = []

TestSuite = X
#FITNESS FUNCTION

p = fitness()

#number of Test Cases (1 matrix)
TS_size = len(dataset)

#number of individuals/instances = partitions
pop_size = 10

#number of features
num_genes = len(var)-1

#number generations
num_generations = 50



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
