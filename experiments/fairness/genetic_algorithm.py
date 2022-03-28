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


var = ["age","gender","marital_status","education","lift_heavy_weight"]

#FITNESS FUNCTION

p = fitness()

result = []
y = []

#number of Test Cases (1 matrix)
TS_size = 10

#number of individuals/instances = partitions
pop_size = 10

#number of features
num_genes = len(var)

#number generations
num_generations = 10

features = ["age","gender","marital_status","education","lift_heavy_weight"]
bounds = [(18,50),(0,1),(0,1),(0,3),(10,50)]


#create Test Suite
features_inst = [[]for j in range(num_genes)]
TestSuite = []

for k in range(num_genes):
    features_inst[k] = [random.randint(bounds[k][0],bounds[k][1]) for i in range(TS_size)]

temp = np.array(features_inst)

for i in range(TS_size):
    TestSuite.append(temp[:,i])


#female=1 male=0
def decide(applicant):
    gender = 1
    if applicant[0][gender] == 1:
        return 0
    else:
        return 1


model = decide 

for element in TestSuite: 
    y.append(model([element]))     
  

individual = Individual(None, TestSuite, y, p)
population = individual.init_population(pop_size, num_genes)

for ch in population:
    ch.get_fitness()


def tournament_selection(population: List, k: int = 2) -> Iterator:

    while True:
        #randomly choose k individuals (2 at a time) and select the best one
        choices = random.choices(population, k=k)
        if choices[0].fitness > choices[1].fitness:
            best = choices[0]
        else:
            best = choices[1]
        
        yield best


def _uniform_crossover(next_individual: Iterator,
                      p_swap: float = 0.5) -> Iterator:

    def _uniform_crossover(ind1, ind2, p_swap):
  
        if len(ind1.genome) != len(ind2.genome):
            # TODO what about variable length genomes?
            raise RuntimeError(
                'genomes must be same length for uniform crossover')

        ind_A = np.array(ind1.genome) #
        ind_B = np.array(ind2.genome) #
        ind_TMP = copy(ind_A) #

        for i in range(len(ind1.genome)):
            if random.random() < p_swap:
                ind_TMP[i] = ind_B[i] #
                ind_B[i] = ind_A[i] #
                ind_A[i] = ind_TMP[i] #
                #ind1.genome[:,i], ind2.genome[:,i] = ind2.genome[:,i], ind1.genome[:,i]

        ind1.genome = list(ind_A) #
        ind2.genome = list(ind_B) #

        return ind1, ind2

    while True:
        parent1 = next(next_individual)
        parent2 = next(next_individual)

        child1, child2 = _uniform_crossover(parent1, parent2, p_swap)

        yield child1
        yield child2


generation_counter = util.inc_generation(context=context)

#results = []
while generation_counter.generation() < num_generations:
    p.setStat()
    print("GENERATION ", generation_counter.generation()+1)
    #sequence of functions, the result of the first one will be the parameter of the next one, and so on
    offspring = pipe(population,
                     #probe.print_individual(prefix='before tournament: '),
                     tournament_selection,
                     #probe.print_individual(prefix='after tournament: \n'),
                     ops.clone,
                     #mutate_bitflip,
                     #probe.print_individual(prefix='before mutation: '),
                     #individual_mutate_randint,
                     #probe.print_individual(prefix='after mutation: '),
                     #probe.print_individual(prefix='before crossover: \n'),
                     _uniform_crossover(p_swap=0.2),
                     #probe.print_individual(prefix='after crossover: \n\n\n'),
                     ops.evaluate,
                     ops.pool(size=len(population)))  # accumulate offspring

   
    population = offspring
 

    #print(probe.best_of_gen(parents))

    generation_counter()  # increment to the next generation

    #util.print_population(parents, context['leap']['generation'])
    
    # count=0
    # parents_pairs={}
    # #parents_pairs collect position individual and fitness function (for each individual in the current population (in the current generation))
    # for i in range(len(parents)):
    #     parents_pairs[i] = parents[i].fitness
    # #sort individuals in the current population in an ascending order (by Fitness Function)
    # import operator
    # sorted_d = sorted(parents_pairs.items(), key=operator.itemgetter(1))

    # #for key, value in sorted_d:
    #     #print("generation", context['leap']['generation'])
    #     #count=count+1
    #     #print("individual", count)
    #     #print(parents[key].genome)
    #     #print(parents[key].fitness)
    #     #print(p.getStat()[key])
    # #for individual in parents:
    # #    print("generation", context['leap']['generation'])
    # #    print(p.getStat()[count])
    # #    count=count+1
    # #    print("individual", count)
    #     #print(individual.genome)
    # #    print(individual.fitness)
    # print("generation", context['leap']['generation'])

    # #print worse and best (FF) individual in the population for each generation (showing FF + num meaningful features + name meaningful features + MI of feature)
    # print("worst: ",p.getStat()[sorted_d[0][0]])
    # print("best: ", p.getStat()[sorted_d[-1][0]])
    # print()
    # best = probe.best_of_gen(parents)

 

best_parent_fitness = 0

for i in range (pop_size):
    if parents[i].fitness > best_parent_fitness:
        best_parent_fitness = parents[i].fitness
        best_parent = parents[i]

print(best_parent.genome)




