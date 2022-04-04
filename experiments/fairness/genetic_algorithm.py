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
from toolz import curry


var = ["age","gender","marital_status","education","lift_heavy_weight"]

#FITNESS FUNCTION

p = fitness()

result = []
y = []
TestSuite = []

#number of Test Cases (1 matrix)
TS_size = 100000

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


TS = np.random.uniform(0,1,(TS_size,num_genes))
for i in range (len(TS)):
    TestSuite.append(decode_columns(TS[i],bounds))
    TestSuite[i]=np.array(TestSuite[i])
    #TestSuite = TestSuite.astype(int)


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

@curry
def mutate(next_individual: Iterator)-> Iterator:
    #swapping of zero with one to retain no of features required
    #for individual in population:
    print("before")
    individual = next(next_individual)
    print(individual.genome)
    index_1 = random.randrange(len(individual.genome))
    index_2 = random.randrange(len(individual.genome))
    while(index_2==index_1 and individual.genome[index_1] != individual.genome[index_2]):
        index_2 = random.randrange(len(individual.genome))

    #swapping the bits
    temp = individual.genome[index_1]
    individual.genome[index_1] = individual.genome[index_2]
    individual.genome[index_2] = temp
    print("after")
    print(individual.genome)
    yield individual

@curry
def pool(next_individual: Iterator, size: int) -> List:
    print(size)
    return [next(next_individual) for _ in range(size)]


best_parent_fitness = 0

for i in range (pop_size):
    if population[i].fitness > best_parent_fitness:
        best_parent_fitness = population[i].fitness
        best_parent = population[i]

print("GENERATION 0 - BEST INDIVIDUAL:")
print(best_parent.genome)
print(best_parent.fitness)
print()

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
                     ops.uniform_crossover(p_swap=0.2),
                     #mutate,
                     #probe.print_individual(prefix='after crossover: \n\n\n'),
                     ops.evaluate,
                     ops.pool(size=len(population)))  # accumulate offspring


    population = offspring
 

    #print(probe.best_of_gen(parents))

    best_parent_fitness = 0

    for i in range (pop_size):
        if population[i].fitness > best_parent_fitness:
            best_parent_fitness = population[i].fitness
            best_parent = population[i]

    print("GENERATION - ",  generation_counter.generation()+1, "BEST INDIVIDUAL:")
    print(best_parent.genome)
    print(best_parent.fitness)
    print()

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

 






