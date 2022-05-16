import random
import numpy as np
import zlib
import math


# --- FUNCTIONS ---

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


def maximize_y(Y):
    y_len_max = 0
    y_len_min = math.inf
    y_max = []
    y_min = []

    for i in range(len(Y)):
        y = np.delete(Y, i, axis=0)
        y_C = ""
        for sub_el in y:
            y_C += str(zlib.compress(str(sub_el).encode()))
        if len(y_C)>y_len_max:
            y_len_max = len(y_C)
            y_max_el = Y[i]
            y_max = str(zlib.compress(str(y_max_el).encode()))
            y_max_index = i
    Y_C = ""           
    for sub_el_2 in Y:
        Y_C += str(zlib.compress(str(sub_el_2).encode()))
        sub_el_2_C = str(zlib.compress(str(sub_el_2).encode()))
        if len(sub_el_2_C)<y_len_min:
                y_len_min = len(sub_el_2_C)
                y_min = sub_el_2_C

    Y_k = np.delete(Y, y_max_index, axis=0)

    Y_k_C = ""
    for sub_el_3 in Y_k:
        Y_k_C += str(zlib.compress(str(sub_el_3).encode()))

    print(len(Y_C),len(y_min),len(Y_k_C))
    NCD.append( ((len(Y_C) - len(y_min))/len(y_max) , Y_k) )

    if len(Y_k) > 2:
        maximize_y(Y_k)


# --------------------------------


NCD = []

TS_size = 10000
num_genes = 2
bounds = [(8,17),(8,17)]


# --- INITIAL TS RND GENERATION ---

TS = np.random.uniform(0,1,(TS_size,num_genes))

for i in range (len(TS)):
    TS.append(decode_columns(TS[i],bounds))
    TS[i]=np.array(TS[i])

TS_init = TS
for i in range(len(TS_init)):
    TS_init[i]=list(TS_init[i])

# ----------------------------------


maximize_y(np.array(TS_init))

TSD_init = max(NCD,key=lambda item:item[0])
TSD_init = TSD_init[0]



def calculate_fitness(candidateTC):
    
    TS = TS_init.append(candidateTC)
    maximize_y(np.array(TS))
    TSD = max(NCD,key=lambda item:item[0])
    scores = TSD
    #using f1_score as it is an imbalanced dataset
    
    return scores


  
def get_fitness(population,data):
  fitness_values = []
  for individual in population:
    df = data
    i=0
    for column in data:
      if(individual[i]==0):
        df = df.drop(column,axis=1)
      i=i+1

    features = df
    individual_fitness = calculate_fitness(features)
    fitness_values.append(individual_fitness)

  return fitness_values


def select_parents(population,fitness_values):
  parents = []
  total = sum(fitness_values)
  norm_fitness_values = [x/total for x in fitness_values]

  #find cumulative fitness values for roulette wheel selection
  cumulative_fitness = []
  start = 0
  for norm_value in norm_fitness_values:
    start+=norm_value
    cumulative_fitness.append(start)

  population_size = len(population)
  for count in range(population_size):
    random_number = random.uniform(0, 1)
    individual_number = 0
    for score in cumulative_fitness:
      if(random_number<=score):
        parents.append(population[individual_number])
        break
      individual_number+=1
      
  return parents


def two_point_crossover(parents,probability):
    random.shuffle(parents)
    #count number of pairs for crossover
    no_of_pairs = round(len(parents)*probability/2)
    chromosome_len = len(parents[0])
    crossover_population = []
  
    for num in range(no_of_pairs):
      length = len(parents)
      parent1_index = random.randrange(length)
      parent2_index = random.randrange(length)
      while(parent1_index == parent2_index):
        parent2_index = random.randrange(length)
        
      start = random.randrange(chromosome_len)
      end = random.randrange(chromosome_len)
      if(start>end):
        start,end = end, start

      parent1 = parents[parent1_index]
      parent2 = parents[parent2_index]
      child1 =  parent1[0:start] 
      child1.extend(parent2[start:end])
      child1.extend(parent1[end:])
      child2 =  parent2[0:start]
      child2.extend(parent1[start:end])
      child2.extend(parent2[end:])
      parents.remove(parent1)
      parents.remove(parent2)
      crossover_population.append(child1)
      crossover_population.append(child2)

    #to append remaining parents which are not undergoing crossover process
    if(len(parents)>0):
      for remaining_parents in parents:
        crossover_population.append(remaining_parents)

    return crossover_population
    
def mutation(crossover_population):
    #swapping of zero with one to retain no of features required
    for individual in crossover_population:
      index_1 = random.randrange(len(individual))
      index_2 = random.randrange(len(individual))
      while(index_2==index_1 and individual[index_1] != individual[index_2]):
        index_2 = random.randrange(len(individual))

      #swapping the bits
      temp = individual[index_1]
      individual[index_1] = individual[index_2]
      individual[index_2] = temp

    return crossover_population