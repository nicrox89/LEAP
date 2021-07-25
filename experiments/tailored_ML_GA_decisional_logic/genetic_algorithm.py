import sys
from leap_ec.algorithm import generational_ea
from leap_ec import representation, ops

from leap_ec.segmented_rep import initializers, decoders
#from problems import fitness
#from problem import fitness
from problemMultiCMI import fitness
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.int_rep.initializers import create_int_vector
from leap_ec.segmented_rep.initializers import create_segmented_sequence
from leap_ec.segmented_rep.ops import apply_mutation
from leap_ec import decoder 
from leap_ec import probe
from leap_ec.binary_rep.ops import mutate_bitflip
from leap_ec.individual import Individual
from leap_ec import util
from leap_ec.context import context
from toolz import pipe
from leap_ec.int_rep.ops import mutate_randint, individual_mutate_randint
import random
from pyitlib import discrete_random_variable as drv
import numpy as np

from leap_ec.problem import FunctionProblem

from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import shap
import pickle
from datetime import datetime



#now = datetime.now().strftime("%Y%m%d_%H%M%S")
#sys.stdout = open('experiments/tailored_ML_GA_decisional_logic/decisional_GA_output_CMI_Decide_Test_100-1000-50<=30'+now+'.out', 'w')



n_Ind = 3000
var = ["age","gender","marital_status","education","lift_heavy_weight"]
var_2 = ["age","Gender:Female","Gender:Male", "marital_status","education","lift_heavy_weight"]
var_values = [[18,50],['F','M'],['single', 'married'],['primary', 'secondary', 'further', 'higher'],[10,50]]
gender_values = [[1,0]]

rows, cols = (n_Ind, var) 
applicants = [[]for j in range(rows)] 
applicants_test = [[]for j in range(rows)] 

genders = ['F','M']
marital_statuses = ['single', 'married']
educations = ['primary', 'secondary', 'further', 'higher']

age_vector = []
gender_vector = []
marital_status_vector = []
education_vector = []
lift_heavy_weight_vector = []
hiring_vector = []
tot_infl = []

change_position = []
change_position_F = []
l = 0
l_F = 0
num_F = []

y = []


#classification logic

def decide2(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if applicant[0][gender] == 1:
        #r = random.randint(0, 1)
        #if r > 0:
        #if features[heavy_weight][n] >= 40:
        if (applicant[0][age] <= 30):
            return 1
        else:
            return 0
    elif applicant[0][gender] == 0:
        #r = random.randint(0, 1)
        #if r > 0:
        #if features[heavy_weight][n] >= 40:
        if (applicant[0][age] <= 30):
            return 1
        else:
            return 0
        #else:
        #    return 0
    #else:
    #    return 1

def decide3(applicant):
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

def decide4(applicant):
    gender = 1
    heavy_weight = 4
    marital_status = 2
    age = 0
    if applicant[0][gender] == 1:
        if  applicant[0][marital_status] == 1:
            return 1
        else:
            return 0
    else:
        return 1

def decide5(applicant):
    gender = 1
    heavy_weight = 4
    marital_status = 2
    age = 0
    if applicant[0][gender] == 1:
        if  applicant[0][heavy_weight] >= 30:
            return 1
        else:
            return 0
    else:
        return 1

def decide(applicant):
    gender = 1
    if applicant[0][gender] == 1:
        return 0
    else:
        return 1

def decide_Test(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if  applicant[0][heavy_weight] >= 41:
        if applicant[0][gender] == 1:
            return 1
        else:
            return 0
    elif applicant[0][heavy_weight] < 41:
        if applicant[0][gender] == 1:
            return 1
        else:
            return 0

def decide_Test2(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if  applicant[0][heavy_weight] >= 21:
        if applicant[0][gender] == 1:
            return 1
        else:
            return 1
    elif applicant[0][heavy_weight] < 21:
        if applicant[0][age] <= 30:
            return 1
        else:
            return 0

def decide_Test3(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if  applicant[0][gender] == 0:
        if applicant[0][2] == 1:
            return 1
        else:
            return 0
    elif applicant[0][gender] == 1:
        if applicant[0][age] <= 30:
            return 1
        else:
            return 0

def decideAll(applicant):
    age = applicant[0][0]
    gender = applicant[0][1]
    marital_status = applicant[0][2]
    education = applicant[0][3]
    heavy_weight = applicant[0][4]

    if gender == 1:
        if marital_status != 0:
            if  heavy_weight <= 30 and age >= 30:
                return 0
            else:
                return 1 # female not married with heavy weight >= 30 and age <= 30
        else:
            if education == 3:
                return 1 # hi education female married
            else:
                return 0 # low education female married
    else:
        #return 1 # male
        if education == 3:
            return 1 # hi education female married
        else:
            return 0 


#FITNESS FUNCTION

p = fitness(decide3)

result = []

#number of individuals (matrixs)
pop_size = 100
#number of instances for each gene(variable) = number of records(observations) of the matrix
gene_size = 10000
#number of features
num_genes = len(var)

features = ["age","gender","marital_status","education","lift_heavy_weight"]
bounds = [(18,50),(0,1),(0,1),(0,3),(10,50)]
#bounds = [(2,3),(0,1),(0,1),(0,1),(20,21),(30,34)]


def set_Partition():
        #num_partition_features = 2
        minPartition_size = 1
        maxPartition_size = num_genes-1
        #maxPartition_size = num_genes-1
        num_partition_features = random.randint(minPartition_size,maxPartition_size)
        partition_features_index = random.sample(range(0, num_genes), num_partition_features)          
        selected_partition_features_index=np.zeros(num_genes)
        selected_partition_features_index[partition_features_index]=1
        return selected_partition_features_index


def init(length, seq_initializer):
    def create():
        ind = initializers.create_segmented_sequence(gene_size, create_int_vector(bounds))
        ind.append([int(x) for x in list(set_Partition())])
        return ind
    return create

#create initial rand population of pop_size individuals

parents = Individual.create_population(n=pop_size,
                                       initialize=init(gene_size, create_int_vector(bounds)),
                                       decoder=decoder.IdentityDecoder(),
                                       problem=FunctionProblem(p.f, True))



# with open('parents.pickle', 'wb') as f:
#    pickle.dump(parents, f)

# with open('parents.pickle','rb') as f:
#     parents = pickle.load(f)

# print(parents)



# Evaluate initial population = calculate Fitness Function for each infividual in the initial population
print("INITIAL POPULATION")
parents = Individual.evaluate_population(parents)



# print initial, random population + Fitness Function for each individual
# ****
#util.print_population(parents, generation=0)

# generation_counter is an optional convenience for generation tracking
generation_counter = util.inc_generation(context=context)


#results = []
while generation_counter.generation() < 50:
    p.setStat()
    print("GENERATION ", generation_counter.generation()+1)
    #sequence of functions, the result of the first one will be the parameter of the next one, and so on
    offspring = pipe(parents,
                     #probe.print_individual(prefix='before tournament: '),
                     ops.tournament_selection,
                     #probe.print_individual(prefix='after tournament: \n'),
                     ops.clone,
                     #mutate_bitflip,
                     #probe.print_individual(prefix='before mutation: '),
                     #individual_mutate_randint,
                     #probe.print_individual(prefix='after mutation: '),
                     #probe.print_individual(prefix='before crossover: \n'),
                     ops.uniform_crossover(p_swap=0.2),
                     #probe.print_individual(prefix='after crossover: \n\n\n'),
                     ops.evaluate,
                     ops.pool(size=len(parents)))  # accumulate offspring

   
    parents = offspring
 

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

    # # ****
    # #print("best of generation:")#print best genome/individual (with best FF in the current pop)
    # #print(best)
    # #print(probe.best_of_gen(parents).fitness)#print best genome/individual FF
    # print()
    # #print(p.getStat()[key])

    # y = []
    # #prediction for each observation/record of the best individual ([best.genome[i]])[0]) = ith observation)
    # #len is about the number of genes in that genome (usually static)
    # for i in range(len(best.genome)):
    #     y.append(classifier.predict([best.genome[i]])[0])

    # #ch put the best individual in an array structure (1 el for each obs)
    # ch = np.array(best.genome)

    # print("MUTUAL INFO FOR EACH FEATURE - BEST INDIVIDUAL OF CURRENT POPULATION")
    # MI = []
    # for i in range(num_genes):
    #     print(features[i])
    #     mi = drv.information_mutual(ch[:,i],np.array(y),cartesian_product=True)
    #     print(mi)
    #     MI.append([features[i],mi])
        
    # result.append([p.getStat()[sorted_d[0][0]],p.getStat()[sorted_d[-1][0]],best,MI])

#SHAPLEY VALUES
#shap_values.shape[1]

expl = shap.LinearExplainer(logistic_regression,X_train)
shap_values = expl.shap_values(X_validation)

#BAR PLOT
shap.summary_plot(shap_values, X_validation, var, plot_type="bar")
#shap.plots.bar(shap_values[0],show=True)
#print(shap_values.base_values)

#SUMMARY PLOT
shap.summary_plot(shap_values,
                  X_validation,
                  feature_names=var)


print()

#sys.stdout.close()
