import sys
from leap_ec.algorithm import generational_ea
from leap_ec import representation, ops

from leap_ec.segmented_rep import initializers, decoders
from problems import fitness
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



n_Ind = 3000
var = ["age","gender","marital_status","education","lift_heavy_weight"]
var_2 = ["age","Gender:Female","Gender:Male", "marital_status","education","lift_heavy_weight"]
var_values = [[18,50],['F','M'],['single', 'married'],['primary', 'secondary', 'further', 'higher'],[5,50]]
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




featuresRange = [(18,50),(0,1),(0,1),(0,3),(5,50)]

#Feature vectors generation
for n in range (n_Ind):
    age = (random.randint(18,50))
    gender = (genders[random.randint(0,1)])
    marital_status = (marital_statuses[random.randint(0,1)])
    education = (educations[random.randint(0,3)])
    lift_heavy_weight = (random.randint(5,50))

    
    # if gender == 'F':
    #     #lift_heavy_weight.append(random.randint(5,15))
    #     lift_heavy_weight = (np.random.choice([random.randint(5,15),random.randint(20,50)],p=[0.8,0.2]))
    # else:
    #     #lift_heavy_weight.append(random.randint(20,50))
    #     lift_heavy_weight = (np.random.choice([random.randint(5,15),random.randint(20,50)],p=[0.2,0.8]))

    applicants[n].append(age)
    applicants[n].append(gender)
    applicants[n].append(marital_status) 
    applicants[n].append(education)
    applicants[n].append(lift_heavy_weight)

    age_vector.append(age)
    gender_vector.append(gender)
    marital_status_vector.append(marital_status)
    education_vector.append(education)
    lift_heavy_weight_vector.append(lift_heavy_weight)

    features = [age_vector,gender_vector,marital_status_vector,education_vector,lift_heavy_weight_vector]


#Ground Truth Generation Logic (with bias embedded in the generation of the dataset=all individuals with LHW < 20 kg = NO/ all men with LHW >= 20 kg = YES / Penalize Women that lift more thatn 20 kg)
gender = 1
heavy_weight = 4
age = 0
for n in range (n_Ind):
    #if  features[heavy_weight][n] >= 20:
    if features[gender][n] == 'F':
        r = random.randint(0, 1)
        if r > 0:
        #if features[heavy_weight][n] >= 40:
            if (features[age][n] < 40):
                hiring_vector.append('Yes')
            else:
                hiring_vector.append('No')
        else:
            hiring_vector.append('No')
    else:
        hiring_vector.append('Yes')
    #else:
        #hiring_vector.append('No')
        
         

#Dataset map

features = [age_vector,gender_vector,marital_status_vector,education_vector,lift_heavy_weight_vector,hiring_vector]

dataSet = pd.DataFrame(features).transpose()
dataSet.columns=['Age','Gender','Marital_Status','Education','Lift_Heavy_Weight','Hiring_Decision']
print("DATASET")
print(dataSet)

# Check for Null Data
dataSet.isnull().sum()

# Replace All Null Data in NaN
dataSet = dataSet.fillna(np.nan)

# Reformat Categorical Variables
dataSet['Gender']=dataSet['Gender'].map({'F': 1, 'M': 0})
dataSet['Marital_Status']=dataSet['Marital_Status'].map({'married': 0, 'single': 1})
dataSet['Education']=dataSet['Education'].map({'primary': 0, 'secondary': 1,'further':2,'higher':3})
dataSet['Hiring_Decision']=dataSet['Hiring_Decision'].map({'Yes': 1, 'No': 0})

# Confirm All Missing Data is Handled
dataSet.isnull().sum()

print(dataSet.head(4))

#print(dataSet)

#classification logic
def decide2(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if applicant[0][gender] == 1:
        r = random.randint(0, 1)
        if r > 0:
        #if features[heavy_weight][n] >= 40:
            if (applicant[0][age] < 40):
                return 1
            else:
               return 0
        else:
            return 0
    else:
        return 1

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

def decide(applicant):
    gender = 1
    heavy_weight = 4
    age = 0
    if applicant[0][gender] == 1:
        return 0
    else:
        return 1


#FITNESS FUNCTION

p = fitness(decide)

result = []

#number of individuals (matrixs)
pop_size = 100
#number of instances for each gene(variable) = number of records(observations) of the matrix
gene_size = 3000
#number of features
num_genes = len(applicants[0])

features = ["age","gender","marital_status","education","lift_heavy_weight"]
bounds = [(18,50),(0,1),(0,1),(0,3),(5,50)]



def init(length, seq_initializer):
    def create():
        return initializers.create_segmented_sequence(gene_size, create_int_vector(bounds))
    return create

#create initial rand population of pop_size individuals

parents = Individual.create_population(n=pop_size,
                                       initialize=init(gene_size, create_int_vector(bounds)),
                                       decoder=decoder.IdentityDecoder(),
                                       problem=FunctionProblem(p.f, True))


# Evaluate initial population = calculate Fitness Function for each infividual in the initial population
parents = Individual.evaluate_population(parents)

# print initial, random population + Fitness Function for each individual
# ****
#util.print_population(parents, generation=0)

# generation_counter is an optional convenience for generation tracking
generation_counter = util.inc_generation(context=context)

#results = []
while generation_counter.generation() < 30:
    p.setStat()
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
    
    count=0
    parents_pairs={}
    #parents_pairs collect position individual and fitness function (for each individual in the current population (in the current generation))
    for i in range(len(parents)):
        parents_pairs[i] = parents[i].fitness
    #sort individuals in the current population in an ascending order (by Fitness Function)
    import operator
    sorted_d = sorted(parents_pairs.items(), key=operator.itemgetter(1))

    #for key, value in sorted_d:
        #print("generation", context['leap']['generation'])
        #count=count+1
        #print("individual", count)
        #print(parents[key].genome)
        #print(parents[key].fitness)
        #print(p.getStat()[key])
    #for individual in parents:
    #    print("generation", context['leap']['generation'])
    #    print(p.getStat()[count])
    #    count=count+1
    #    print("individual", count)
        #print(individual.genome)
    #    print(individual.fitness)
    print("generation", context['leap']['generation'])

    #print worse and best (FF) individual in the population for each generation (showing FF + num meaningful features + name meaningful features + MI of feature)
    print("worst: ",p.getStat()[sorted_d[0][0]])
    print("best: ", p.getStat()[sorted_d[-1][0]])
    print()
    best = probe.best_of_gen(parents)

    # ****
    #print("best of generation:")#print best genome/individual (with best FF in the current pop)
    #print(best)
    #print(probe.best_of_gen(parents).fitness)#print best genome/individual FF
    print()
    #print(p.getStat()[key])

    y = []
    #prediction for each observation/record of the best individual ([best.genome[i]])[0]) = ith observation)
    #len is about the number of genes in that genome (usually static)
    for i in range(len(best.genome)):
        y.append(classifier.predict([best.genome[i]])[0])

    #ch put the best individual in an array structure (1 el for each obs)
    ch = np.array(best.genome)

    print("MUTUAL INFO FOR EACH FEATURE - BEST INDIVIDUAL OF CURRENT POPULATION")
    MI = []
    for i in range(num_genes):
        print(features[i])
        mi = drv.information_mutual(ch[:,i],np.array(y),cartesian_product=True)
        print(mi)
        MI.append([features[i],mi])
        
    result.append([p.getStat()[sorted_d[0][0]],p.getStat()[sorted_d[-1][0]],best,MI])

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