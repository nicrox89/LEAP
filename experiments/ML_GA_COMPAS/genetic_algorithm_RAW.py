import sys
from leap_ec.algorithm import generational_ea
from leap_ec import representation, ops

from leap_ec.segmented_rep import initializers, decoders
#from problems import fitness
#from problem import fitness
from problemMultiCMI2 import fitness
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
from datetime import datetime

from leap_ec.problem import FunctionProblem

from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from dateutil.relativedelta import relativedelta
import datetime


import shap
import pickle




n_Ind = 3000
var=["gender_factor", "age_factor", "race_factor", 
    "priors_count", "crime_factor"]
#var_values = [['Female','Male'],[21, 69],['African-American', 'Asian', 'Hispanic', 'Native-American', 'Other', 'Caucasian'],
#    [0,36], ['F','M']]
var_values = [[0,1],[21, 69],[0,1,2,3,4,5,6],
    [0,36], [0,1]]
gender_values = [[1,0]]

rows, cols = (n_Ind, var) 
applicants = [[]for j in range(rows)] 
applicants_test = [[]for j in range(rows)] 

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




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
#import QII
import shap
import random
from shap import plots
#import xgboost

#now = datetime.now().strftime("%Y%m%d_%H%M%S")
#sys.stdout = open('experiments/tailored_ML_GA_Adult_rf/adult_GA_output'+now+'.out', 'w')



from sklearn.model_selection import RepeatedStratifiedKFold
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores



sns.set(style='white', context='notebook', palette='deep')

dataset = pd.read_csv("experiments/data/compas-scores-raw.csv")

index_female = []

# Check for Null Data
dataset.isnull().sum()

# Replace All Null Data in NaN
dataset = dataset.fillna(np.nan)

# Peek at data
dataset.head(2)

x=[]
y=[]
age=[]
time_difference=[]
difference_in_years=[]

#pd.to_datetime(dataset['DateOfBirth'], format='%m/%d/%y', errors='ignore')
#dataset['DateOfBirth'].astype('datetime64[ns]')

DOB_list = dataset['DateOfBirth'].values.tolist()
DOB_list2 = pd.to_datetime(DOB_list, format='%m/%d/%y')

DS_list = dataset['Screening_Date'].values.tolist()
DS_list2 = pd.to_datetime(DS_list, format='%m/%d/%y %H:%M')

for i in range(len(DOB_list2)):
    year_B=int(DOB_list2[i].strftime('%Y'))
    year_S=int(DS_list2[i].strftime('%Y'))
    age.append(abs(year_S-year_B))

# for i in range (len(DOB_list2)):
#     x[i] = datetime.datetime(DOB_list2[i]) 
#     y[i] = datetime.datetime(DOB_list2[i]) 
#     time_difference[i] = relativedelta(y[i], x[i])
#     difference_in_years[i] = time_difference[i].years

# Drop the data we don't want to use
dataset=dataset[['Sex_Code_Text','Ethnic_Code_Text','Language','LegalStatus','CustodyStatus','MaritalStatus','RecSupervisionLevel','ScoreText']]
dataset['Age']=age
print(dataset.head())

# Drop the NaN rows now 
dataset.dropna(how='any',inplace=True)

# Confirm All Missing Data is Handled
dataset.isnull().sum()
# Reformat Column We Are Predicting
#dataset[dataset['ScoreText'].str.contains('N/A')==False]
dataset['ScoreText']=dataset['ScoreText'].map({'Low': 0, 'Medium': 1, 'High': 2}).astype(int)
dataset.head(4)

var=['Sex_Code_Text','Ethnic_Code_Text','Language','LegalStatus','CustodyStatus','MaritalStatus','RecSupervisionLevel']



# Create Married Column - Binary Yes(1) or No(0)
#dataset["marital.status"] = dataset["marital.status"].replace(['Never-married','Divorced','Separated','Widowed'], 'Single')
#dataset["marital.status"] = dataset["marital.status"].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
#dataset["marital.status"] = dataset["marital.status"].map({"Married":1, "Single":0})
#dataset["marital.status"] = dataset["marital.status"].astype(int)

# Convert Sex value to 0 and 1
dataset["Sex_Code_Text"] = dataset["Sex_Code_Text"].map({"Male":0, "Female":1})

# Convert Race
dataset["Ethnic_Code_Text"] = dataset["Ethnic_Code_Text"].map({'African-Am': 0,'African-American': 0, 'Arabic': 1, 'Asian': 2,'Caucasian': 3, 'Hispanic': 4, 'Native American': 5, 'Oriental': 6, 'Other': 7}).astype(int)

dataset['Language'] = dataset['Language'].map({'English': 0, 'Spanish': 1})

dataset['LegalStatus'] = dataset['LegalStatus'].map({'Conditional Release': 0, 'Deferred Sentencing': 1, 'Other':2, 'Parole Violator':3, 'Post Sentence':4, 'Pretrial': 5, 'Probation Violator':6})

dataset['CustodyStatus'] = dataset['CustodyStatus'].map({'Jail Inmate': 0, 'Parole': 1, 'Pretrial Defendant':2, 'Prison Inmate':3, 'Probation':4, 'Residential Program':5})

dataset['MaritalStatus'] = dataset['MaritalStatus'].map({'Divorced': 0, 'Married': 1, 'Separated':2, 'Significant Other':3, 'Single': 4, 'Unknown':5, 'Widowed':6})


print(dataset.head())

feature_names=['Sex_Code_Text','Ethnic_Code_Text','Language','LegalStatus','CustodyStatus','MaritalStatus','RecSupervisionLevel','Age','ScoreText']

# Split-out Validation Dataset and Create Test Variables
array = dataset.values
X = array[:,0:7]
Y = array[:,7]
print('Split Data: X')
print(X)
print('Split Data: Y')
print(Y)
validation_size = 0.20
seed = 7
num_folds = 10
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = train_test_split(X,Y,
    test_size=validation_size,random_state=seed)

results = []
names = []

kfold = KFold(n_splits=10)#, random_state=seed)
#cv_results = cross_val_score(LinearRegression(), X_train, Y_train, cv=kfold, scoring='accuracy')
#cv_results = cross_val_score(RandomForestRegressor(), X_train, Y_train, cv=kfold, scoring='accuracy')
cv_results = cross_val_score(LogisticRegression(), X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % ("LR", cv_results.mean(), cv_results.std())
print(msg)

# Finalize Model

#Linear Regression 
#linear_regression = LinearRegression()
#linear_regression.fit(X_train, Y_train)
#predictions = linear_regression.predict(X_validation)

# #Random Forest
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_validation = sc.transform(X_validation)
# random_forest_regression = RandomForestRegressor(n_estimators=20, random_state=0)
# random_forest_regression.fit(X_train, Y_train)
# predictions = random_forest_regression.predict(X_validation)

# Build the model with the random forest regression algorithm:
# model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
# model.fit(X_train, Y_train)

#RF --- 0.65
# from sklearn.ensemble import RandomForestClassifier
# from numpy import mean
# from numpy import std
# model = RandomForestClassifier(n_estimators=100)
# model.fit(X_train, Y_train)
# predictions = model.predict(X_validation)
# scores = evaluate_model(X_train, Y_train, model)
# print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
# ----

#DT 0.63
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, Y_train)
dtree_predictions = dtree_model.predict(X_validation)
 
# # creating a confusion matrix
cm = confusion_matrix(Y_validation, dtree_predictions)
print(cm)

# ----

#SVM  0.63
# from sklearn.svm import SVC
# svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, Y_train)
# svm_predictions = svm_model_linear.predict(X_validation)
 
# # model accuracy for X_test 
# accuracy = svm_model_linear.score(X_validation, Y_validation)
 
# # creating a confusion matrix
# cm = confusion_matrix(Y_validation, svm_predictions)
# print(cm)


#KNN  0.61
# from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(n_neighbors = 7).fit(X_train, Y_train)
 
# # accuracy on X_test
# accuracy = knn.score(X_validation, Y_validation)
# print(accuracy)
 
# # creating a confusion matrix
# knn_predictions = knn.predict(X_validation)
# cm = confusion_matrix(Y_validation, knn_predictions)
# print (cm)


#NAIVE  0.62
# from sklearn.naive_bayes import GaussianNB
# gnb = GaussianNB().fit(X_train, Y_train)
# gnb_predictions = gnb.predict(X_validation)
 
# # accuracy on X_test
# accuracy = gnb.score(X_validation, Y_validation)
# print(accuracy)
 
# # creating a confusion matrix
# cm = confusion_matrix(Y_validation, gnb_predictions)
# print(cm)

#Logistic Regression 0.58
# logistic_regression = LogisticRegression()
# logistic_regression.fit(X_train, Y_train)
# predictions = logistic_regression.predict(X_validation)
# print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))
# print(confusion_matrix(Y_validation, predictions))
# print(classification_report(Y_validation, predictions))


# Prediction
#'sex','age','race','priors_count','c_charge_degree'
#new_record = [[1,69,4,0,1]]
#print(logistic_regression.predict(new_record)[0])

#print(model.predict(new_record)[0])
#print(dtree_model.predict(new_record)[0])

#print(svm_model_linear.predict(new_record)[0])

#print(knn.predict(new_record)[0])

#print(gnb.predict(new_record)[0])



#FITNESS FUNCTION

#classifier = gnb

#classifier = knn
#classifier = svm_model_linear
classifier = dtree_model
#classifier = model
#classifier = logistic_regression
p = fitness(classifier)

result = []

#number of individuals (matrixs)
pop_size = 100
#number of instances for each gene(variable) = number of records(observations) of the matrix
gene_size = 1000
#number of features
num_genes = len(var)

#feature_names = ["age","workclass","education","education.num","marital.status","occupation","relationship","race","sex","capital.gain","capital.loss","hours.per.week"]
#bounds = [(min(dataset["age"]),max(dataset["age"])),(min(dataset["workclass"]),max(dataset["workclass"])),(min(dataset["education"]),max(dataset["education"])),(min(dataset["education.num"]),max(dataset["education.num"])),(min(dataset["marital.status"]),max(dataset["marital.status"])),(min(dataset["occupation"]),max(dataset["occupation"])),(min(dataset["relationship"]),max(dataset["relationship"])),(min(dataset["race"]),max(dataset["race"])),(min(dataset["sex"]),max(dataset["sex"])),(min(dataset["capital.gain"]),max(dataset["capital.gain"])),(min(dataset["capital.loss"]),max(dataset["capital.loss"])),(min(dataset["hours.per.week"]),max(dataset["hours.per.week"]))]

feature_names=['Sex_Code_Text','Ethnic_Code_Text','Language','LegalStatus','CustodyStatus','MaritalStatus','RecSupervisionLevel','Age']
bounds = [[0,1],[0,7],[0,6],[0,1],[0,6],[0,5],[0,6],[5,55]]

def set_Partition():
        #num_partition_features = 2
        minPartition_size = 1
        #maxPartition_size = 2
        maxPartition_size = num_genes-1
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



print()

#sys.stdout.close()

