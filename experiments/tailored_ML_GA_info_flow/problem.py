#!/usr/bin/env python3
import numpy as np
import math
from pyitlib import discrete_random_variable as drv
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
#from dit.multivariate import entropy
#from pyinform.shannon import entropy
from functools import reduce
import itertools
import random

class fitness():

    def __init__(self, d):
        self.decide = d
        self.stat = []

    def f(self, chromosome):
        num_genes = len(chromosome[0]) # columns
        Var = ["age","gender","marital_status","education","lift_heavy_weight"]

        y = []
        #predict
        #len(chromosome) = (observations)
        for i in range(len(chromosome)-1): #to remove the last row that represents the selected partition bit
            y.append(self.decide([chromosome[i]]))

        selected_partition_features_index = chromosome[-1]
        
        #compute mutual array using y
        M = []
        features = []
        partiton_features = []
        selected_partition_dict = []

        vect_YZ = []
        vect_XYZ = []
        vect_XZ = []
        vect_Z = []

        ch = np.array(chromosome[0:-1])     

        selected_features_index = [x==1 for x in selected_partition_features_index]

        # selected_features_index = boolean selected features array
        selected_partition = ch[:,selected_features_index]

        num_partition_features = sum(selected_partition_features_index)

        #genes=[ch[:,0],ch[:,1],ch[:,2],ch[:,3],ch[:,4]]

        #features_dict = {"".join(Var[0]):genes[0]}
        #print(d)

        #for k in range(len(features)):
        #    features_dict["".join(Var[k])] = genes[k]

        # num_partition_features = random.randint(1,num_genes-1)
        # #num_partition_features = 3
        # partition_features_index = random.sample(range(0, num_genes), num_partition_features)          
        # selected_partition_features_index=np.zeros(num_genes)
        # selected_partition_features_index[partition_features_index]=1
        # ch = np.append(ch,[selected_partition_features_index],axis=0)


        # selected_partition_dict = {"".join(Var[sort_partition_features_index[0]]):selected_partition[:,0]}

        # n = 0

        
        # for k in sort_partition_features_index:
        #     selected_partition_dict["".join(Var[k])] = selected_partition[:,n]
        #     n = n + 1

        #MI

        #CHAIN RULE MULTIPLE VARIABLES -----

        a = drv.information_mutual(selected_partition[:,0],np.array(y),cartesian_product=True)
        M.append(a)

        if num_partition_features > 1:
            b = drv.information_mutual_conditional(selected_partition[:,1],np.array(y),selected_partition[:,0],cartesian_product=True)
            M.append(b)

            if num_partition_features > 2:
                i = 2

                for i in range(2,num_partition_features):
                    vect_YZ.append(np.array(y))
                    vect_XYZ.append(np.array(y))
                    vect_XZ.append(selected_partition[:,i])
                    vect_XYZ.append(selected_partition[:,i])

                    for j in range(i,-1,-1):
                        if (i != j):
                            vect_XZ.append(selected_partition[:,j])
                            vect_YZ.append(selected_partition[:,j])
                            vect_XYZ.append(selected_partition[:,j])
                            vect_Z.append(selected_partition[:,j])

                    H_XZ = drv.entropy_joint(vect_XZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                    H_Z = drv.entropy_joint(vect_Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                    H_XYZ = drv.entropy_joint(vect_XYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                    H_YZ = drv.entropy_joint(vect_YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                    c = (H_XZ-H_Z)-(H_XYZ-H_YZ)

                    del vect_XZ[:]
                    del vect_YZ[:]
                    del vect_XYZ[:]
                    del vect_Z[:]

                    i = i + 1

                    M.append(c)

        print()

        MI = sum(M)

        #-------

        #-----



        partition_index = [i for i, val in enumerate(selected_features_index) if val]
        Var_np = np.array(Var)
        partition_name = Var_np[partition_index]

        selected_partition_dict = {"".join(Var[partition_index[0]]):M[0]}
          
        for k in range(num_partition_features):
            selected_partition_dict["".join(Var[partition_index[k]])] = M[k]

        #OF = MI + (-num_partition_features+1)
        #OF = MI * math.exp(-num_partition_features)
        #OF = MI 
        #OF = MI/num_partition_features
        #OF = MI - ((num_partition_features-1)/(num_genes-1))

        self.stat.append([OF,partition_name,num_partition_features])


        print("FITNESS Individual")
        #print(ch)
        print(OF,num_partition_features)
        print(partition_name)
        print(selected_partition_dict)
  
        return OF

    def entropy_pers(*X):
            return  np.sum(-p * np.log2(p) if p > 0 else 0 for p in
                (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(X, classes))))
                    for classes in itertools.product(*[set(x) for x in X])))



    # def _vstack_pad(Arrays, fill_value):
    #     max_length = max([A.shape[-1] for A in Arrays])
    #     Arrays = [np.append(A, np.tile(fill_value,
    #                                 np.append(A.shape[:-1],
    #                                             max_length -
    #                                             A.shape[-1]).astype(int)))
    #             for A in Arrays]
    #     return np.vstack((Arrays))

    # def C_JE(X,Y,Z,Alphabet_X, Alphabet_Y, Alphabet_Z, cartesian_product):

    #     X = np.reshape(X, (-1, X.shape[-1]))
    #     Y = np.reshape(Y, (-1, Y.shape[-1]))
    #     Z = np.reshape(Z, (-1, Z.shape[-1]))
    #     Alphabet_X = np.reshape(Alphabet_X, (-1, Alphabet_X.shape[-1]))
    #     Alphabet_Y = np.reshape(Alphabet_Y, (-1, Alphabet_Y.shape[-1]))
    #     Alphabet_Z = np.reshape(Alphabet_Z, (-1, Alphabet_Z.shape[-1]))
 
    #     for i in range(X.shape[0]):
    #         I_ = (drv.entropy_joint(np.vstack((X[i], Z[i])), 2, -1,
    #                             'ML',
    #                             _vstack_pad((Alphabet_X[i],
    #                                         Alphabet_Z[i]),
    #                                         -1)) +
    #             drv.entropy_joint(np.vstack((Y[i], Z[i])), 2, -1,
    #                             'ML',
    #                             _vstack_pad((Alphabet_Y[i],
    #                                         Alphabet_Z[i]),
    #                                         -1)) -
    #             drv.entropy_joint(np.vstack((X[i], Y[i], Z[i])), 2, -1,
    #                             'ML',
    #                             _vstack_pad((Alphabet_X[i],
    #                                         Alphabet_Y[i],
    #                                         Alphabet_Z[i]), -1)) -
    #             drv.entropy_joint(Z[i], 2, -1, 'ML', Alphabet_Z[i]))
    #         I[i] = I_

    #     # Reverse re-shaping
    #     I = np.reshape(I, orig_shape_I)

    #     return I

    def getStat(self):
        return self.stat

    def setStat(self):
        self.stat = []