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
import copy
import map_and_encode as mp

class fitness():

    def __init__(self, d, vars, bounds, splits):
        self.decide = d
        self.stat = []
        self.Var = vars
        self.bounds = bounds
        self.splits = splits

    def f(self, chromosome):
        
        #Var = ["age","gender","marital_status","education","lift_heavy_weight"]
        y = []
        single_contribution = []

        #predict
        #len(chromosome) = (observations)
        for i in range(len(chromosome)-1): #to remove the last row that represents the selected partition bit
            y.append(self.decide.predict([chromosome[i]])[0].astype(np.int32))

        selected_partition_features_index = chromosome[-1]
        num_genes = len(selected_partition_features_index) # columns

        #remove last row (partition)
        ch = np.array(chromosome[0:-1])   
        #ch = mp.encode_columns_splits(ch, self.bounds, self.splits)[0]  

        # ch[:,0] = [self.binary(x,30) for x in ch[:,0]]
        # ch[:,3] = [self.binary(x,1) for x in ch[:,3]]
        # ch[:,4] = [self.binary(x,30) for x in ch[:,4]]

        selected_features_index = [x==1 for x in selected_partition_features_index]

        # selected_features_index = boolean selected features array
        selected_partition = ch[:,selected_features_index]

        num_partition_features = sum(selected_partition_features_index)

        partition_index = [i for i, val in enumerate(selected_features_index) if val]
        Var_np = np.array(self.Var)
        partition_name = Var_np[partition_index]

        
        #self.chain_rule_H(ch, y, selected_partition, num_partition_features)

        multi_contribution_chain = self.chain_rule(ch, y, selected_partition, num_partition_features)

        multi_contribution_CMI = self.multivariate_CMI(ch, y, selected_features_index, selected_partition, num_partition_features)
        #multi_contribution_CMI_TEST = self.multivariate_CMI_TEST(ch, y, selected_features_index, selected_partition, num_partition_features)
        #multi_contribution_CMI_TEST_CHAIN = self.multivariate_CMI_TEST_CHAIN(ch, y, selected_features_index, selected_partition, num_partition_features)
        #multi_contribution_CMI_TEST_CHAIN2 = self.multivariate_CMI_TEST_CHAIN2(ch, y, selected_features_index, selected_partition, num_partition_features)

        single_contribution_CMI, single_contribution_MI = self.single_CMI(partition_name, self.Var, num_genes, ch, y, partition_index, selected_features_index, selected_partition, num_partition_features)
        single_contribution_CMI_all, single_contribution_MI_all = self.single_CMI_all(partition_name, self.Var, num_genes, ch, y, partition_index, selected_features_index, selected_partition, num_partition_features)

        #genes=[ch[:,0],ch[:,1],ch[:,2],ch[:,3],ch[:,4]]

        #features_dict = {"".join(self.Var[0]):genes[0]}
        #print(d)

        #for k in range(len(features)):
        #    features_dict["".join(self.Var[k])] = genes[k]

        # num_partition_features = random.randint(1,num_genes-1)
        # #num_partition_features = 3
        # partition_features_index = random.sample(range(0, num_genes), num_partition_features)          
        # selected_partition_features_index=np.zeros(num_genes)
        # selected_partition_features_index[partition_features_index]=1
        # ch = np.append(ch,[selected_partition_features_index],axis=0)

        # selected_partition_dict = {"".join(self.Var[sort_partition_features_index[0]]):selected_partition[:,0]}
        # n = 0
        
        # for k in sort_partition_features_index:
        #     selected_partition_dict["".join(self.Var[k])] = selected_partition[:,n]
        #     n = n + 1


        #OF = multi_contribution_chain
        #OF = multi_contribution_CMI_TEST
        #OF = multi_contribution_chain + (-num_partition_features+1)
        #OF = multi_contribution_CMI_TEST  + (-num_partition_features)
        #OF = multi_contribution_chain * math.exp(-num_partition_features)
        #OF = multi_contribution_CMI_TEST * math.exp(-num_partition_features)
        #OF = MI 
        #OF = multi_contribution_CMI/num_partition_features


        #OF = ((sum(multi_contribution_CMI*list(single_contribution_CMI.values())[i] for i in range (num_partition_features)))/num_partition_features)
        #OF = multi_contribution_CMI+math.exp(-(num_partition_features))

        #OF = ((sum(multi_contribution_CMI*list(single_contribution_MI.values())[i] for i in range (num_partition_features)))/num_partition_features)
        #OF = ((sum(multi_contribution_CMI*list(single_contribution_CMI.values())[i] for i in range (num_partition_features)))/num_partition_features)+(sum((np.array(list(single_contribution_MI.values())))*np.array(list(single_contribution_CMI.values()))))
        #OF = ((sum(multi_contribution_CMI*list(single_contribution_CMI.values())[i] for i in range (num_partition_features)))/num_partition_features)*math.exp(-sum((np.array(list(single_contribution_MI.values())))))
        bool_vect_del = [False for j in range(num_partition_features)]
        num_partition_features_test = num_partition_features - 1
        test = []

        if num_partition_features != 1:

            for i in range (num_partition_features):
                bool_vect_del = [False for j in range(num_partition_features)]
                bool_vect_del[i] = True
                selected_features_index_test = copy.copy(selected_features_index)
                selected_features_index_test[partition_index[i]] = False
                selected_partition_test = np.delete(selected_partition, bool_vect_del, axis=1)
                multi_contribution_CMI_test = self.multivariate_CMI(ch, y, selected_features_index_test, selected_partition_test, num_partition_features_test)
                test.append(multi_contribution_CMI_test-multi_contribution_CMI)
        
        delta_ro = abs(np.std(list(single_contribution_CMI.values())) - (np.std(list(single_contribution_MI.values()))))
        #OF = ((sum(multi_contribution_CMI*(list(single_contribution_CMI.values())[i]) for i in range (num_partition_features)))/num_partition_features)*math.exp(-delta_ro)

        #OF = ((sum(multi_contribution_CMI*(list(single_contribution_CMI.values())[i]) for i in range (num_partition_features)))/num_partition_features)

        if num_partition_features > 1:
            OF=multi_contribution_CMI + ((abs(np.mean(sum(test))))/num_partition_features)
            #OF=multi_contribution_CMI * (1-math.exp(-abs(np.mean(sum(test))))/num_partition_features)
        else:
            OF = multi_contribution_CMI * 2
        
        #OF = ((sum(multi_contribution_CMI*(list(single_contribution_CMI.values())[i]/list(single_contribution_MI.values())[i]) for i in range (num_partition_features)))/num_partition_features)*math.exp(-sum((np.array(list(single_contribution_MI.values())))))

        #OF = multi_contribution_CMI + (num_genes-num_partition_features)
        #OF = multi_contribution_chain/num_partition_features
        #OF = multi_contribution_chain - ((num_partition_features-1)/(num_genes-1))
        self.stat.append([OF,partition_name,num_partition_features])

        print("FITNESS Individual")
        #print("MULTI CMI,MI+(num_genes-num_partition_features)")
        #print(ch)
        print(OF,num_partition_features)
        print(partition_name)
        print("PARTITION JOINT CMI")
        print(multi_contribution_CMI)
        print("COMPARE WITH CHAIN RULE")
        print(multi_contribution_chain)
        print("SINGLE CONTRIBUTION CMI")
        print(single_contribution_CMI)
        print("SINGLE CONTRIBUTION MI")
        print(single_contribution_MI)
        print("SINGLE CONTRIBUTION CMI ALL")
        print(single_contribution_CMI_all)
        # print("SINGLE CONTRIBUTION MI ALL")
        # print(single_contribution_MI_all)s
        print()

  
        return OF

    
    
    #FORMULA MULTIVARIATE CONDITIONAL MUTUAL INFORMATION

    def multivariate_CMI(self, ch, y, selected_features_index, selected_partition, num_partition_features):

        vect_YZ = []
        vect_XYZ = []
        vect_XZ = []
        vect_Z = []

        vect_XY = []

        XX = []
        vect_xYZ = []
        
        # demonstration comparing with simple CMI (X;Y|Z) -----
        # Z = np.delete(ch, [False,True,False,False,True], axis=1)
        # vect_XYZ.append(np.array(ch[:,1]))
        # vect_XZ.append(np.array(ch[:,1]))
        # vect_XYZ.append(np.array(ch[:,4]))
        # vect_XZ.append(np.array(ch[:,4]))
        
        
        # demonstration comparing with CHAIN RULE (X1,X2,X2;Y) no Z -----
        # Z = np.delete(ch, [True,True,True,True,True], axis=1)
        # vect_XYZ.append(np.array(ch[:,0]))
        # vect_XYZ.append(np.array(ch[:,1]))
        # vect_XYZ.append(np.array(ch[:,2]))
        # vect_XYZ.append(np.array(ch[:,3]))
        # vect_XYZ.append(np.array(ch[:,4]))
        # vect_XZ.append(np.array(ch[:,0]))
        # vect_XZ.append(np.array(ch[:,1]))
        # vect_XZ.append(np.array(ch[:,2]))
        # vect_XZ.append(np.array(ch[:,3]))
        # vect_XZ.append(np.array(ch[:,4]))


        # demonstration single contribution multi cond (X;Y|Z,W,T,S) -----
        # Z = np.delete(ch, [False,True,False,False,False], axis=1)
        # vect_XYZ.append(np.array(ch[:,1]))
        # vect_XZ.append(np.array(ch[:,1]))

        # vect_XY.append(np.array(ch[:,1])) #
        
        Z = np.delete(ch, selected_features_index, axis=1)     #

        for i in range(num_partition_features):                #
            vect_XYZ.append(np.array(selected_partition[:,i])) #
            vect_XZ.append(np.array(selected_partition[:,i]))  #

        vect_XYZ.append(np.array(y))
        vect_YZ.append(np.array(y))

        vect_xYZ.append(np.array(y))
        vect_XY.append(np.array(y)) #
        
        for j in range(len(Z[0])):
            vect_XYZ.append(np.array(Z[:,j]))
            vect_XZ.append(np.array(Z[:,j]))
            vect_YZ.append(np.array(Z[:,j]))
            vect_Z.append(np.array(Z[:,j]))

            vect_xYZ.append(np.array(Z[:,j]))
            

        H_XYZ = drv.entropy_joint(vect_XYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        if len(vect_Z)!=0:
            H_Z = drv.entropy_joint(vect_Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_YZ = drv.entropy_joint(vect_YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_XZ = drv.entropy_joint(vect_XZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        #H_XY = drv.entropy_joint(vect_XY, base=2, fill_value=-1, estimator='ML', keep_dims=False) #

        # vect_XYZ_lst = [list(vect_XYZ[i]) for i in range(len(vect_XYZ))]
        # H_XYZ = self.entropy_pers(*vect_XYZ_lst)
        # H_XYZ_test = drv.entropy_joint(vect_XYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)

        # H_Z = self.entropy_pers(vect_Z)
        # H_YZ = self.entropy_pers(vect_YZ)
        # H_XZ = self.entropy_pers(vect_XZ)

        for k in range(len(selected_partition[0])):
            for kk in range(len(selected_partition[0])):
                if (k != kk):
                    vect_xYZ.append(np.array(selected_partition[:,kk]))
            H_xYZ = drv.entropy_joint(vect_xYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            XX.append(H_xYZ)

            del vect_xYZ[:]

        #cc = H_XYZ - H_Z - (H_XYZ - XX[0]) - (H_XYZ - XX[1]) ..... - YY

        #demonstration comparting with simple CMI (X;Y|Z)

        #CMI = drv.information_mutual_conditional(ch[:,1],np.array(y),ch[:,0],cartesian_product=True)

        X = H_XYZ - H_YZ
        YY = H_XYZ - H_XZ
        # X = H_XYZ - H_Z - H_XY
        # YY = H_XYZ - H_Z - H_XY

        if len(vect_Z)!=0:
            c = H_XYZ - H_Z - X - YY 
        else:
            c = H_XYZ - X - YY


        del vect_YZ[:]
        del vect_XYZ[:]
        del vect_XZ[:]
        del vect_Z[:]


        return c

    
    def multivariate_CMI_TEST(self, ch, y, selected_features_index, selected_partition, num_partition_features):

        vect_YZ = []
        vect_XYZ = []
        vect_XZ = []
        vect_Z = []

        vect_XY = []

        XX = []
        vect_xYZ = []
        
        # # demonstration comparing with simple CMI (X;Y|Z) -----
        # Z = np.delete(ch, [False,True,False,False,True], axis=1)
        # vect_XYZ.append(np.array(ch[:,1]))
        # vect_XZ.append(np.array(ch[:,1]))
        

        # demonstration comparing with CHAIN RULE (X1,X2,X2;Y) no Z -----
        # Z = np.delete(ch, [True,True,True,True,True], axis=1)
        # vect_XYZ.append(np.array(ch[:,0]))
        # vect_XYZ.append(np.array(ch[:,1]))
        # vect_XYZ.append(np.array(ch[:,2]))
        # vect_XYZ.append(np.array(ch[:,3]))
        # vect_XYZ.append(np.array(ch[:,4]))
        # vect_XZ.append(np.array(ch[:,0]))
        # vect_XZ.append(np.array(ch[:,1]))
        # vect_XZ.append(np.array(ch[:,2]))
        # vect_XZ.append(np.array(ch[:,3]))
        # vect_XZ.append(np.array(ch[:,4]))


        # demonstration single contribution multi cond (X;Y|Z,W,T,S) -----
        Z = np.delete(ch, [False,True,False,False,True], axis=1)
        vect_XYZ.append(np.array(ch[:,1]))
        vect_XZ.append(np.array(ch[:,1]))
        vect_XYZ.append(np.array(ch[:,4]))
        vect_XZ.append(np.array(ch[:,4]))

        # vect_XY.append(np.array(ch[:,1])) #
        
        # Z = np.delete(ch, selected_features_index, axis=1)     #

        # for i in range(num_partition_features):                #
        #     vect_XYZ.append(np.array(selected_partition[:,i])) #
        #     vect_XZ.append(np.array(selected_partition[:,i]))  #

        vect_XYZ.append(np.array(y))
        vect_YZ.append(np.array(y))
        
        for j in range(len(Z[0])):
            vect_XYZ.append(np.array(Z[:,j]))
            vect_XZ.append(np.array(Z[:,j]))
            vect_YZ.append(np.array(Z[:,j]))
            vect_Z.append(np.array(Z[:,j]))
            

        H_XYZ = drv.entropy_joint(vect_XYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        if len(vect_Z)!=0:
            H_Z = drv.entropy_joint(vect_Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_YZ = drv.entropy_joint(vect_YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_XZ = drv.entropy_joint(vect_XZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        #H_XY = drv.entropy_joint(vect_XY, base=2, fill_value=-1, estimator='ML', keep_dims=False) #

        # vect_XYZ_lst = [list(vect_XYZ[i]) for i in range(len(vect_XYZ))]
        # H_XYZ = self.entropy_pers(*vect_XYZ_lst)
        # H_XYZ_test = drv.entropy_joint(vect_XYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)

        # H_Z = self.entropy_pers(vect_Z)
        # H_YZ = self.entropy_pers(vect_YZ)
        # H_XZ = self.entropy_pers(vect_XZ)

        #cc = H_XYZ - H_Z - (H_XYZ - XX[0]) - (H_XYZ - XX[1]) ..... - YY

        #demonstration comparting with simple CMI (X;Y|Z)

        #CMI = drv.information_mutual_conditional(ch[:,1],np.array(y),ch[:,0],cartesian_product=True)

        if len(vect_Z)!=0:
            c = (H_XZ - H_Z) - (H_XYZ - H_YZ)
        else:
            c = (H_XZ) - (H_XYZ - H_YZ)


        del vect_YZ[:]
        del vect_XYZ[:]
        del vect_XZ[:]
        del vect_Z[:]

        print("CONTRIBUTION PARTITION ", c)
        return c

    def multivariate_CMI_TEST_CHAIN(self, ch, y, selected_features_index, selected_partition, num_partition_features):

        M = []

        vectYZ = []
        vectXYZ = []
        vectXZ = []
        vectZ = []

        #demonstration with Multivariate CMI
        #selected_partition = ch
        #num_partition_features = 2

        vectXZ.append(selected_partition[:,0])
        vectXYZ.append(selected_partition[:,0])

        vectXYZ.append(np.array(y))
        vectYZ.append(np.array(y))

        Z = np.delete(ch, selected_features_index, axis=1)     #

        for j in range(len(Z[0])):
            vectXZ.append(np.array(Z[:,j]))
            vectXYZ.append(np.array(Z[:,j]))
            vectYZ.append(np.array(Z[:,j]))
            vectZ.append(np.array(Z[:,j]))

        _H_XZ = drv.entropy_joint(vectXZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        _H_Z = drv.entropy_joint(vectZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        _H_XYZ = drv.entropy_joint(vectXYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        _H_YZ = drv.entropy_joint(vectYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        

        a = (_H_XZ-_H_Z)-(_H_XYZ-_H_YZ)
        M.append(a)

        if num_partition_features > 1:

            #i = 1

            for i in range(1,num_partition_features):

                vectXZ[0] = selected_partition[:,i]
                vectXYZ[0] = selected_partition[:,i]

                vectXZ.append(selected_partition[:,i-1])
                vectXYZ.append(selected_partition[:,i-1])
                vectYZ.append(selected_partition[:,i-1])
                vectZ.append(selected_partition[:,i-1])

                _H_XZ = drv.entropy_joint(vectXZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                _H_Z = drv.entropy_joint(vectZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                _H_XYZ = drv.entropy_joint(vectXYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                _H_YZ = drv.entropy_joint(vectYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                
                b = (_H_XZ-_H_Z)-(_H_XYZ-_H_YZ)
                M.append(b)

        MI = sum(M)

        del vectXZ[:]
        del vectYZ[:]
        del vectXYZ[:]
        del vectZ[:]

        return MI

#----------------
        # vect_X2YZ = []
        # vect_X1YZ = []
        # vect_X1Z = []
        # vect_Z = []
        # vect_X1X2Z =[]
        # vect_X1X2YZ = []
        # vect_YZ = []


        # XX = []
        # vect_xYZ = []
        
        # # demonstration comparing with simple CMI (X;Y|Z) -----
        # #Z = np.delete(ch, [False,True,False,False,True], axis=1)
        # Z = np.delete(ch, selected_features_index, axis=1)     #

        
        # vect_X1Z.append(np.array(ch[:,1]))
        # vect_X1X2Z.append(np.array(ch[:,1]))
        # vect_X1X2YZ.append(np.array(ch[:,1]))
        # vect_X1YZ.append(np.array(ch[:,1]))
        # vect_X1X2YZ.append(np.array(ch[:,4]))
        # vect_X1X2Z.append(np.array(ch[:,4]))
        

        # # demonstration comparing with CHAIN RULE (X1,X2,X2;Y) no Z -----
        # # Z = np.delete(ch, [True,True,True,True,True], axis=1)
        # # vect_XYZ.append(np.array(ch[:,0]))
        # # vect_XYZ.append(np.array(ch[:,1]))
        # # vect_XYZ.append(np.array(ch[:,2]))
        # # vect_XYZ.append(np.array(ch[:,3]))
        # # vect_XYZ.append(np.array(ch[:,4]))
        # # vect_XZ.append(np.array(ch[:,0]))
        # # vect_XZ.append(np.array(ch[:,1]))
        # # vect_XZ.append(np.array(ch[:,2]))
        # # vect_XZ.append(np.array(ch[:,3]))
        # # vect_XZ.append(np.array(ch[:,4]))


        # # demonstration single contribution multi cond (X;Y|Z,W,T,S) -----
        # # Z = np.delete(ch, [False,True,False,False,False], axis=1)
        # # vect_XYZ.append(np.array(ch[:,1]))
        # # vect_XZ.append(np.array(ch[:,1]))

        # # vect_XY.append(np.array(ch[:,1])) #
        
        # # Z = np.delete(ch, selected_features_index, axis=1)     #

        # # for i in range(num_partition_features):                #
        # #     vect_XYZ.append(np.array(selected_partition[:,i])) #
        # #     vect_XZ.append(np.array(selected_partition[:,i]))  #

        # vect_X1YZ.append(np.array(y))
        # vect_X1X2YZ.append(np.array(y))
        # vect_YZ.append(np.array(y))

        
        # for j in range(len(Z[0])):
        #     vect_X1YZ.append(np.array(Z[:,j]))
        #     vect_X1Z.append(np.array(Z[:,j]))
        #     vect_X1X2Z.append(np.array(Z[:,j]))
        #     vect_X1X2YZ.append(np.array(Z[:,j]))
        #     vect_Z.append(np.array(Z[:,j]))
        #     vect_YZ.append(np.array(Z[:,j]))

            

        # H_X1YZ = drv.entropy_joint(vect_X1YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        # if len(vect_Z)!=0:
        #     H_Z = drv.entropy_joint(vect_Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        # H_X1X2YZ = drv.entropy_joint(vect_X1X2YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        # H_YZ = drv.entropy_joint(vect_YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        # H_X1Z = drv.entropy_joint(vect_X1Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        # H_X1X2Z = drv.entropy_joint(vect_X1X2Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
       
        # #H_XY = drv.entropy_joint(vect_XY, base=2, fill_value=-1, estimator='ML', keep_dims=False) #

        # # vect_XYZ_lst = [list(vect_XYZ[i]) for i in range(len(vect_XYZ))]
        # # H_XYZ = self.entropy_pers(*vect_XYZ_lst)
        # # H_XYZ_test = drv.entropy_joint(vect_XYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)

        # # H_Z = self.entropy_pers(vect_Z)
        # # H_YZ = self.entropy_pers(vect_YZ)
        # # H_XZ = self.entropy_pers(vect_XZ)

        # #cc = H_XYZ - H_Z - (H_XYZ - XX[0]) - (H_XYZ - XX[1]) ..... - YY

        # #demonstration comparting with simple CMI (X;Y|Z)

        # #CMI = drv.information_mutual_conditional(ch[:,1],np.array(y),ch[:,0],cartesian_product=True)

        # if len(vect_Z)!=0:
        #     cc = ((H_X1Z - H_Z) + (H_X1X2Z - H_X1Z))-((H_X1YZ - H_YZ) + (H_X1X2YZ - H_X1YZ))
        # #else:
        #     #c = (H_X1YZ) - (H_X2YZ - H_X1Z)


        # del vect_X1YZ[:]
        # del vect_X2YZ[:]
        # del vect_X1Z[:]
        # del vect_Z[:]

        # print("CMI PARTITION CHAIN", cc)
        # return cc

    def multivariate_CMI_TEST_CHAIN2(self, ch, y, selected_features_index, selected_partition, num_partition_features):

        vect_X2YZ = []
        vect_X1YZ = []
        vect_X1Z = []
        vect_Z = []
        vect_X1X2Z =[]
        vect_X1X2YZ = []
        vect_YZ = []


        XX = []
        vect_xYZ = []
        
        # demonstration comparing with simple CMI (X;Y|Z) -----
        Z = np.delete(ch, [False,True,False,False,False], axis=1)
        vect_X1Z.append(np.array(ch[:,1]))
        vect_X1X2Z.append(np.array(ch[:,1]))
        vect_X1X2YZ.append(np.array(ch[:,1]))
        vect_X1YZ.append(np.array(ch[:,1]))

        
        #Z = np.delete(ch, selected_features_index, axis=1)     #

        
        # vect_X1Z.append(np.array(ch[:,1]))
        # vect_X1X2Z.append(np.array(ch[:,1]))
        # vect_X1X2YZ.append(np.array(ch[:,1]))
        # vect_X1YZ.append(np.array(ch[:,1]))
        # vect_X1X2YZ.append(np.array(ch[:,4]))
        # vect_X1X2Z.append(np.array(ch[:,4]))
        

        # demonstration comparing with CHAIN RULE (X1,X2,X2;Y) no Z -----
        # Z = np.delete(ch, [True,True,True,True,True], axis=1)
        # vect_XYZ.append(np.array(ch[:,0]))
        # vect_XYZ.append(np.array(ch[:,1]))
        # vect_XYZ.append(np.array(ch[:,2]))
        # vect_XYZ.append(np.array(ch[:,3]))
        # vect_XYZ.append(np.array(ch[:,4]))
        # vect_XZ.append(np.array(ch[:,0]))
        # vect_XZ.append(np.array(ch[:,1]))
        # vect_XZ.append(np.array(ch[:,2]))
        # vect_XZ.append(np.array(ch[:,3]))
        # vect_XZ.append(np.array(ch[:,4]))


        # demonstration single contribution multi cond (X;Y|Z,W,T,S) -----
        # Z = np.delete(ch, [False,True,False,False,False], axis=1)
        # vect_XYZ.append(np.array(ch[:,1]))
        # vect_XZ.append(np.array(ch[:,1]))

        # vect_XY.append(np.array(ch[:,1])) #
        
        # Z = np.delete(ch, selected_features_index, axis=1)     #

        # for i in range(num_partition_features):                #
        #     vect_XYZ.append(np.array(selected_partition[:,i])) #
        #     vect_XZ.append(np.array(selected_partition[:,i]))  #

        vect_X1YZ.append(np.array(y))
        vect_X1X2YZ.append(np.array(y))
        vect_YZ.append(np.array(y))

        
        for j in range(len(Z[0])):
            vect_X1YZ.append(np.array(Z[:,j]))
            vect_X1Z.append(np.array(Z[:,j]))
            vect_X1X2Z.append(np.array(Z[:,j]))
            vect_X1X2YZ.append(np.array(Z[:,j]))
            vect_Z.append(np.array(Z[:,j]))
            vect_YZ.append(np.array(Z[:,j]))

            

        H_X1YZ = drv.entropy_joint(vect_X1YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        if len(vect_Z)!=0:
            H_Z = drv.entropy_joint(vect_Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_X1X2YZ = drv.entropy_joint(vect_X1X2YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_YZ = drv.entropy_joint(vect_YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_X1Z = drv.entropy_joint(vect_X1Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_X1X2Z = drv.entropy_joint(vect_X1X2Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
       
        #H_XY = drv.entropy_joint(vect_XY, base=2, fill_value=-1, estimator='ML', keep_dims=False) #

        # vect_XYZ_lst = [list(vect_XYZ[i]) for i in range(len(vect_XYZ))]
        # H_XYZ = self.entropy_pers(*vect_XYZ_lst)
        # H_XYZ_test = drv.entropy_joint(vect_XYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)

        # H_Z = self.entropy_pers(vect_Z)
        # H_YZ = self.entropy_pers(vect_YZ)
        # H_XZ = self.entropy_pers(vect_XZ)

        #cc = H_XYZ - H_Z - (H_XYZ - XX[0]) - (H_XYZ - XX[1]) ..... - YY

        #demonstration comparting with simple CMI (X;Y|Z)

        #CMI = drv.information_mutual_conditional(ch[:,1],np.array(y),ch[:,0],cartesian_product=True)

        if len(vect_Z)!=0:
            ccc = ((H_X1Z - H_Z) - (H_X1YZ - H_YZ)) + ((H_X1X2Z - H_X1Z) - (H_X1X2YZ - H_X1YZ))
        #else:
            #c = (H_X1YZ) - (H_X2YZ - H_X1Z)


        del vect_X1YZ[:]
        del vect_X2YZ[:]
        del vect_X1Z[:]
        del vect_Z[:]
        del vect_X1X2Z[:]
        del vect_X1X2YZ[:]
        del vect_YZ[:]

#---
        Z = ch
        Z = np.delete(ch, [False,False,False,False,True], axis=1)
        vect_X1Z.append(np.array(ch[:,4]))
        vect_X1X2Z.append(np.array(ch[:,4]))
        vect_X1X2YZ.append(np.array(ch[:,4]))
        vect_X1YZ.append(np.array(ch[:,4]))


        vect_X1YZ.append(np.array(y))
        vect_X1X2YZ.append(np.array(y))
        vect_YZ.append(np.array(y))

        
        for j in range(len(Z[0])):
            vect_X1YZ.append(np.array(Z[:,j]))
            vect_X1Z.append(np.array(Z[:,j]))
            vect_X1X2Z.append(np.array(Z[:,j]))
            vect_X1X2YZ.append(np.array(Z[:,j]))
            vect_Z.append(np.array(Z[:,j]))
            vect_YZ.append(np.array(Z[:,j]))

            
        H_X1YZ = drv.entropy_joint(vect_X1YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        if len(vect_Z)!=0:
            H_Z = drv.entropy_joint(vect_Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_X1X2YZ = drv.entropy_joint(vect_X1X2YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_YZ = drv.entropy_joint(vect_YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_X1Z = drv.entropy_joint(vect_X1Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_X1X2Z = drv.entropy_joint(vect_X1X2Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
       
        if len(vect_Z)!=0:
            cccc = ((H_X1Z - H_Z) - (H_X1YZ - H_YZ)) + ((H_X1X2Z - H_X1Z) - (H_X1X2YZ - H_X1YZ))



        print("CMI PARTITION CHAIN", ccc)
        return ccc

    def single_CMI(self, partition_name, Var, num_genes, ch, y, partition_index, selected_features_index, selected_partition, num_partition_features):

        v_YZ = []
        v_XYZ = []
        v_XZ = []
        v_Z = []

        ind_contr = []
        var_name = []

        single_contribution = []

        #num_partition_features = 1

        for l in range(num_partition_features):

            partition_index_copy = copy.copy(partition_index)
            partition_index_l = partition_index_copy.pop(l)

            new_selected_features_index = [False for _ in range(num_genes)]
            new_selected_features_index[partition_index_l] = True

            #demonstration
            # if l == 0:
            #     new_selected_features_index = [False,True,False,False,False]
            # else:
            #     new_selected_features_index = [False,False,False,False,True]

            #new_selected_features_index = [False,True,False,False,False] #
            Z_cond = np.delete(ch, new_selected_features_index, axis=1)
            #Z_cond = np.delete(ch, new_selected_features_index, axis=1) #
            
            v_XYZ.append(np.array(selected_partition[:,l]))
            v_XZ.append(np.array(selected_partition[:,l]))

            #v_XYZ.append(np.array(ch[:,1])) #
            #v_XZ.append(np.array(ch[:,1])) #


            v_XYZ.append(np.array(y))
            v_YZ.append(np.array(y))

            
            for j in range(len(Z_cond[0])):
                v_XYZ.append(np.array(Z_cond[:,j]))
                v_XZ.append(np.array(Z_cond[:,j]))
                v_YZ.append(np.array(Z_cond[:,j]))
                v_Z.append(np.array(Z_cond[:,j]))


            H_XYZ_ = drv.entropy_joint(v_XYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            H_Z_ = drv.entropy_joint(v_Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            H_YZ_ = drv.entropy_joint(v_YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            H_XZ_ = drv.entropy_joint(v_XZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)

            # H_XYZ_ = self.entropy_pers(v_XYZ)
            # H_Z_ = self.entropy_pers(v_Z)
            # H_YZ_ = self.entropy_pers(v_YZ)
            # H_XZ_ = self.entropy_pers(v_XZ)

            X_ = H_XYZ_ - H_YZ_
            YY_ = H_XYZ_ - H_XZ_

            var_name.append(self.Var[partition_index_l])
            ind_contr.append(H_XYZ_ - H_Z_ - X_ - YY_)


            test = (H_XZ_-H_Z_)-(H_XYZ_-H_YZ_)

            single_contribution_CMI_list = dict(zip(var_name, ind_contr))

            del v_YZ[:]
            del v_XYZ[:]
            del v_XZ[:]
            del v_Z[:]

        # print("SINGLE CONTRIBUTION CMI")
        # print(single_contribution_CMI_list)

            #TEST


        for var_ind in range(num_partition_features):
            single_contribution.append(drv.information_mutual(selected_partition[:,var_ind],np.array(y),cartesian_product=True))

        single_contribution_MI_list = dict(zip(partition_name, single_contribution))

        # print("SINGLE CONTRIBUTION MI")
        # print(single_contribution_MI_list)


        return single_contribution_CMI_list, single_contribution_MI_list

    def single_CMI_all(self, partition_name, Var, num_genes, ch, y, partition_index, selected_features_index, selected_partition, num_partition_features):

        v_YZ = []
        v_XYZ = []
        v_XZ = []
        v_Z = []

        ind_contr = []
        var_name = []

        single_contribution = []

        selected_partition = ch
        num_partition_features = 5
        partition_index = [0,1,2,3,4]

        for l in range(num_partition_features):

            partition_index_copy = copy.copy(partition_index)
            partition_index_l = partition_index_copy.pop(l)

            new_selected_features_index = [False for _ in range(num_genes)]
            new_selected_features_index[partition_index_l] = True

            Z_cond = np.delete(ch, new_selected_features_index, axis=1)

            
            v_XYZ.append(np.array(selected_partition[:,l]))
            v_XZ.append(np.array(selected_partition[:,l]))

            v_XYZ.append(np.array(y))
            v_YZ.append(np.array(y))

            
            for j in range(len(Z_cond[0])):
                v_XYZ.append(np.array(Z_cond[:,j]))
                v_XZ.append(np.array(Z_cond[:,j]))
                v_YZ.append(np.array(Z_cond[:,j]))
                v_Z.append(np.array(Z_cond[:,j]))


            H_XYZ_ = drv.entropy_joint(v_XYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            H_Z_ = drv.entropy_joint(v_Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            H_YZ_ = drv.entropy_joint(v_YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            H_XZ_ = drv.entropy_joint(v_XZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            # H_XYZ_ = self.entropy_pers(v_XYZ)
            # H_Z_ = self.entropy_pers(v_Z)
            # H_YZ_ = self.entropy_pers(v_YZ)
            # H_XZ_ = self.entropy_pers(v_XZ)

            X_ = H_XYZ_ - H_YZ_
            YY_ = H_XYZ_ - H_XZ_

            var_name.append(self.Var[partition_index_l])
            ind_contr.append(H_XYZ_ - H_Z_ - X_ - YY_)

            test = (H_XZ_-H_Z_)-(H_XYZ_-H_YZ_)

            single_contribution_CMI_list = dict(zip(var_name, ind_contr))

            del v_YZ[:]
            del v_XYZ[:]
            del v_XZ[:]
            del v_Z[:]

        # print("SINGLE CONTRIBUTION CMI")
        # print(single_contribution_CMI_list)

            #TEST


        for var_ind in range(num_partition_features):
            single_contribution.append(drv.information_mutual(selected_partition[:,var_ind],np.array(y),cartesian_product=True))

        single_contribution_MI_list = dict(zip(partition_name, single_contribution))

        # print("SINGLE CONTRIBUTION MI")
        # print(single_contribution_MI_list)


        return single_contribution_CMI_list, single_contribution_MI_list


    #CHAIN RULE MULTIPLE VARIABLES

    def chain_rule(self, ch, y, selected_partition, num_partition_features):

        M = []

        vectYZ = []
        vectXYZ = []
        vectXZ = []
        vectZ = []

        #demonstration with Multivariate CMI
        #selected_partition = ch
        #num_partition_features = 2

        a = drv.information_mutual(selected_partition[:,0],np.array(y),cartesian_product=True)
        M.append(a)

        if num_partition_features > 1:
            b = drv.information_mutual_conditional(selected_partition[:,1],np.array(y),selected_partition[:,0],cartesian_product=True)
            M.append(b)

            if num_partition_features > 2:
                i = 2

                for i in range(2,num_partition_features):
                    vectYZ.append(np.array(y))
                    vectXYZ.append(np.array(y))
                    vectXZ.append(selected_partition[:,i])
                    vectXYZ.append(selected_partition[:,i])

                    for j in range(i,-1,-1):
                        if (i != j):
                            vectXZ.append(selected_partition[:,j])
                            vectYZ.append(selected_partition[:,j])
                            vectXYZ.append(selected_partition[:,j])
                            vectZ.append(selected_partition[:,j])

                    _H_XZ = drv.entropy_joint(vectXZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                    _H_Z = drv.entropy_joint(vectZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                    _H_XYZ = drv.entropy_joint(vectXYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                    _H_YZ = drv.entropy_joint(vectYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
                    # _H_XZ = self.entropy_pers(vectXZ)
                    # _H_Z = self.entropy_pers(vectZ)
                    # _H_XYZ = self.entropy_pers(vectXYZ)
                    # _H_YZ = self.entropy_pers(vectYZ)
                                       
                    c = (_H_XZ-_H_Z)-(_H_XYZ-_H_YZ)

                    del vectXZ[:]
                    del vectYZ[:]
                    del vectXYZ[:]
                    del vectZ[:]

                    i = i + 1

                    M.append(c)

        MI = sum(M)

        return MI


    def chain_rule_H(self, ch, y, selected_partition, num_partition_features):

        M = []
        MM = []
        MMM = []

        vectYZ = []
        vectXYZ = []
        vectXZ = []
        vect1 = []
        vect2 = []
        

        #demonstration with Multivariate CMI
        selected_partition = ch
        num_partition_features = 4

        a = drv.entropy(ch[:,1], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        M.append(a)
        b = drv.entropy_conditional(y,ch[:,1], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        M.append(b)

        H_1 = drv.entropy_joint([ch[:,0],y,ch[:,1]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_2 = drv.entropy_joint([y,ch[:,1]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        c = H_1 - H_2
        M.append(c)

        H_1 = drv.entropy_joint([ch[:,2],ch[:,0],y,ch[:,1]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_2 = drv.entropy_joint([ch[:,0],y,ch[:,1]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        c = H_1 - H_2
        M.append(c)

        H_1 = drv.entropy_joint([ch[:,3],ch[:,2],ch[:,0],y,ch[:,1]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_2 = drv.entropy_joint([ch[:,2],ch[:,0],y,ch[:,1]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        c = H_1 - H_2
        M.append(c)

        H_1 = drv.entropy_joint([ch[:,4],ch[:,3],ch[:,2],ch[:,0],y,ch[:,1]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_2 = drv.entropy_joint([ch[:,3],ch[:,2],ch[:,0],y,ch[:,1]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        c = H_1 - H_2
        M.append(c)


        MI_1 = sum(M)



        aa = drv.entropy(y, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        MM.append(aa)
        bb = drv.entropy_conditional(ch[:,0],y, base=2, fill_value=-1, estimator='ML', keep_dims=False)
        MM.append(bb)

        H_1 = drv.entropy_joint([ch[:,2],y,ch[:,0]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_2 = drv.entropy_joint([y,ch[:,0]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        cc = H_1 - H_2
        MM.append(cc)

        H_1 = drv.entropy_joint([ch[:,3],y,ch[:,0],ch[:,2]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_2 = drv.entropy_joint([y,ch[:,0],ch[:,2]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        cc = H_1 - H_2
        MM.append(cc)

        H_1 = drv.entropy_joint([ch[:,4],y,ch[:,0],ch[:,2],ch[:,3]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_2 = drv.entropy_joint([y,ch[:,0],ch[:,2],ch[:,3]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        cc = H_1 - H_2
        MM.append(cc)


        MI_2 = sum(MM)


        aaa = drv.entropy(ch[:,0], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        MMM.append(aaa)
        bbb = drv.entropy_conditional(ch[:,2],ch[:,0], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        MMM.append(bbb)

        H_1 = drv.entropy_joint([ch[:,3],ch[:,2],ch[:,0]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_2 = drv.entropy_joint([ch[:,2],ch[:,0]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        ccc = H_1 - H_2
        MMM.append(ccc)

        H_1 = drv.entropy_joint([ch[:,4],ch[:,3],ch[:,2],ch[:,0]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        H_2 = drv.entropy_joint([ch[:,3],ch[:,2],ch[:,0]], base=2, fill_value=-1, estimator='ML', keep_dims=False)
        ccc = H_1 - H_2
        MMM.append(ccc)


        MI_3 = sum(MMM)


        return (MI_1 - MI_2)


    def entropy_pers(self,*vect):
            
            return  np.sum(-p * np.log2(p) if p > 0 else 0 for p in
                (np.mean(reduce(np.logical_and, (predictions == c for predictions, c in zip(vect, classes))))
                    for classes in itertools.product(*[set(x) for x in vect])))

    def binary(self,v,t):                    
        ret = 0
        if v>t:
            ret = 1
        return ret


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