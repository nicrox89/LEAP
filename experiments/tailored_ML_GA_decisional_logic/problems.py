#!/usr/bin/env python3
import numpy as np
import math
from pyitlib import discrete_random_variable as drv
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score
from sklearn.feature_selection import mutual_info_regression
from scipy import stats
from dit.multivariate import entropy

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
        for i in range(len(chromosome)):
            y.append(self.decide([chromosome[i]]))
        
        #compute mutual array using y
        M = []
        condMI = []
        condIndexes = []
        vect_XZ = []
        vect_XYZ = []
        vect_YZ = [] 
        vect_Z = []
        ch = np.array(chromosome)
        

        #TEST Entropy Joint
        # a = ch[:,0]
        # b = ch[:,1]
        # vect = [a,b]
        # vect_y = [a,b,y]
        # vect_no_a = [b,y]
        # H_1 = drv.entropy_joint(vect, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
        # H_2 = drv.entropy_joint(b, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
        # H_3 = drv.entropy_joint(vect_y, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
        # H_4 = drv.entropy_joint(vect_no_a, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
        # tot = (H_1-H_2)-(H_3-H_4)
        # m = (drv.information_mutual_conditional(a,np.array(y),b,cartesian_product=True))
        v = ch[:,[0,2,3,4]]

        jev = [drv.entropy_joint(x, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False) for x in v]

        a = drv.information_mutual_conditional(ch[:,1],np.array(y),ch[:,4],cartesian_product=True)
        b = drv.information_mutual_conditional(ch[:,4],np.array(y),ch[:,1],cartesian_product=True)             


        for i in range(num_genes):
            vect_YZ.append(np.array(y))
            vect_XYZ.append(np.array(y))
            vect_XZ.append(ch[:,i])
            vect_XYZ.append(ch[:,i])

            for j in range(num_genes):
                if (i != j):
                    condIndexes.append(j)
                    vect_XZ.append(ch[:,j])
                    vect_YZ.append(ch[:,j])
                    vect_XYZ.append(ch[:,j])
                    vect_Z.append(ch[:,j])

                    print("Single Conditional Mutual Information drv")
                    print(Var[i],"-",Var[j])
                    c = drv.information_mutual_conditional(ch[:,i],np.array(y),ch[:,j],cartesian_product=True)
                    print(c)

                    print("Single Conditional Mutual Information je")
                    print(Var[i],"-",Var[j])
                    xz = [ch[:,i], ch[:,j]]
                    z = ch[:,j]
                    xyz = [ch[:,i], y, ch[:,j]]
                    yz = [y, ch[:,j]]
                    jeXZ = drv.entropy_joint(xz, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
                    jeZ = drv.entropy_joint(z, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
                    jeXYZ = drv.entropy_joint(xyz, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
                    jeYZ = drv.entropy_joint(yz, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
                    cc = (jeXZ - jeZ) - (jeXYZ - jeYZ)
                    print(c)

            H_XZ = drv.entropy_joint(vect_XZ, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
            #H_XZ_test =  stats.entropy(vect_XZ)
            H_Z = drv.entropy_joint(vect_Z, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
            H_XYZ = drv.entropy_joint(vect_XYZ, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
            H_YZ = drv.entropy_joint(vect_YZ, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
            C_MI = (H_XZ-H_Z)-(H_XYZ-H_YZ)

            # H_YZ = drv.entropy_joint(vect_XZ, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
            # H_Z = drv.entropy_joint(vect_Z, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
            # H_YXZ = drv.entropy_joint(vect_XYZ, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
            # H_XZ = drv.entropy_joint(vect_YZ, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False)
            # CMI_test = (H_XZ-H_Z)-(H_XYZ-H_YZ)

            print("Multiple Conditional Mutual Information")
            print(Var[i])
            print(C_MI)
            condMI.append(C_MI)
            del vect_XZ[:]
            del vect_YZ[:]
            del vect_XYZ[:]
            del vect_Z[:]
            M.append(drv.information_mutual(ch[:,i],np.array(y),cartesian_product=True))
            print("Mutual Information")
            print(M[i])





        

            #mi.append(drv.information_mutual_conditional(a,np.array(y),b,cartesian_product=True))
            #mi.append(drv.information_mutual_conditional(np.array(y),a,b,cartesian_product=True))
            #M.append(drv.information_mutual_conditional(ch[:,i],ch[:,i+1],np.array(y),cartesian_product=True))
            #M.append(mutual_info_score(ch[:,i],np.array(y),contingency=None))
            #M.append(normalized_mutual_info_score(ch[:,i],np.array(y),average_method='arithmetic'))
            #M.append(mutual_info_regression(ch[:,i],np.array(y),discrete_features='auto'))
    
        #------v2.0
        #dictionary Variables-MI

        d = {"".join(Var[0]):M[0]}
        #print(d)

        for i in range(len(Var)):
            d["".join(Var[i])] = M[i]
        #print(d)

        sort_orders = sorted(d.items(), key=lambda x: x[1], reverse=True)
        #print(sort_orders)

        best = []
        #------
        den = 0
        num = 0
        res = 0
        summary = 0
        summary = sum(M)
        threshold = summary * 0.5
        
        M.sort(reverse=True)
        #print(M)

        #den = 0
        while num < threshold:
            num=num+M[den]
            den=den+1
        if den!=0:
            #res = num/den
            #den_n = (den-1) / (8 - 1)
            #res = num-den_n
            res = num*math.exp(-den)
            #res = num + den
            #res = math.pow(num,-den)
        #------v2.0
        for i in range(den):
            best.append(sort_orders[i])
        self.stat.append([res,den,best])
        #------
        #self.stat.append([res,den])
            #print("MI:",res)
            #print("num features:",den)
        return res

    def getStat(self):
        return self.stat

    def setStat(self):
        self.stat = []