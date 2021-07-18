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

        _vect_YZ = []
        _vect_Z = []
        _vect_YXZ  = []
        _vect_XZ = []
        ch = np.array(chromosome)

        a = np.array(['1','2','3','0'])
        b = np.array(['3','4','1','5'])
        c = np.array(['0','4','1','5'])
        aa = [int(x) for x in a]
        bb = [int(x) for x in b]
        cc = [int(x) for x in c]
        #je =  stats.entropy(aa,bb,cc)
        #print("joint entropy : ",je)

        #out = np.nansum(-ch[:,0] * np.log2(ch[:,0]))
        #print(out)
        
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
        
        #v = ch[:,[0,2,3,4]]

        #jev = [drv.entropy_joint(x, base=2, fill_value=-1, estimator='ML', Alphabet_X=None, keep_dims=False) for x in v]

        a = drv.information_mutual_conditional(ch[:,1],np.array(y),ch[:,4],cartesian_product=True)
        b = drv.information_mutual_conditional(ch[:,4],np.array(y),ch[:,1],cartesian_product=True)             

        #cje = self.C_JE(ch[:,1], np.array(y), ch[:,0], Alphabet_X=None, Alphabet_Y=None, Alphabet_Z=None, cartesian_product=False)

        for i in range(num_genes):
            vect_YZ.append(np.array(y))
            vect_XYZ.append(np.array(y))
            vect_XZ.append(ch[:,i])
            vect_XYZ.append(ch[:,i])

            _vect_YZ.append(np.array(y))
            _vect_YXZ.append(np.array(y))
            _vect_XZ.append(ch[:,i])
            _vect_YXZ.append(ch[:,i])

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
                    print(cc)

            H_XZ = drv.entropy_joint(vect_XZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            H_Z = drv.entropy_joint(vect_Z, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            H_XYZ = drv.entropy_joint(vect_XYZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            H_YZ = drv.entropy_joint(vect_YZ, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            C_MI = (H_XZ-H_Z)-(H_XYZ-H_YZ)

            print()            
            print("Multivariate Conditional Mutual Information")
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

            HX = drv.entropy(ch[:,1], base=2, fill_value=-1, estimator='ML', keep_dims=False)
            HXcondY = drv.entropy_conditional(ch[:,1],y, base=2, fill_value=-1, estimator='ML', keep_dims=False)
            Mutual_Info = HX - HXcondY
            print("Mutual Information Gender je")
            print(Mutual_Info)

            #Gender Single CMI 
            print("Gender Single CMI ")
            print(drv.information_mutual_conditional(ch[:,1],np.array(y),[ch[:,0],ch[:,2],ch[:,3],ch[:,4]],cartesian_product=True))

            print()

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
            #res = num*math.exp(-den)
            res = sum(M)
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