
from pyitlib import discrete_random_variable as drv

def __init__(self, classifier):
    self.classifier = classifier

def f(chromosome):
    num_chromosomes = len(chromosome[0])
    y=[]
    #predict
    for i in range(num_chromosomes):
        y.append(self.classifier.predict(chromosome[i]))
    
    #compute mutual array using y
    M=[]
    for i in range(num_chromosomes):
        M.append(drv.information_mutual(y[i], chromosome[:,i], cartesian_product=True))

    den = 0
    num = 0
    threshold = sum(M) * 0.3
    M.sort(reverse=True)

    den = 0
    while num < threshold:
        num=num+M[den]
        den=den+1

    return num/den