from dataclasses import dataclass
import random
from copy import deepcopy


class Individual:
    
    def __init__(self, genome, data, y, fit, var):
        self.genome = genome
        self.fitness = None
        self.data = data
        self.y = y 
        self.fit = fit
        self.var = var

    def init_population(self, population_size, individual_size):
        population = []

        for i in range(population_size):
            #individual = [0]*individual_size

            def generateBinaryString(N):
                S = ""
                for i in range(N):
                    x = random.randint(0, 1)
                    S += str(x)
                return (S, [int(i) for i in S])

            def addCombination(N, num):
                if 2**N < num:
                    num = 2**N
                    
                d = dict()
                for _ in range(num):
                    s = generateBinaryString(N)
                    while s[0] in d:
                        s = generateBinaryString(N)
                    d[s[0]] = s[1]
                return list(d.values())

        pop = addCombination(individual_size, population_size)
        population = [Individual(val, self.data, self.y, self.fit, self.var) for val in pop]
        
        return population

    def get_fitness(self, b):
        self.fitness = self.fit.f(self.genome, self.data, self.y, b, self.var)

    def clone(self):
        """Create a 'clone' of this `Individual`, copying the genome, but not
        fitness.

        A deep copy of the genome will be created, so if your `Individual`
        has a custom genome type, it's important that it implements the
        `__deepcopy__()` method.

        >>> from leap_ec.binary_rep.problems import MaxOnes
        >>> from leap_ec.decoder import IdentityDecoder
        >>> ind = Individual([0, 1, 1, 0], IdentityDecoder(), MaxOnes())
        >>> ind_copy = ind.clone()
        >>> ind_copy.genome == ind.genome
        True
        >>> ind_copy.problem == ind.problem
        True
        >>> ind_copy.decoder == ind.decoder
        True
        """
        new_genome = deepcopy(self.genome)
        cloned = type(self)(new_genome, self.data, self.y, self.fit)
        cloned.fitness = None
        return cloned

    #def evaluate(self):
    #    self.get_fitness()