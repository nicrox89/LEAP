
from copy import copy
import random
import numpy as np

def uniform_crossover_2(ind1, ind2, p_swap):
    """ Recombination operator that can potentially swap any matching pair of
    genes between two individuals with some probability.

    It is assumed that ind1.genome and ind2.genome are lists of things.

    :param ind1: The first individual
    :param ind2: The second individual
    :param p_swap:

    :return: a copy of both individuals with individual.genome bits
                swapped based on probability
    """
    if len(ind1.genome[0]) != len(ind2.genome[0]):
        # TODO what about variable length genomes?
        raise RuntimeError(
            'genomes must be same length for uniform crossover')

    ind_A = np.array(ind1.genome) #
    ind_B = np.array(ind2.genome) #
    ind_TMP = copy(ind_A) #

    for i in range(len(ind1.genome[0])):
        if random.random() < p_swap:
            ind_TMP[:, i] = ind_B[:, i] #
            ind_B[:, i] = ind_A[:, i] #
            ind_A[:, i] = ind_TMP[:, i] #
            #ind1.genome[:,i], ind2.genome[:,i] = ind2.genome[:,i], ind1.genome[:,i]

    ind1.genome = list(ind_A) #
    ind2.genome = list(ind_B) #

    return ind1, ind2