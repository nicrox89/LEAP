import sys
from leap_ec.algorithm import generational_ea
from leap_ec import representation, ops

from classifier import binaryClassifier

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

from leap_ec.problem import FunctionProblem

p = fitness(binaryClassifier())


#number of individuals(matrixs)
pop_size = 20
#number of instances for each gene(variable) = number of records(observations) of the matrix
gene_size = 10

#bounds = ((0, 1), (0, 1), (0, 1), (0, 1)) # 4 variables normalized between 0 and 1
bounds = ((17,90), (0, 1), (0,99999), (1,99)) # 4 variables original bounds (int)

# #THE MATRIX
# seqs = [] # Save sequences for next step
# for i in range(pop_size):
#     seq = create_segmented_sequence(gene_size, create_int_vector(bounds)) # a sample - check float
#     seqs.append(seq)

# p.f(seq)

# ea = generational_ea(generations=10, pop_size=pop_size,

#                      # Solve a MaxOnes Boolean optimization problem
#                      problem=FunctionProblem(p.f, True),

#                      representation=representation.Representation(
#                          # Genotype and phenotype are the same for this task
#                          decoder=decoder.IdentityDecoder(),
#                          # Initial genomes are random real sequences
#                          #initialize=initializers.create_segmented_sequence(gene_size, create_real_vector(bounds))
#                          # Initial genomes are random discrete sequences
#                          initialize=initializers.create_segmented_sequence(gene_size, create_int_vector(bounds))

#                      ),

#                      # The operator pipeline
#                      pipeline=[ops.tournament_selection,
#                                # Select parents via tournament_selection selection
#                                ops.clone,  # Copy them (just to be safe)
#                                #probe.print_individual(prefix='before mutation: '),
#                                # Basic mutation: defaults to a 1/L mutation rate
#                                # segmented vectors mutation
#                                # apply_mutation,
#                                mutate_bitflip,
#                                # discrete vectors mutation
#                                # mutate_randint,
#                                #probe.print_individual(prefix='after mutation: '),
#                                # Crossover with a 40% chance of swapping each gene
#                                ops.uniform_crossover(p_swap=0.4),
#                                ops.evaluate,  # Evaluate fitness
#                                # Collect offspring into a new population
#                                ops.pool(size=pop_size)
#                                #yield (generation_counter.generation(), bsf)
#                                ])


# print('Generation, Best_Individual')
# for i, best in ea:
#     print(f"{i}, {best}")


def init(length, seq_initializer):
    def create():
        return initializers.create_segmented_sequence(gene_size, create_int_vector(bounds))
    return create

#create initial rand population of 5 individuals

parents = Individual.create_population(n=pop_size,
                                       initialize=init(gene_size, create_int_vector(bounds)),
                                       decoder=decoder.IdentityDecoder(),
                                       problem=FunctionProblem(p.f, True))


# Evaluate initial population
parents = Individual.evaluate_population(parents)


# print initial, random population
util.print_population(parents, generation=0)

# generation_counter is an optional convenience for generation tracking
generation_counter = util.inc_generation(context=context)

while generation_counter.generation() < 6:
    #sequence of functions, the result of the first one will be the parameter of the next one, and so on
    offspring = pipe(parents,
                     ops.tournament_selection,
                     ops.clone,
                     mutate_bitflip,
                     #util.print_individual(prefix='before mutation: '),
                     #individual_mutate_randint,
                     #util.print_individual(prefix='after mutation: '),
                     ops.uniform_crossover(p_swap=0.2),
                     ops.evaluate,
                     ops.pool(size=len(parents)))  # accumulate offspring

    parents = offspring

    generation_counter()  # increment to the next generation

    #util.print_population(parents, context['leap']['generation'])
    
    count=0

    for individual in parents:
        print("generation", context['leap']['generation'])
        count=count+1
        print("individual", count)
        print(individual.genome)
        print(individual.fitness)

    #probe.best_of_gen(parents)
        