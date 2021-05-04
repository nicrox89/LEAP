from leap_ec.algorithm import generational_ea
from leap_ec import representation, ops


from leap_ec.segmented_rep import initializers, decoder
from leap_ec.segmented_rep.problems import f as function
from leap_ec.real_rep.initializers import create_real_vector
from leap_ec.segmented_rep.initializers import create_segmented_sequence
from leap_ec.segmented_rep.ops import apply_mutation

from leap_ec.problem import FunctionProblem

from experiments.classifier import binaryClassifier

bounds = ((0, 1), (0, 1), (0, 1), (0, 1)) # 4 variables normalized between 0 and 1

pop_size = 5

#THE MATRIX
seqs = [] # Save sequences for next step
for i in range(pop_size):
    seq = create_segmented_sequence(4, create_real_vector(bounds)) # a sample
    seqs.append(seq)


ea = generational_ea(generations=10, pop_size=pop_size,

                     # Solve a MaxOnes Boolean optimization problem
                     problem=FunctionProblem(function, True),

                     representation=representation.Representation(
                         # Genotype and phenotype are the same for this task
                         decoder=decoder.IdentityDecoder(),
                         # Initial genomes are random binary sequences
                         initialize=initializers.create_binary_sequence(length=10)
                     ),

                     # The operator pipeline
                     pipeline=[ops.tournament_selection,
                               # Select parents via tournament_selection selection
                               ops.clone,  # Copy them (just to be safe)
                               # Basic mutation: defaults to a 1/L mutation rate
                                apply_mutation,
                               # Crossover with a 40% chance of swapping each gene
                               ops.uniform_crossover(p_swap=0.4),
                               ops.evaluate,  # Evaluate fitness
                               # Collect offspring into a new population
                               ops.pool(size=pop_size)
                               ])

print('Generation, Best_Individual')
for i, best in ea:
    print(f"{i}, {best}")