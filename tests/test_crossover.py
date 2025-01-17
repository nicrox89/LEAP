"""
    Unit tests for crossover operators
"""
import itertools

import pytest

from leap_ec.individual import Individual
import leap_ec.ops as ops
import leap_ec.statistical_helpers as stat


def test_uniform_crossover():
    pop = [Individual([0, 0]),
           Individual([1, 1])]

    # We need a cyclic generator because there are only two individuals in the population, and once the first two
    # are selected for uniform crossover, the next two parents are selected and crossed over.  The cyclic iterator
    # ensures we just select the same two individuals again even though the yield statements in the uniform
    # crossover operator are not invoked again.
    i = ops.naive_cyclic_selection(pop)

    # Do swap with 100% certainty, which will cause the two individuals' genomes to exchange values
    new_pop = list(itertools.islice(ops.uniform_crossover(i, p_swap=1.0), 2))
    assert new_pop[0].genome == [1,1]
    assert new_pop[1].genome == [0,0]

    # Note because we didn't clone the selected individuals, *the original population was changed*.
    assert pop[0].genome == [1,1]
    assert pop[1].genome == [0,0]

# These tests are now moot given the advent of variable length segments.

# def test_uniform_crossover_bad_len():
#     """ Test assertion for mis-matched genome lengths
#     """
#     pop = [Individual([0, 0, 1]),
#            Individual([1, 1])]
#
#     i = ops.naive_cyclic_selection(pop)
#
#     with pytest.raises(RuntimeError):
#         new_pop = list(itertools.islice(ops.uniform_crossover(i), 2))
#
#
# def test_n_ary_crossover_bad_lengths():
#     """ Genomes should be the same length for crossover
#
#     (Separate tests for variable length crossover.)
#     """
#     pop = [Individual([0, 0, 1]),
#            Individual([1, 1])]
#
#     i = ops.naive_cyclic_selection(pop)
#
#     with pytest.raises(RuntimeError):
#         new_pop = list(itertools.islice(ops.n_ary_crossover(i), 2))


def test_n_ary_crossover_bad_crossover_points():
    """ Test assertions for having more crossover points than genome length """
    pop = [Individual([0, 0]),
           Individual([1, 1])]

    i = ops.naive_cyclic_selection(pop)

    with pytest.raises(RuntimeError):
        new_pop = list(itertools.islice(ops.n_ary_crossover(i, num_points=3), 2))


def test_n_ary_crossover():
    """If we crossover two individuals with two bits each, the children should either be swapped copies of their parents,
    or they should exchange the second bit and keep the first bit unmodified."""
    pop = [Individual([0, 0]),
           Individual([1, 1])]

    i = ops.naive_cyclic_selection(pop)

    new_pop = list(itertools.islice(ops.n_ary_crossover(i), 2))

    # Given that there are only two genes, one [0,0] and the other [1,1] and a single crossover point, and that the
    # only two valid crossover points are 0 or 1, then there are two possible valid states for offspring with single
    # point crossover.
    assert pop[0].genome == [1,1] or pop[0].genome == [0,1]
    assert pop[1].genome == [0,0] or pop[1].genome == [1,0]


@pytest.mark.stochastic
def test_n_ary_crossover_probability():
    """If we perform crossover with a probabilty of 0.5, then the individuals will be unmodified 50% of the time."""
    N = 1000
    unmodified_count = 0

    for i in range(N):

        pop = [Individual([0, 0]),
               Individual([1, 1])]
        i = ops.naive_cyclic_selection(pop)
        new_pop = list(itertools.islice(ops.n_ary_crossover(i, p=0.5), 2))

        if new_pop[0].genome == [0, 0] and new_pop[1].genome == [1, 1]:
            unmodified_count += 1

    p = 0.01
    observed_dist = {'Unmodified': unmodified_count, 'Modified': N - unmodified_count }
    assert(stat.equals_uniform(observed_dist, p=p))