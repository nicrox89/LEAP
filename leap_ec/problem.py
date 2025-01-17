"""
Defines the abstract-base classes Problem, ScalarProblem,
and FunctionProblem.

"""
from math import nan, floor
import random
from abc import ABC, abstractmethod

from leap_ec.context import context


##############################
# Class Problem
##############################
class Problem(ABC):
    """
        Abstract Base Class used to define problem definitions.

        A `Problem` is in charge of two major parts of an EA's behavior:

         1. Fitness evaluation (the `evaluate()` method)

         2. Fitness comparision (the `worse_than()` and `equivalent()` methods)
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def evaluate(self, phenome, *args, **kwargs):
        """
        Decode and evaluate the given individual based on its genome.

        Practitioners *must* over-ride this member function.

        Note that by default the individual comparison operators assume a
        maximization problem; if this is a minimization problem, then just
        negate the value when returning the fitness.

        :param phenome:
        :return: fitness
        """
        raise NotImplementedError

    @abstractmethod
    def worse_than(self, first_fitness, second_fitness):
        raise NotImplementedError

    @abstractmethod
    def equivalent(self, first_fitness, second_fitness):
        raise NotImplementedError


##############################
# Class ScalarProblem
##############################
class ScalarProblem(Problem):
    def __init__(self, maximize):
        super().__init__()
        self.maximize = maximize

    def worse_than(self, first_fitness, second_fitness):
        """
            Used in Individual.__lt__().

            By default returns first_fitness < second_fitness if a maximization
            problem, else first_fitness > second_fitness if a minimization
            problem.  Please over-ride if this does not hold for your problem.

            :return: true if the first individual is less fit than the second
        """
        # NaN is assigned if the individual is non-viable, which can happen if
        # an exception is thrown during evaluation. We consider NaN fitnesses to
        # always be the worse possible with regards to ordering.
        if first_fitness is nan:
            if second_fitness is nan:
                # both are nan, so to reduce bias bitflip a coin to arbitrarily
                # select one that is worst.
                return random.choice([True, False])
            # Doesn't matter how awful second_fitness is, nan will already be
            # considered worse.
            return True
        elif second_fitness is nan:
            # No matter how awful the first_fitness is, if it's not a NaN the
            # NaN will always be worse
            return False

        # TODO If we accidentally pass an Individual in as first_ or second_fitness,
        # TODO then this can result in an infinite loop.  Add some error
        # handling for this.
        if self.maximize:
            return first_fitness < second_fitness
        else:
            return first_fitness > second_fitness

    def equivalent(self, first_fitness, second_fitness):
        """
            Used in Individual.__eq__().

            By default returns first.fitness== second.fitness.  Please
            over-ride if this does not hold for your problem.

            :return: true if the first individual is equal to the second
        """
        if first_fitness is nan and second_fitness is nan:
            return True

        # TODO Should we consider (abs(first_fitness-second_fitness) < epsilon)
        return first_fitness == second_fitness


##############################
# Class FunctionProblem
##############################
class FunctionProblem(ScalarProblem):

    def __init__(self, fitness_function, maximize):
        super().__init__(maximize)
        self.fitness_function = fitness_function

    def evaluate(self, phenome, *args, **kwargs):
        return self.fitness_function(phenome, *args, **kwargs)


##############################
# Class ConstantProblem
##############################
class ConstantProblem(ScalarProblem):
    """A flat landscape, where all phenotypes have the same fitness.

    This is sometimes useful for sanity checks or as a control in certain
    kinds of research.

    .. math::

       f(\\vec{x}) = c

    :param float c: the fitness value to return for any input.

    .. plot::
       :include-source:

       from leap_ec.problem import ConstantProblem
       from leap_ec.real_rep.problems import plot_2d_problem
       bounds = ConstantProblem.bounds
       plot_2d_problem(ConstantProblem(), xlim=bounds, ylim=bounds, granularity=0.025)

    """

    """Default bounds."""
    bounds = (-1.0, 1.0)

    def __init__(self, maximize=False, c=1.0):
        super().__init__(maximize)
        self.c = c

    def evaluate(self, phenome, *args, **kwargs):
        """
        Return a contant value for any input phenome:

        >>> phenome = [0.5, 0.8, 1.5]
        >>> ConstantProblem().evaluate(phenome)
        1.0

        >>> ConstantProblem(c=500.0).evaluate('foo bar')
        500.0

        :param phenome: real-valued vector to be evaluated
        :return: 1.0, or the constant defined in the constructor
        """
        return self.c

    def __str__(self):
        return ConstantProblem.__name__


########################
# Class AlternatingProblem
########################
class AlternatingProblem(Problem):
    def __init__(self, problems, modulo, context=context):
        assert(len(problems) > 0)
        assert(modulo > 0)
        assert(context is not None)
        self.problems = problems
        self.modulo = modulo
        self.context = context
        self.current_problem_idx = 0

    def _get_current_problem(self):
        assert('leap' in self.context)
        assert('generation' in self.context['leap'])
        step = self.context['leap']['generation']

        i = floor(step / self.modulo) % len(self.problems)

        return self.problems[i]

    def evaluate(self, phenome):
        return self._get_current_problem().evaluate(phenome)

    def worse_than(self, first_fitness, second_fitness):
        return self._get_current_problem().worse_than(first_fitness, second_fitness)

    def equivalent(self, first_fitness, second_fitness):
        return self._get_current_problem().equivalent(first_fitness, second_fitness)
    