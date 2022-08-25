# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import sys
import array
import random
import numpy
# import all the DEAP parts
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

numCities = 100
random.seed(169)
x = numpy.random.rand(numCities)
y = numpy.random.rand(numCities)
# We want to minimize the distance so the weights have to be negative
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# The individuals are just single integer (typecode='i') array of dimension 1xnumCities
# We also assign the creator.FitnessMin that was just created in the line above
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)
toolbox = base.Toolbox()
# Attribute generator
toolbox.register("indices", random.sample, range(numCities), numCities)
# Structure initializers
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def evalTSP(individual):
    diffx = numpy.diff(x[individual])
    diffy = numpy.diff(y[individual])
    distance = numpy.sum(diffx ** 2 + diffy ** 2)
    return distance,


toolbox.register("evaluate", evalTSP)


def main():
    random.seed(169)
    # start with a population of 300 individuals
    pop = toolbox.population(n=1000)
    # only save the very best one
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    # use one of the built in GA's with a probablilty of mating of 0.7
    # a probability of mutating 0.2 and 140 generations.
    algorithms.eaSimple(pop, toolbox, 0.7, 0.2, 140, stats=stats,
                        halloffame=hof)
    # plot the best one
    ind = hof[0]
    plt.figure(2)
    plt.plot(x[ind], y[ind])
    plt.ion()
    plt.show()
    plt.pause(0.001)
    return pop, stats, hof


if __name__ == "__main__":
    main()
