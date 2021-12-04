import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm
import pandas as pd
from pprint import pprint
import random

def onemax(gene):
    return np.sum(gene)

class Gene():

    def __init__(self, gene) -> None:
        self.gene = gene
        self.fitness = self.get_fitness()
        
    
    @classmethod
    def make_random_instance(cls, length_gene):
        gene = cls(np.random.randint(0, 2, length_gene))
        return gene

    def get_fitness(self) -> float:
        return onemax(self.gene)

    def __lt__(self, other):
        return self.get_fitness() < other.get_fitness()

    def modify_gene(self, gene):
        self.gene = gene
        self.fitness = self.get_fitness()


class Population():

    def __init__(self) -> None:
        self.generation = 0
        self.rng = np.random.default_rng(1234)

        self.crossover_rate = 0
        self.mutation_rate = 0
        self.population = 50
        self.length_gene = 10
        self.generation_gap = 1.0
        self.num_elete_selection = 0
        
        self.population_selection = int((self.population * self.generation_gap )// 2 * 2) 
        self.population_copy = self.population - self.population_selection
        if self.num_elete_selection > self.population_copy:
            self.num_elete_selection = self.population_copy

        # generate each instances
        self.genes = [Gene.make_random_instance(self.length_gene) for _ in range(self.population)]

    
    def calc_fitness(self):
        self.fitness = []
        for gene in self.genes:
            self.fitness.append(self.onemax(gene))

    def reset(self):
        self.genes = [Gene.make_random_instance(self.length_gene) for _ in range(self.population)]
        self.generation = 0
            

    def step(self):
        next_generation = []
        # copy genes from parents generation
        if self.population_copy:
            # elete selection
            if self.num_elete_selection:
                next_generation.extend(self.elete_selection())
            
            num_random_selection = self.population_copy - self.num_elete_selection        
            if num_random_selection:
                # random selection from not selected genes in elete selection
                next_generation.extend(self.random_selection(
                    sorted(self.genes, reverse=True)[self.num_elete_selection:], 
                    num_selection=num_random_selection
                ))

        # selection
        selected_genes = []


        # crossover


        # mutation


        # calculate fitness
        # self.calc_fitness()

    # fitness function for onemax problem
    def onemax(self, gene):
        return np.sum(gene)
    
    def elete_selection(self):
        return sorted(self.genes, reverse=True)[:self.num_elete_selection]
    
    def random_selection(self, genes, num_selection):
        return random.choices(genes, k=num_selection)

    def uniform_crossover(self, gene1, gene2, length_gen):
        mask = self.rng.integers(low=0, high=1, size=length_gen)
        g1 = gene1 & mask | gene2 & ~mask
        g2 = gene1 & ~mask | gene2 & mask
        return g1, g2


if __name__ == '__main__':
    p = [Gene.make_random_instance(10) for _ in range(10)]
    print(len(p))
    pprint(p)
    for pop in p:
        print(pop.gene, pop.fitness)
    # print()
    print(max(p))


