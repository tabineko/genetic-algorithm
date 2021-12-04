import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean
from numpy.core.numeric import cross
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

    def get_fitness(self) -> int:
        return onemax(self.gene)

    def __lt__(self, other):
        return self.get_fitness() < other.get_fitness()

    def modify_gene(self, gene):
        self.gene = gene
        self.fitness = self.get_fitness()


class Population():

    def __init__(self) -> None:
        self.generation = 0

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
        self.fitnesses = []
        for gene in self.genes:
            self.fitnesses.append(self.onemax(gene))

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
        selected_genes = self.tournament_selection()

        # crossover
        # shuffle the selected list and reshape into (n, 2)
        selected_genes = random.shuffle(selected_genes)
        selected_genes = np.array(selected_genes).reshape(-1, 2).tolist()

        # loop length_of_genes_conbination times to generate new genes.
        # make new genes and generate Gene object and add it to nextgen_list.
        # crossovered_genes = []
        for gene1, gene2 in selected_genes:
            g1, g2 = self.uniform_crossover(gene1.gene, gene2.gene)
            gene1.modify_gene(g1)
            gene2.modify_gene(g2)
            # crossovered_genes.extend([Gene(gene=g1), Gene(gene=g2)])
        selected_genes = np.array(selected_genes).flatten().tolist()

        # mutation
        selected_genes = self.mutation(selected_genes)

        next_generation.extend(selected_genes)

        self.genes = next_generation
        self.calc_fitness()
        np.mean(self.fitnesses)
        np.max(self.fitnesses)
        np.min(self.fitnesses)
        self.generation += 1



    # fitness function for onemax problem
    def onemax(self, gene):
        return np.sum(gene.gene)
    
    def elete_selection(self):
        return sorted(self.genes, reverse=True)[:self.num_elete_selection]
    
    def random_selection(self, genes, num_selection):
        return random.choices(genes, k=num_selection)

    def roulette_selection(self):
        pass

    def tournament_selection(self, tournament_size=2):
        # winners = []
        # for _ in range(self.population_selection):
        #     sampled = random.sample(self.genes, k=tournament_size)
        #     winners.append(max(sampled))
        
        return [max(random.sample(self.genes, k=tournament_size)) for _ in range(self.population_selection)]

    def rank_selection(self):
        pass

    def uniform_crossover(self, gene1, gene2):
        mask = self.rng.integers(low=0, high=1, size=self.length_gene)
        g1 = gene1 & mask | gene2 & ~mask
        g2 = gene1 & ~mask | gene2 & mask
        return g1, g2
    
    def mutation(self, genes):
        for gene in genes:
            mask=random.choices([0, 1], k=self.length_gene, weights=[1-self.mutation_rate, self.mutation_rate])
            gene.modify_gene(gene.gene ^ mask)
        
        return genes



if __name__ == '__main__':
    p = [Gene.make_random_instance(10) for _ in range(10)]
    print(len(p))
    pprint(p)
    for pop in p:
        print(pop.gene, pop.fitness)
    # print()
    print(max(p))


