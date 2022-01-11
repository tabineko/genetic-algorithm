import numpy as np
import matplotlib.pyplot as plt
import random
# https://github.com/waqqasiq/n-queen-problem-using-genetic-algorithm/blob/master/N-Queen_GeneticAlgo.py

# fitness_function. count collisions and sub it from possible highest fitness.

def fitness_function(gene):
    '''calculate fitness

    calculate fitness of genetics for n-queen problem

    Args:
        gene (np.ndarray): list of genetics
    Returns:
        int: fitness value of given genetics
    '''
    n = len(gene)
    max_fitness = (n * (n-1)) / 2
    
    h_collisions = len(gene) - len(np.unique(gene))
    d_collisions = 0

    for i, g in enumerate(gene):
        for j in range(n):
            # same column
            if g != j:
                diff_row = abs(j - g)
                # left diag
                if i - diff_row >= 0:
                    if gene[i - diff_row] == j:
                        d_collisions += 1
                # right diag
                if i + diff_row < n:
                    if gene[i + diff_row] == j:
                        d_collisions += 1
    
    # print(h_collisions, d_collisions)
    return int(max_fitness - (h_collisions + d_collisions))


class Gene():
    ''' Genetic class

    An object represents each individual

    Attributes:
        gene (np.ndarray): represents genetics
        fitness (int): fitness value

    '''
    def __init__(self, gene) -> None:
        self.gene = gene
        self.fitness = self.calculate_fitness()
        
    
    @classmethod
    def make_random_instance(cls, length_gene):
        gene = cls(np.random.randint(0, length_gene, length_gene))
        return gene

    def calculate_fitness(self) -> int:
        return fitness_function(self.gene)

    def __lt__(self, other):
        return self.calculate_fitness() < other.calculate_fitness()

    def modify_gene(self, gene):
        self.gene = gene
        self.fitness = self.calculate_fitness()


class Population():
    ''' set of some gene class

    a class of population which is set of genetics

    Attributes:
        generation (int): number of generation of current population
        max (int): maximum fitness of current population
        min (int): minimum fitness of current population
        mean (int): mean value of fitness of current population
        rng (np.random.rnd): random generator
        crossover_prob (float): probability to mutate of each selected pairs
        mutation_prob (float): probability of mutation of each gene of each indiv.
        population (int): the number of population of each genearation
        length_gene (int): length of each genetics
        generation_gap (float): ratio of how 

    
    '''

    def __init__(self, crossover_prob=0.1, mutation_prob=0.005, population=10, tournament_size=2, 
        length_gene=10, generation_gap=0.8, num_elete_selection=0, print_step=False) -> None:
        self.generation = 0
        self.max = 0
        self.min = 0
        self.mean = 0
        self.rng = np.random.default_rng(1234)

        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.population = population
        self.length_gene = length_gene
        self.generation_gap = generation_gap
        self.num_elete_selection = num_elete_selection
        self.tournament_size = tournament_size
        self.print_step = print_step
        
        # recalculate selection num from parents generation
        self.population_selection = int((self.population * self.generation_gap )// 2 * 2) 
        self.population_copy = self.population - self.population_selection
        if self.num_elete_selection > self.population_copy:
            self.num_elete_selection = self.population_copy
        print(self.population_selection, self.num_elete_selection, self.population_copy)
        

        # generate each instances
        self.genes = [Gene.make_random_instance(self.length_gene) for _ in range(self.population)]
        self.calc_fitnesses()

    def calc_fitnesses(self):
        self.fitnesses = []
        for gene in self.genes:
            self.fitnesses.append(fitness_function(gene.gene))

    def reset(self):
        self.genes = [Gene.make_random_instance(self.length_gene) for _ in range(self.population)]
        self.generation = 0
        self.max = 0
        self.min = 0
        self.mean = 0
        self.calc_fitnesses()
            

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
        selected_genes = self.tournament_selection(tournament_size=self.tournament_size)

        # crossover
        # shuffle the selected list and reshape into (n, 2)
        selected_genes = random.sample(selected_genes, len(selected_genes))
        selected_genes = np.array(selected_genes).reshape(-1, 2).tolist()

        # loop length_of_genes_conbination times to generate new genes.
        # make new genes and generate Gene object and add it to nextgen_list.
        crossovered_genes = []
        for gene1, gene2 in selected_genes:
            g1, g2 = self.uniform_crossover(gene1.gene, gene2.gene)
            crossovered_genes.extend([Gene(gene=g1), Gene(gene=g2)])
        # selected_genes = np.array(selected_genes).flatten().tolist()
        selected_genes = crossovered_genes

        # mutation
        selected_genes = self.mutation(selected_genes)

        next_generation.extend(selected_genes)

        self.genes = next_generation
        self.calc_fitnesses()
        self.generation += 1
        self.max = np.max(self.fitnesses)
        self.min = np.min(self.fitnesses)
        self.mean = np.mean(self.fitnesses)

        if self.print_step:
            print(self.generation, 
            np.mean(self.fitnesses),
            np.max(self.fitnesses),
            np.min(self.fitnesses))      

    
    def elete_selection(self):
        return sorted(self.genes, reverse=True)[:self.num_elete_selection]
    
    def random_selection(self, genes, num_selection):
        return random.choices(genes, k=num_selection)

    def roulette_selection(self):
        pass

    def tournament_selection(self, tournament_size):
        winners = []
        for _ in range(self.population_selection):
            sampled = random.sample(self.genes, k=tournament_size)
            winners.append(sampled[np.argmax(sampled)])
        
        return winners
        
        # return [max(random.sample(self.genes, k=tournament_size)) for _ in range(self.population_selection)]

    def rank_selection(self):
        pass

    def uniform_crossover(self, gene1, gene2):
        # if do not crossover
        if random.random() > self.crossover_prob:
            return gene1, gene2
        
        # if do crossover
        mask = np.array(random.choices([True, False], k=self.length_gene))

        g1 = gene1 * mask + gene2 * ~mask
        g2 = gene1 * ~mask + gene2 * mask
        
        return g1, g2
    
    def mutation(self, genes):
        for gene in genes:
            mask = np.array(random.choices([True, False], k=self.length_gene, weights=[1-self.mutation_prob, self.mutation_prob]))
            rands = random.choices(list(range(self.length_gene)), k=self.length_gene)
            gene.modify_gene(gene.gene * mask + rands * ~mask)
        
        return genes


if __name__ == '__main__':

    # example = np.array([4, 1, 2, 0, 5, 3])
    # print(fitness_function(example)) 

    # pop = Population(length_gene=4, crossover_prob=0.8, population=5, num_elete_selection=0, print_step=False)
    # for gene in pop.genes:
    #     print(gene.gene)
    
    # print()

    # pop.step()

    # print()

    # for gene in pop.genes:
    #     print(gene.gene)

    # eletes = [4]
    # hist = [[]]

    # queen_num = 7
    # max_fitness = (queen_num*(queen_num-1))/2 

    # for j, s in enumerate(eletes):
    #     pop = Population(length_gene=queen_num, crossover_prob=0.5, mutation_prob=0.3, population=100, num_elete_selection=s, print_step=True)

    #     for i in range(50):
    #         while pop.max < max_fitness:
    #             pop.step()
    #             print(pop.max)
    #         hist[j].append(pop.generation)
    #         print(pop.max, max(pop.genes).gene)
    #         pop.reset()
    
    # plt.boxplot(hist, labels=list(map(str, eletes)))
    # plt.ylabel('generation')
    # plt.xlabel('number of elete selection')
    # plt.show()

    crossover_prob = [0.9, 0.7, 0.5, 0.3, 0.1]
    hist = [[], [], [], [], []]

    queen_num = 7
    max_fitness = (queen_num*(queen_num-1))/2 

    for j, s in enumerate(crossover_prob):
        pop = Population(length_gene=queen_num, crossover_prob=s, mutation_prob=0.3, population=100, num_elete_selection=10, print_step=False)

        for i in range(20):
            while pop.max < max_fitness:
                pop.step()
                # print(pop.max)
            hist[j].append(pop.generation)
            print(pop.max, max(pop.genes).gene)
            pop.reset()
    
    plt.boxplot(hist, labels=list(map(str, crossover_prob)))
    plt.ylabel('generation')
    plt.xlabel('crossover probability')
    plt.show()
    
    # mutation_prob = [0.9, 0.5, 0.3, 0.1, 0.01, 0.001]
    # hist = [[], [], [], [], [], []]

    # queen_num = 7
    # max_fitness = (queen_num*(queen_num-1))/2 

    # for j, s in enumerate(mutation_prob):
    #     pop = Population(length_gene=queen_num, crossover_prob=0.4, mutation_prob=s, population=100, num_elete_selection=10, print_step=False)

    #     for i in range(50):
    #         while pop.max < max_fitness:
    #             pop.step()
    #             # print(pop.max)
    #         hist[j].append(pop.generation)
    #         print(pop.generation, max(pop.genes).gene)
    #         pop.reset()
    
    # plt.boxplot(hist, labels=list(map(str, mutation_prob)))
    # plt.ylabel('generation')
    # plt.xlabel('mutation probability')
    # plt.show()