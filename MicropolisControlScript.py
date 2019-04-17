import numpy as np
import random
from gym_micropolis.envs.corecontrol import MicropolisControl
from gym_micropolis.envs.tilemap import TileMap,zoneSize
from gi.repository import Gtk as gtk
from time import sleep
import json

# We only take a few buildings at first
buildings = ['NuclearPowerPlant','Residential','Commercial','Industrial','Road']
class Quimby :

    def __init__(self,map_h,map_w,gen_size,n_population,n_steps_evaluation):
        """
        :param map_h: Maximal height of the map
        :param map_w: Max width
        :param gen_size: lenght of the genome
        :param n_population: number of elements in population
        :param n_steps_evaluation: number of steps for evaluation
        """
        self.map_h = map_h
        self.map_w = map_w
        self.gen_size = gen_size
        self.n_population = n_population
        self.n_steps_evaluation = n_steps_evaluation

        # List of available buildings
        buildings = ['NuclearPowerPlant', 'Residential', 'Commercial', 'Industrial', 'Road']

        # Generate random first generation
        self.genomes = [[[np.random.randint(1, map_h), np.random.randint(1, map_w), random.choice(buildings)] for i in range(gen_size)]for p in range(n_population)]
        self.genomes = self.clean_genomes(self, self.genomes)


    def mate(self,g1,g2) :
        """
        Take 2 genomes and build a new one using crossing
        """
        r = np.random.randint(len(g1))
        return(g1[0:r] + g2[r:])

    def mutate_genome(self,g1) :
        """
         Take a genome and apply a random transformation
        """
        p = np.random.randint(len(g1))
        g1[p] = [np.random.randint(1,self.map_h),np.random.randint(1,self.map_w),random.choice(buildings)]

    def build_city(self,g1,display) :
        """
        Build a city for a given genome an return it
        """
        m = MicropolisControl(self.map_h,self.map_w,display=display)
        m.clearMap()
        # Build
        for b in g1 :
            m.doBotTool(b[0],b[1],b[2])
        return m

    def evaluate(self,g1,display=False) :
        """
            Build and evaluate a city
        """
        m = self.build_city(g1,display)
        for i in range(self.n_steps_evaluation):
            m.engine.simTick()
            if display: m.render()
        if display:
            m.win1.destroy()
        pop = m.engine.cityPop
        m.close()
        return pop

    def save_populations(self,path='population_save.json'):
        """
        Save the population in a json file
        :return:
        """
        a = [{'Population': self.populations[i], 'City': self.genomes[i]} for i in range(self.n_population)]
        with open(path, 'w') as fp:
            json.dump(a, fp)


    def get_best_gennome(self):
        return self.genomes[np.argmax(self.populations)]


    def evaluate_all(self):
        """
        Launch the Evaluation for all the genome
        :return:
        """
        self.populations = []
        for g in self.genomes:
            self.populations.append(self.evaluate(g, display=False))

    def overlaps(self, tool_a, tool_b):
        '''
        tool_a and b are like [9, 9, 'Industrial']
        '''
        size_a = zoneSize[tool_a[-1]]
        size_b = zoneSize[tool_b[-1]]

        # Top left corners
        ax0, ay0 = tool_a[0:2]
        if size_a > 1:
            ax0 += 1
            ay0 += 1
        bx0, by0 = tool_b[0:2]
        if size_b > 1:
            bx0 += 1
            by0 += 1
            # Bottom right corners
        ax1, ay1 = ax0 + size_a - 1, ay0 + size_a - 1
        bx1, by1 = bx0 + size_b - 1, by0 + size_b - 1

        # intersection theorem
        if ay0 > by1 or ay1 < by0:
            return False
        if ax0 > bx1 or ax1 < bx0:
            return False

        return True

    def clean_chromosome(self, chromosome):
        '''
        removes useless tools in chromosome
        :param chromosome: the chromosome to clean
        :return:
        '''
        nc = []
        for gene in reversed(chromosome):
            if sum([self.overlaps(gene, tool) for tool in nc]) == 0:
                nc.insert(0, gene)
        return nc

    def clean_genomes(self, genomes):
        '''
        removes useless tools in chromosome
        :param chromosome: the chromosome to clean
        :return:
        '''
        genomes = [self.clean_genome(c) for c in genomes]
        return genomes

    def crossover(self, couple, rotation_index=0):
        '''
        parents is a list of the selected parents for reproduction
        rotation_index selects the first parents in the list for reproduction
        outputs a list of how to merge the parents
        '''
        nb_parents = len(couple)
        p2 = [len(x) - 1 for x in couple]
        child = []

        # We want to scroll until parents are empty
        while p2 != [-1] * nb_parents:
            if p2[rotation_index] != -1:
                # Select the parents next tool
                gene = couple[rotation_index][p2[rotation_index]]
                p2[rotation_index] -= 1

                touching = sum([self.overlaps(gene, tool) for tool in child])

                # if it can be added add it to children indexs
                if not touching and gene:
                    child.insert(0, gene)

            # We'll start the next loop with the next parent
            if not touching or p2[rotation_index] == -1:
                rotation_index = (rotation_index + 1) % nb_parents

        return child
