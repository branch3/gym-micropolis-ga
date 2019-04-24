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

    def __init__(self, map_h=30, map_w=30, chromosome_len=100, n_population=100, n_steps_evaluation=39):
        """
        :param map_h: Maximal height of the map
        :param map_w: Max width
        :param chromosome_len: lenght of the genome
        :param n_population: number of elements in population
        :param n_steps_evaluation: number of steps for evaluation (39 = 5 years)
        """
        self.padding = 0
        self.map_h = map_h
        self.map_w = map_w
        self.chromosome_len = chromosome_len
        self.n_population = n_population
        self.n_steps_evaluation = n_steps_evaluation
        self.pop_progression = []
        self.populations = []

        # List of available buildings
        self.buildings = ['NuclearPowerPlant', 'Residential', 'Commercial', 'Industrial', 'Road']

        # Generate random first generation
        self.genomes = [[self.random_gene() for i in range(chromosome_len)] for p in range(n_population)]
        self.genomes = self.clean_genomes(self.genomes)

        # Parameters for GA by default:
        self.eval = "last"
        self.crossover_method = "split"
        self.nb_generations = 10
        self.mutation_rate = 0.2
        self.p_selection = 0.2
        self.couple_size = 2
        self.nb_child = 8
        self.nb_lucky = 0


    def random_gene(self, safe_mode=True):
        '''
        returns a random gene
        :param map_w:
        :param map_h:
        :param buildings:
        :return:
        '''
        # Let's pick a building randomly
        building = random.choice(self.buildings)
        # We pick coordinates (might change later depending on safe mode)
        x, y = (random.randint(0, self.map_w-1), random.randint(0, self.map_h-1))

        tool = [x, y, building]

        if safe_mode:
            # Get safe coordinates for this building
            x0, y0, x1, y1 = self.get_zone_edges(tool)
            safe_x_lowerbound, safe_y_lowerbound = [x-x0, y-y0]
            safe_x_higherbound, safe_y_higherbound = [self.map_w-(x1-x), self.map_h-(y1-y)]
            # Pick new safe positions
            x, y = (random.randint(safe_x_lowerbound, safe_x_higherbound - 1), random.randint(safe_y_lowerbound, safe_y_higherbound - 1))
            # update the tool
            tool = [x, y, building]

        return tool

    def mutate_genomes(self,mutation_rate=0.2):
        """
        mutations are randomly adding new random genes at the end of randomly selected chromosomes.
        Mutations makes able to create alleles that didn't exist before
        """
        nb_mutations = int(round(mutation_rate*self.n_population))

        for m in range(nb_mutations):
            self.genomes[random.randint(0, self.n_population-1)].append(self.random_gene())

        return

    def build_city(self,city,display=False) :
        """
        Build a city for a given genome an return it
        """
        m = MicropolisControl(self.map_h,self.map_w, PADDING=self.padding, display=display)
        m.clearMap()
        # Build
        for tool in city :
            m.doBotTool(tool[0],tool[1],tool[2])
        return m

    def evaluate(self,g1,display=False) :
        """
            Build and evaluate a city
        """
        m = self.build_city(g1,display)
        pop = 0
        for i in range(self.n_steps_evaluation):
            m.engine.simTick()
            pop += m.engine.cityPop
            if display: m.render()

        if self.eval=="last":
            pop = m.engine.cityPop
        if self.eval=="average":
            pop = int(round(pop/self.n_steps_evaluation))
        if self.eval=="score":
            pop = m.engine.cityScore

        if display:
            m.win1.destroy()
        m.close()

        return pop

    def save_population(self,path='population_save.json'):
        """
        Save the population in a json file
        :return:
        """
        # sort the genomes inverse order
        self.populations,self.genomes = zip(*sorted(zip(self.populations, self.genomes), reverse=True))

        a = [{'Population': self.populations[i], 'City': self.genomes[i]} for i in range(self.n_population)]
        with open(path, 'w') as fp:
            json.dump(a, fp)
        return


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
        return

    def get_zone_edges(self, tool):
        '''
        returns the top left and bottom right corners coordinates of the tool zone
        :param tool: a tool array such as [1,3,'Residential']
        :return: [x0,y0,x1,y1]
        '''
        size = zoneSize[tool[-1]]

        # Top left corners
        x0, y0 = tool[0:2]

        if size == 1:
            return [x0, y0, x0 + size - 1, y0 + size - 1]
        else:
            return [x0-1, y0-1, x0-1 + size - 1, y0-1 + size - 1]

    def overlaps(self, tool_a, tool_b):
        '''
        tool_a and b are like [9, 9, 'Industrial']
        '''

        ax0, ay0, ax1, ay1 = self.get_zone_edges(tool_a)
        bx0, by0, bx1, by1 = self.get_zone_edges(tool_b)

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
        genomes = [self.clean_chromosome(c) for c in genomes]
        return genomes

    def crossover_switch(self, couple):
        '''
        :param couple is a list of the selected parents for reproduction
        :return: child
        '''
        nb_parents = len(couple)
        # Which parent to start
        rotation_index = random.randint(0, nb_parents-1)

        # Stores the progression in the parents (we progress backwards)
        couple_idx = [len(x) - 1 for x in couple]

        child = []

        # We want to scroll until parents are empty
        while couple_idx != [-1] * nb_parents:
            if couple_idx[rotation_index] != -1:
                # Select the parents next tool
                gene = couple[rotation_index][couple_idx[rotation_index]]
                couple_idx[rotation_index] -= 1

                touching = sum([self.overlaps(gene, tool) for tool in child])

                # if it can be added add it to children indexs
                if not touching and gene:
                    child.insert(0, gene)

            # We'll start the next loop with the next parent
            if not touching or couple_idx[rotation_index] == -1:
                rotation_index = (rotation_index + 1) % nb_parents

        return child

    def crossover_split(self, couple, nb_splits=1):
        '''
        :param couple is a list of the selected parents for reproduction
        :param nb_splits is how many splits are made
        :return: child
        '''
        nb_parents = len(couple)
        # Which parent to start
        rotation_index = random.randint(0, nb_parents-1)

        # We clean the parents chromosome (if it hasn't been done before)
        # couple = self.clean_genomes(couple)

        # We sort the parents chromosomes
        couple = [sorted(c, key=lambda gene: gene[0]) for c in couple]

        # We select random split indexs for each parent
        idxs = []
        for c in couple :
            # We randomly pick split indexes and sort them for each parent
            t = sorted([random.randint(0, len(c)-1) for i in range(nb_splits)])
            t.append(len(c))
            t.insert(0,0)
            idxs.append(t)
        # idxs = [[0,len(c)] for c in couple]
        # idxs_computed = [sorted([random.randint(0,len(c)-1) for i in range(nb_splits)]) for c in couple]

        # We add the pieces into the child
        child = []
        i = 0
        child.extend(couple[rotation_index][idxs[rotation_index][i]:idxs[rotation_index][i + 1]])
        while i < nb_splits:
            i = i+1
            rotation_index = (rotation_index + 1) % nb_parents
            child.extend(couple[rotation_index][idxs[rotation_index][i]:idxs[rotation_index][i + 1]])

        # We clean the child
        return self.clean_chromosome(child)
        # return child

    def mate(self, parents, couple_size=2, method="split", nb_splits=1):
        '''
        updates the genome in Quimby with the new generation generated on the parents using the switch crossover
        :param parents: an array of the possible parents selected for reproduction
        :param method : either "split" or "switch" for the crossover method
        :return:
        '''
        # Shuffle the parents
        random.shuffle(parents)

        children = []

        while len(children) < self.n_population:
            # Let's pick random parents for our couple
            p_idx = [np.random.randint(0, len(parents)) for _ in range(couple_size)]
            # if all parents in the couple are different
            if len(p_idx) == len(set(p_idx)):
                couple = [parents[i] for i in p_idx]
                if method == "switch":
                    children.append(self.crossover_switch(couple))
                else:
                    children.append(self.crossover_split(couple, nb_splits=nb_splits))

        self.genomes = children
        return


    def ga(self, nb_generations=10, mutation_rate=0.2, p_selection=0.2, eval="last", couple_size=2, nb_lucky=0, nb_child=8, crossover="split", nb_splits=1):
        '''
        Genetic Algorithm implementation
        :param nb_generations: how many generations does the algorithm goes through before stopping
        :param nb_parents: the amount of good parents selected for reproduction
        :param nb_lucky: the amount of lucky few that will be randomly selected for reproduction
        :param nb_child: how many child per couples
        :param couple_size: how many parents per couple
        :return:
        '''
        self.eval = eval
        self.nb_generations = nb_generations
        self.mutation_rate = mutation_rate
        self.p_selection = p_selection
        self.couple_size = couple_size
        self.crossover_method = crossover
        self.nb_child = nb_child
        self.nb_lucky = nb_lucky
        self.couple_size = couple_size

        # Run fitness function on initial genome
        self.evaluate_all()
        self.pop_progression.append(np.max(self.populations))

        # Loop through generations
        for i in range(nb_generations):
            # Sort genomes by score, the best up top
            ranked_genomes = [x for _, x in sorted(zip(self.populations, self.genomes), reverse=True)]
            # Select only the top for the parents
            parents = ranked_genomes[:int(round(len(ranked_genomes) * p_selection))]

            # Update genome with new generation
            self.mate(parents=parents, couple_size=couple_size, method=crossover, nb_splits=nb_splits)

            # Apply mutation to new genome
            self.mutate_genomes(mutation_rate)
            # Run fitness function on new genome
            self.evaluate_all()
            self.pop_progression.append(np.max(self.populations))

        # Sort genome
        self.sort_genome()

        return

    def sort_genome(self):
        self.populations, self.genomes = zip(*sorted(zip(self.populations, self.genomes), reverse=True))
        return

    def print_param(self):
        print("- "*50)
        print("mapSize:"+str(self.map_h)+", chromosome_len:"+str(self.chromosome_len)+", n_population:"+str(self.n_population)+", n_steps_evaluation:"+str(self.n_steps_evaluation)+", nb_generations:" + str(self.nb_generations) + ", eval:" + str(self.eval))
        print("crossover:" + str(self.crossover_method) + ", mutation_rate:" + str(self.mutation_rate) + ", p_selection:" + str(self.p_selection) + ", couple_size:" + str(self.couple_size))
        print("best score in last gen = "+str(self.pop_progression[-1]))

        return

