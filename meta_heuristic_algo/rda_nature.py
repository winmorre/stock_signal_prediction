import math
import random

import numpy as np

from _utilities import sigmoid, call_counter, compute_fitness, compute_accuracy, sort_agents, initialize


class RdaNature:
    def __init__(self, num_agents, lower_params, upper_params, max_iter, xtrain, xtest, ytrain, ytest, seed=0.0):
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.xtrain, self.xtest, self.ytrain, self.ytest = xtrain, xtest, ytrain, ytest
        self.seed = seed

        self.num_features = self.xtrain.shape[1]
        self.obj_function = compute_fitness(weight_acc=0.9)
        self.trans_function = sigmoid
        self.deer = []
        self.fitness = None
        self.accuracy = None
        self.Leader_agent = np.zeros((1, self.num_features))
        self.Leader_fitness = float("-inf")
        self.Leader_accuracy = float("-inf")
        self.lower_params = lower_params
        self.upper_params = upper_params
        self.LB = len(self.lower_params)
        self.UB = len(self.upper_params)
        self.gamma = 0.5
        self.alpha = 0.2
        self.beta = 0.1
        self.num_males = int(0.25 * num_agents)
        self.num_hinds = self.num_agents - self.num_males
        self.males = []
        self.hinds = []
        self.num_coms = int(self.num_males * self.gamma)
        self.num_stags = self.num_males - self.num_coms
        self.coms = []
        self.stags = []
        self.population_pool = None
        self.num_harems = []
        self.harem = []
        self.cur_iter = 0
        self.max_evals = np.float64("inf")

    def initialize(self):
        self.deer = initialize(num_agents=self.num_agents, num_features=self.num_features, seed=self.seed)
        self.population_pool = list(self.deer)
        self.males = self.deer[:self.num_males, :]
        self.hinds = self.deer[self.num_males:, :]
        self.fitness = self.obj_function(self.deer, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.deer, self.fitness = sort_agents(agents=self.deer, fitness=self.fitness)
        self.accuracy = compute_accuracy(agents=self.deer, xtrain=self.xtrain, ytrain=self.ytrain,
                                         xtest=self.xtest, ytest=self.ytest)
        self.Leader_agent, self.Leader_fitness = self.deer[0], self.fitness[0]

    def male_deer_roaring(self):
        for i in range(self.num_males):
            r1 = np.random.random()  # r is a random number in [0, 1]
            r2 = np.random.random()
            r3 = np.random.random()
            new_male = self.males[i].copy()
            if r3 >= 0.5:  # Eq. (3)
                new_male += r1 * (((self.UB - self.LB) * r2) + self.LB)
            else:
                new_male -= r1 * (((self.UB - self.LB) * r2) + self.LB)

            # apply transformation function on the new male
            for j in range(self.num_features):
                trans_value = self.trans_function(new_male[j])
                if np.random.random() < trans_value:
                    new_male[j] = 1
                else:
                    new_male[j] = 0

            if self.obj_function(new_male, self.xtrain, self.ytrain, self.xtest, self.ytest) < self.obj_function(
                    self.males[i],
                    self.xtrain, self.ytrain, self.xtest, self.ytest):
                self.males[i] = new_male

    def select_male_commanders_and_coms(self):
        self.coms = self.males[:self.num_coms, :]
        self.stags = self.males[self.num_coms:, :]

    def commanders_and_stags_fight(self):
        for i in range(self.num_coms):
            chosen_com = self.coms[i].copy()
            chosen_stag = random.choice(self.stags)
            r1 = np.random.random()
            r2 = np.random.random()
            new_male_1 = (chosen_com + chosen_stag) / 2 + r1 * (((self.UB - self.LB) * r2) + self.LB)  # Eq. (6)
            new_male_2 = (chosen_com + chosen_stag) / 2 - r1 * (((self.UB - self.LB) * r2) + self.LB)  # Eq. (7)

            # apply transformation function on new_male_1
            for j in range(self.num_features):
                trans_value = self.trans_function(new_male_1[j])
                if np.random.random() < trans_value:
                    new_male_1[j] = 1
                else:
                    new_male_1[j] = 0

            # apply transformation function on new_male_2
            for j in range(self.num_features):
                trans_value = self.trans_function(new_male_2[j])
                if np.random.random() < trans_value:
                    new_male_2[j] = 1
                else:
                    new_male_2[j] = 0

            fitness = np.zeros(4)
            fitness[0] = self.obj_function(chosen_com, self.xtrain, self.ytrain, self.xtest, self.ytest)
            fitness[1] = self.obj_function(chosen_stag, self.xtrain, self.ytrain, self.xtest, self.ytest)
            fitness[2] = self.obj_function(new_male_1, self.xtrain, self.ytrain, self.xtest, self.ytest)
            fitness[3] = self.obj_function(new_male_2, self.xtrain, self.ytrain, self.xtest, self.ytest)

            bestfit = np.max(fitness)
            if fitness[0] < fitness[1] == bestfit:
                self.coms[i] = chosen_stag.copy()
            elif fitness[0] < fitness[2] == bestfit:
                self.coms[i] = new_male_1.copy()
            elif fitness[0] < fitness[3] == bestfit:
                self.coms[i] = new_male_2.copy()

    def harems_formation(self):
        self.fitness = self.obj_function(self.coms, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.coms, self.fitness = sort_agents(self.coms, fitness=self.fitness)
        norm = np.linalg.norm(self.fitness)
        normal_fit = self.fitness / norm
        total = np.sum(normal_fit)
        power = normal_fit / total  # Eq. (9)
        self.num_harems = [int(x * self.num_hinds) for x in power]  # Eq.(10)
        max_harem_size = np.max(self.num_harems)
        self.harem = np.empty(shape=(self.num_coms, max_harem_size, self.num_features))
        random.shuffle(self.hinds)
        itr = 0
        for i in range(self.num_coms):
            harem_size = self.num_harems[i]
            for j in range(harem_size):
                self.harem[i][j] = self.hinds[itr]
                itr += 1

    def mate_commander_with_hinds(self):
        """
        Mating of commanders with hinds in his harem
        """
        num_harem_mate = [int(x * self.alpha) for x in self.num_harems]  # Eq. (11)
        self.population_pool = list(self.deer)

        for i in range(self.num_coms):
            random.shuffle(self.harem[i])
            for j in range(num_harem_mate[i]):
                r = np.random.random()  # r is a random number in [0, 1]
                offspring = (self.coms[i] + self.harem[i][j]) / 2 + (self.UB - self.LB) * r  # Eq. (12)

                # apply transformation function on offspring
                for j in range(self.num_features):
                    trans_value = self.trans_function(offspring[j])
                    if np.random.random() < trans_value:
                        offspring[j] = 1
                    else:
                        offspring[j] = 0
                self.population_pool.append(list(offspring))

                # if number of commanders is greater than 1, inter-harem mating takes place
                if self.num_coms > 1:
                    # mating of commander with hinds in another harem
                    k = i
                    while k == i:
                        k = random.choice(range(self.num_coms))

                    num_mate = int(self.num_harems[k] * self.beta)  # Eq. (13)

                    np.random.shuffle(self.harem[k])
                    for m in range(num_mate):
                        r = np.random.random()
                        offspring = (self.coms[i] + self.harem[k][m]) / 2 + (self.UB - self.LB) * r
                        # apply transformation function on offspring
                        for f in range(self.num_features):
                            trans_value = self.trans_function(offspring[f])
                            if np.random.random() < trans_value:
                                offspring[f] = 1
                            else:
                                offspring[f] = 0
                        self.population_pool.append(list(offspring))

        for stag in self.stags:
            dist = np.zeros(self.num_hinds)
            for i in range(self.num_hinds):
                dist[i] = math.sqrt(np.sum((stag - self.hinds[i]) * (stag - self.hinds[i])))
            min_dist = np.min(dist)
            for i in range(self.num_hinds):
                distance = math.sqrt(np.sum((stag - self.hinds[i]) * (stag - self.hinds[i])))  # Eq. (14)
                if distance == min_dist:
                    r = np.random.random()  # r is a random number in [0, 1]
                    offspring = (stag + self.hinds[i]) / 2 + (self.UB - self.LB) * r

                    # apply transformation function on offspring
                    for j in range(self.num_features):
                        trans_value = self.trans_function(offspring[j])
                        if np.random.random() < trans_value:
                            offspring[j] = 1
                        else:
                            offspring[j] = 0
                    self.population_pool.append(list(offspring))

                    break

        # selection of next generation
        self.population_pool = np.array(self.population_pool)
        self.fitness = self.obj_function(self.population_pool, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.population_pool, self.fitness = sort_agents(self.coms, fitness=self.fitness)
        maximum = sum([f for f in self.fitness])
        selection_probs = [f / maximum for f in self.fitness]
        indices = np.random.choice(len(self.population_pool), size=self.num_agents, replace=True, p=selection_probs)
        self.deer = self.population_pool[indices]

    def check_end(self):
        return (self.cur_iter >= self.max_iter) or (self.obj_function.cur_evals >= self.max_evals)

    def post_processing(self):
        self.fitness = self.obj_function(self.population, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.deer, self.fitness = sort_agents(agents=self.deer, fitness=self.fitness)
        self.accuracy = compute_accuracy(agents=self.deer, xtrain=self.xtrain, ytrain=self.ytrain,
                                         xtest=self.xtest, ytest=self.ytest)
        if self.fitness[0] > self.Leader_fitness:
            self.Leader_fitness = self.fitness[0]
            self.Leader_agent = self.deer[0, :]
            self.Leader_accuracy = self.accuracy[0]

    def run(self):
        self.initialize()
        while not self.check_end():
            self.male_deer_roaring()
            self.select_male_commanders_and_coms()
            self.commanders_and_stags_fight()
            self.harems_formation()
            self.mate_commander_with_hinds()

            self.cur_iter += 1
