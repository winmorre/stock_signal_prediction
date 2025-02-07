import time

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import random, math
from _utilities import call_counter, compute_fitness, sort_agents, initialize, compute_accuracy


def sigmoid(val):
    if val < 0:
        return 1 - 1 / (1 + np.exp(val))
    else:
        return 1 / (1 + np.exp(-val))


def v_func(val):
    return abs(val / (np.sqrt(1 + val * val)))


def u_func(val, alpha=2, beta=1.5):
    return abs(alpha * np.power(abs(val), beta))


def get_trans_function(shape):
    if (shape.lower() == 's'):
        return sigmoid

    elif (shape.lower() == 'v'):
        return v_func

    elif (shape.lower() == 'u'):
        return u_func

    else:
        print('\n[Error!] We don\'t currently support {}-shaped transfer functions...\n'.format(shape))
        exit(1)


class RDA:
    def __init__(self, num_agents, max_iter, xtrain, ytrain, xtest=None, ytest=None,
                 save_conv_graph=False, seed=0, default_model=None, verbose=True):
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.save_conv_graph = save_conv_graph
        self.default_model = default_model
        self.verbose = verbose
        self.seed = seed

        self.trans_func = sigmoid
        self.num_males = None
        self.num_hinds = None
        self.num_stags = None
        self.num_harems = None
        self.num_coms = None
        self.males = None
        self.stags = None
        self.hinds = None
        self.harems = None
        self.coms = None
        self.population_pool = None
        self.gamma = 0.5
        self.alpha = 0.2
        self.beta = 0.1
        self.UB = 10
        self.LB = -10
        self.num_features = None
        self.weight_acc = 0.9
        self.obj_function = call_counter(compute_fitness(self.weight_acc))
        self.population = initialize(num_agents=self.num_agents, num_features=self.num_features, seed=self.seed)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.cur_iter = 0
        self.seed = 0
        self.accuracy = None
        self.Leader_agent = None
        self.Leader_fitness = None
        self.val_size = 0.30
        self.max_evals = 20
        self.start_time = None
        self.end_time = None

    def initialize(self):
        self.val_size = 0.30
        self.obj_function = call_counter(compute_fitness(self.weight_acc))
        self.start_time = time.time()
        np.random.seed(self.seed)

        self.num_features = self.xtrain.shape[1]
        self.population = initialize(num_agents=self.num_agents, num_features=self.num_features, seed=self.seed)
        self.fitness = self.obj_function(self.population, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.accuracy = compute_accuracy(agents=self.population, xtrain=self.xtrain, ytrain=self.ytrain,
                                         xtest=self.xtest, ytest=self.ytest)
        self.Leader_agent, self.Leader_fitness = self.population[0], self.fitness[0]

    def roar(self):
        for i in range(self.num_males):
            r1 = np.random.random()  # r1 is a random number in [0,1]
            r2 = np.random.random()  # r2 is a random number in [0,1]
            r3 = np.random.random()  # r3 is a random number in [0,1]
            new_male = self.males[i].copy()

            if r3 >= 0.5:
                new_male += r1 * (((self.UB - self.LB) * r2) + self.LB)
            else:
                new_male -= r1 * (((self.UB - self.LB) * r2) + self.LB)

            for j in range(self.num_features):
                trans_value = sigmoid(new_male[j])
                if np.random.random() < trans_value:
                    new_male[j] = 1
                else:
                    new_male[j] = 0

            if self.obj_function(new_male, self.xtrain, self.ytrain, self.xtest, self.ytest) < self.obj_function(
                    self.males[i], self.xtrain, self.ytrain, self.xtest, self.ytest):
                self.males[i] = new_male

    def fight(self):
        # fight between male commanders and stags
        for i in range(self.num_coms):
            chosen_com = self.coms[i].copy()
            chosen_stag = random.choice(self.stags)
            r1 = np.random.random()
            r2 = np.random.random()

            new_male_1 = (chosen_com + chosen_stag) / 2 + r1 * (((self.UB - self.LB) * r2) + self.LB)
            new_male_2 = (chosen_com + chosen_stag) / 2 - r1 * (((self.UB - self.LB) * r2) + self.LB)

            for j in range(self.num_features):
                trans_value = self.trans_func(new_male_1[j])
                trans_value_2 = self.trans_func(new_male_2[j])
                if np.random.random() < trans_value:
                    new_male_1[j] = 1
                else:
                    new_male_1[j] = 0

                if np.random.random() < trans_value_2:
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

    def form_harems(self):
        # formation of harems
        com_fitness = self.obj_function(self.coms, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.coms, com_fitness = sort_agents(agents=self.coms, fitness=com_fitness)
        norm = np.linalg.norm(com_fitness)
        normal_fit = com_fitness / norm
        total = np.sum(normal_fit)
        power = normal_fit / total
        self.num_harems = [int(x * self.num_hinds) for x in power]
        max_harem_size = np.max(self.num_harems)
        self.harems = np.empty(shape=(self.num_coms, max_harem_size, self.num_features))
        random.shuffle(self.hinds)

        itr = 0
        for i in range(self.num_coms):
            harem_size = self.num_harems[i]
            for j in range(harem_size):
                self.harems[i][j] = self.hinds[itr]
                itr += 1

    def mate(self):
        # mating of commander with hinds in his harem
        num_harem_mate = [int(x * self.alpha) for x in self.num_harems]
        self.population_pool = list(self.population)
        for i in range(self.num_coms):
            np.random.shuffle(self.harems[i])
            for j in range(num_harem_mate[i]):
                r = np.random.random()
                offspring = (self.coms[i] + self.harems[i][j]) / 2 + (self.UB - self.LB) * r
                for k in range(self.num_features):
                    trans_value = self.trans_func(offspring[k])
                    if np.random.random() < trans_value:
                        offspring[k] = 1
                    else:
                        offspring[k] = 0
                self.population_pool.append(list(offspring))

                # if the number of commanders is greater than 1, inter-harem mating takes place
                if self.num_coms > 1:
                    k = i
                    while k == i:
                        k = random.choice(range(self.num_coms))

                    num_mate = int(self.num_harems[k] * self.beta)

                    random.shuffle(self.harems[k])

                    for k in range(num_mate):
                        r = np.random.random()
                        offspring = (self.coms[i] + self.harems[k][j]) / 2 + (self.UB - self.LB) * r
                        for o in range(self.num_features):
                            trans_value = self.trans_func(offspring[o])
                            if np.random.random() < trans_value:
                                offspring[o] = 1
                            else:
                                offspring[o] = 0
                        self.population_pool.append(list(offspring))

        # mating a stag with nearest hind
        for stag in self.stags:
            dist = np.zeros(self.num_hinds)
            for i in range(self.num_hinds):
                dist[i] = np.sqrt(np.sum((stag - self.hinds[i]) * (stag - self.hinds[i])))
            min_dist = np.min(dist)
            for i in range(self.num_hinds):
                distance = math.sqrt(np.sum((stag - self.hinds[i]) * (stag - self.hinds[i])))
                if distance == min_dist:
                    r = np.random.random()
                    offspring = (stag + self.hinds[i]) / 2 + (self.UB - self.LB) * r

                    # apply transformation function on offspring
                    for j in range(self.num_features):
                        trans_value = self.trans_func(offspring[j])
                        if np.random.random() < trans_value:
                            offspring[j] = 1
                        else:
                            offspring[j] = 0
                    self.population_pool.append(list(offspring))

                    break

    def select_next_generations(self):
        self.population_pool = np.array(self.population_pool)
        fitness = self.obj_function(self.population_pool, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.population_pool, fitness = sort_agents(agents=self.population_pool, fitness=fitness)
        maximum = sum([f for f in fitness])
        selection_probs = [f / maximum for f in fitness]
        indices = np.random.choice(len(self.population_pool), p=selection_probs, size=self.num_agents, replace=True)
        deer = self.population_pool[indices]
        return deer

    def next(self):
        self.print('\n================================================================================')
        self.print('                          Iteration - {}'.format(self.cur_iter + 1))
        self.print('================================================================================\n')

        self.num_males = int(0.25 * self.num_agents)
        self.num_hinds = self.num_agents - self.num_males
        self.males = self.population[:self.num_males, :]
        self.hinds = self.population[self.num_males:, :]

        self.roar()

        # selection of male commanders and stag
        self.num_coms = int(self.num_males * self.gamma)
        self.num_stags = self.num_males - self.num_coms
        self.coms = self.males[:self.num_coms, :]
        self.stags = self.males[self.num_coms:, :]

        self.fight()
        self.form_harems()
        self.mate()
        self.select_next_generations()
        self.cur_iter += 1

    def check_end(self):
        return (self.cur_iter >= self.max_iter) or (self.obj_function.cur_evals >= self.max_evals)

    def run(self):
        self.initialize()

        while not self.check_end():
            self.next()

        self.end_time = time.time()

        self.Leader_fitness = self.obj_function(self.Leader_agent, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.Leader_accuracy = compute_accuracy(self.Leader_agent, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.print('\n------------- Leader Agent ---------------')
        self.print('Fitness: {}'.format(self.Leader_fitness))
        self.print('Number of Features: {}'.format(int(np.sum(self.Leader_agent))))
        self.print('----------------------------------------\n')
        return self


