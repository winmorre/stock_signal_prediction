import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Bidirectional, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from _utilities import sigmoid, call_counter, compute_fitness, compute_accuracy, sort_agents, initialize

"""
Whale optimization approaches
"""


class WOA:
    def __init__(self, num_agents, max_iter, xdata, ydata, seed=0):
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.xdata = xdata
        self.ydata = ydata
        self.seed = seed
        self.num_features = 10

        self.trans_function = sigmoid
        self.weight = 1.0
        self.population = None
        self.cur_iter = 0
        self.max_evals = np.float64("inf")
        self.solution = None
        self.fitness = None
        self.accuracy = None
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.xdata, self.ydata, test_size=0.2,
                                                                            random_state=42)
        self.obj_function = call_counter(compute_fitness(self.weight))
        self.Leader_fitness = 0
        self.Leader_agent = 0
        self.Leader_accuracy = 0
        self.weight_acc = 0.9

    def initialize(self):
        self.num_features = self.xtrain.shape[1]
        self.population = initialize(num_agents=self.num_agents, num_features=self.num_features, seed=self.seed)
        self.fitness = self.obj_function(self.population, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.accuracy = compute_accuracy(agents=self.population, xtrain=self.xtrain, ytrain=self.ytrain,
                                         xtest=self.xtest, ytest=self.ytest)
        self.Leader_agent, self.Leader_fitness = self.population[0], self.fitness[0]

    def forage(self):
        a = 2 - self.cur_iter * (2 / self.max_iter)

        # update the position of each whale
        for i in range(self.num_agents):
            r = np.random.rand()
            A = (2 * a * r) - a
            C = 2 * r
            l = np.random.uniform(-1, 1)
            p = np.random.random()
            b = 1  # defines shape of the spiral

            if p < 0.5:
                # Shrinking Encircling mechanism
                if abs(A) >= 1:
                    rand_agent_index = np.random.randint(0, self.num_agents)
                    rand_agent = self.population[rand_agent_index, :]
                    mod_dist_rand_agent = abs(C * rand_agent - self.population[i, :])
                    self.population[i, :] = rand_agent - (A * mod_dist_rand_agent)
                else:
                    mod_dist_rand_agent = abs(C * self.Leader_agent - self.population[i, :])
                    self.population[i, :] = self.Leader_agent - (A * mod_dist_rand_agent)

            else:
                dist_leader = abs(self.Leader_agent - self.population[i, :])
                self.population[i, :] = dist_leader * np.exp(b * l) * np.cos(l * 2 * np.pi) + self.Leader_agent

            # apply transformation function on the updated whale
            for j in range(self.num_features):
                trans_value = self.trans_function(self.population[i, j])
                if np.random.rand() < trans_value:
                    self.population[i, j] = 1
                else:
                    self.population[i, j] = 0

    def check_end(self):
        return (self.cur_iter >= self.max_iter) or (self.obj_function.cur_evals >= self.max_evals)

    def post_processing(self):
        self.fitness = self.obj_function(self.population, self.xtrain, self.ytrain, self.xtest, self.ytest)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.accuracy = compute_accuracy(agents=self.population, xtrain=self.xtrain, ytrain=self.ytrain,
                                         xtest=self.xtest, ytest=self.ytest)
        if self.fitness[0] > self.Leader_fitness:
            self.Leader_fitness = self.fitness[0]
            self.Leader_agent = self.population[0, :]
            self.Leader_accuracy = self.accuracy[0]

    def next(self):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter + 1))
        print('================================================================================\n')

        self.forage()
        self.cur_iter += 1

    def run(self):
        self.initialize()
        while not self.check_end():
            self.next()
            self.post_processing()
