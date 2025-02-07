import numpy as np
import random
import math
import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Bidirectional, Dropout
from keras.optimizers import Adam


def sigmoid(val):
    if val < 0:
        return 1 - 1 / (1 + np.exp(val))

    return 1 / (1 + np.exp(-val))


def create_model(hyperparams, input_shape):
    model = Sequential()
    model.add(Conv1D(filters=int(hyperparams[0]), kernel_size=int(hyperparams[1]), activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=int(hyperparams[2])))
    model.add(Bidirectional(LSTM(units=int(hyperparams[3]))))
    model.add(Dropout(rate=hyperparams[4]))
    model.add(Dense(units=int(hyperparams[5]), activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer=Adam(learning_rate=hyperparams[6]), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def compute_fitness(hyperparams, xtrain, ytrain, xtest=None, ytest=None):
    xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)
    model = create_model(hyperparams, xtrain.shape[1:])
    model.fit(xtrain, ytrain, epochs=10, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(xval, yval, verbose=0)
    return -accuracy  # Negative because we want to minimize the objective function


def initialize(num_agents, num_features, seed):
    np.random.seed(seed)
    return np.random.uniform(low=[32, 3, 2, 50, 0.1, 10, 1e-5],
                             high=[128, 7, 4, 200, 0.5, 50, 1e-3],
                             size=(num_agents, num_features))


def sort_agents(agents, fitness):
    idx = np.argsort(fitness)
    return agents[idx], fitness[idx]


class RDA:
    def __init__(self, num_agents, max_iter, xtrain, ytrain, xtest=None, ytest=None, seed=0, verbose=True):
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
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
        self.upper_bound = [128, 7, 4, 200, 0.5, 50, 1e-3]
        self.lower_bound = [32, 3, 2, 50, 0.1, 10, 1e-5]
        self.UB = len(self.upper_bound)
        self.LB = len(self.lower_bound)
        self.num_features = 7
        self.obj_function = compute_fitness
        self.population = initialize(num_agents=self.num_agents, num_features=self.num_features, seed=self.seed)
        self.fitness = np.apply_along_axis(self.obj_function, 1, self.population, self.xtrain, self.ytrain, self.xtest,
                                           self.ytest)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.cur_iter = 0
        self.accuracy = None
        self.Leader_agent = self.population[0]
        self.Leader_fitness = self.fitness[0]
        self.start_time = None
        self.end_time = None
        self.fitness_pool = np.empty(shape=self.num_agents)

    def initialize(self):
        self.start_time = time.time()
        np.random.seed(self.seed)

        self.num_features = self.xtrain.shape[1]
        self.population = initialize(num_agents=self.num_agents, num_features=self.num_features, seed=self.seed)
        self.fitness = np.apply_along_axis(self.obj_function, 1, self.population, self.xtrain, self.ytrain, self.xtest,
                                           self.ytest)
        self.population, self.fitness = sort_agents(agents=self.population, fitness=self.fitness)
        self.Leader_agent, self.Leader_fitness = self.population[0], self.fitness[0]

    def roar(self):
        for i in range(self.num_males):
            r1 = np.random.random()
            r2 = np.random.random()
            r3 = np.random.random()
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
        com_fitness = np.apply_along_axis(self.obj_function, 1, self.coms, self.xtrain, self.ytrain, self.xtest,
                                          self.ytest)
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
                self.harems[i, j, :] = self.hinds[itr, :]
                itr += 1

    def mate(self):
        for i in range(self.num_coms):
            harem_size = self.num_harems[i]
            for j in range(harem_size):
                r1 = np.random.random()
                r2 = np.random.random()
                new_hind = self.harems[i, j, :].copy()

                if r2 >= 0.5:
                    new_hind += r1 * (((self.UB - self.LB) * r2) + self.LB)
                else:
                    new_hind -= r1 * (((self.UB - self.LB) * r2) + self.LB)

                for k in range(self.num_features):
                    trans_value = sigmoid(new_hind[k])
                    if np.random.random() < trans_value:
                        new_hind[k] = 1
                    else:
                        new_hind[k] = 0

                if self.obj_function(new_hind, self.xtrain, self.ytrain, self.xtest, self.ytest) < self.obj_function(
                        self.harems[i, j, :], self.xtrain, self.ytrain, self.xtest, self.ytest):
                    self.harems[i, j, :] = new_hind

    def select_next_generations(self):
        self.population_pool = np.empty(shape=(self.num_agents, self.num_features))


        for i in range(self.num_males):
            self.population_pool[i, :] = self.males[i]
            self.fitness_pool[i] = self.obj_function(self.males[i], self.xtrain, self.ytrain, self.xtest, self.ytest)

        itr = self.num_males
        for i in range(self.num_coms):
            harem_size = self.num_harems[i]
            for j in range(harem_size):
                self.population_pool[itr, :] = self.harems[i, j, :]
                self.fitness_pool[itr] = self.obj_function(self.harems[i, j, :], self.xtrain, self.ytrain, self.xtest,
                                                           self.ytest)
                itr += 1

        self.population_pool, self.fitness_pool = sort_agents(agents=self.population_pool, fitness=self.fitness_pool)

    def next(self):
        self.num_males = int(np.ceil(self.gamma * self.num_agents))
        self.num_hinds = self.num_agents - self.num_males
        self.num_stags = int(np.ceil(self.alpha * self.num_males))
        self.num_coms = self.num_males - self.num_stags

        self.males = np.empty(shape=(self.num_males, self.num_features))
        self.stags = np.empty(shape=(self.num_stags, self.num_features))
        self.coms = np.empty(shape=(self.num_coms, self.num_features))
        self.hinds = np.empty(shape=(self.num_hinds, self.num_features))

        for i in range(self.num_stags):
            self.stags[i, :] = self.population[i, :]

        for i in range(self.num_coms):
            self.coms[i, :] = self.population[self.num_stags + i, :]

        for i in range(self.num_hinds):
            self.hinds[i, :] = self.population[self.num_males + i, :]

        self.roar()
        self.fight()
        self.form_harems()
        self.mate()
        self.select_next_generations()

        self.Leader_agent = self.population_pool[0]
        self.Leader_fitness = self.fitness_pool[0]
        self.population = self.population_pool
        self.fitness = self.fitness_pool

    def check_end(self):
        if self.cur_iter >= self.max_iter:
            return True
        else:
            return False

    def run(self):
        self.initialize()

        while not self.check_end():
            if self.verbose:
                print(f'Iteration: {self.cur_iter}, Best Fitness: {self.Leader_fitness}')

            self.next()
            self.cur_iter += 1

        self.end_time = time.time()
        if self.verbose:
            print(f'Total iterations: {self.cur_iter}')
            print(f'Best solution: {self.Leader_agent}')
            print(f'Best solution fitness: {self.Leader_fitness}')
            print(f'Total execution time: {self.end_time - self.start_time}')


if __name__ == "__main__":
    rda = RDA(num_agents=30, max_iter=100, xtrain=xtrain, ytrain=ytrain, xtest=xtest, ytest=ytest)
    rda.run()
