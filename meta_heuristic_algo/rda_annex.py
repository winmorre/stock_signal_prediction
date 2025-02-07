import math
import random

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization,
    Activation, Add, MaxPooling1D, Flatten, Dense, LSTM,
    Bidirectional, Attention, Concatenate, Dropout
)
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score

from _utilities import call_counter, compute_fitness, sort_agents


def sigmoid(val):
    if val < 0:
        return 1 - 1 / (1 + np.exp(val))
    else:
        return 1 / (1 + np.exp(-val))


def create_cnn_bilstm_model(input_shape, conv_filters, kernel_size, pool_size, dropout_rate, lstm_units, dense_units,
                            learning_rate):
    inputs = Input(shape=input_shape)
    model = Sequential()
    # CNN part
    x = Conv1D(filters=conv_filters, kernel_size=kernel_size, padding="same", activate="relu")(inputs)
    model.add(x)
    x = BatchNormalization()(x)
    model.add(x)
    x = MaxPooling1D(pool_size=pool_size)(x)
    model.add(x)

    # BiLSTM part
    x = Bidirectional(LSTM(units=lstm_units, return_sequences=True))(x)
    model.add(x)
    x = Dropout(rate=dropout_rate)(x)
    model.add(x)
    x = Flatten()(x)
    model.add(x)
    # Dense part
    x = Dense(units=dense_units, activation="softmax")(x)
    model.add(x)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


class RDA:
    def __init__(self, pop_size, num_generations, lower_bounds, upper_bounds, xtrain, xtest, ytrain, ytest):
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.num_params = len(lower_bounds)
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xtest = xtest
        self.ytest = ytest
        self.gamma = 0.5
        self.alpha = 0.2
        self.beta = 0.1
        self.num_hinds = 0
        self.num_coms = 0
        self.num_stags = 0
        self.num_harems = []
        self.weight_acc = 0.9
        self.stags = []
        self.hinds = []
        self.coms = []
        self.harems = []
        self.UB = self.num_params
        self.LB = -self.num_params
        self.obj_function = call_counter(compute_fitness(self.weight_acc))
        self.males = []
        self.num_males = 0
        self.females = []
        self.num_females = 0
        self.num_features = self.xtrain.shape[1]
        self.population_pool = []

    def initialize_population(self):
        population = np.random.uniform(self.lower_bounds, self.upper_bounds, (self.pop_size, self.num_params))
        return population

    def select_stags_and_hinds(self, population, fitness_scores):
        sorted_indices = np.argsort(fitness_scores)[::-1]
        stags = population[sorted_indices[:self.pop_size // 2]]
        hinds = population[sorted_indices[self.pop_size // 2:]]
        return stags, hinds

    def assign_hinds_to_stags(self, stags, hinds):
        harems = {i: [] for i in range(len(stags))}
        for i, hind in enumerate(hinds):
            stag_index = i % len(stags)
            harems[stag_index].append(hind)
        return harems

    def mutate(self, individual):
        mutation_prob = 0.1
        for i in range(self.num_params):
            if np.random.rand() < mutation_prob:
                individual[i] = np.random.uniform(self.lower_bounds[i], self.upper_bounds[i])

        return individual

    def fitness(self, individual):
        conv_filters = int(individual[0])
        kernel_size = int(individual[1])
        pool_size = int(individual[2])
        lstm_units = int(individual[3])
        dropout_rate = individual[4]
        learning_rate = individual[5]
        input_shape = self.xtrain.shape[1]
        num_classes = self.ytrain.shape[1]

        model = create_cnn_bilstm_model(
            input_shape=input_shape,
            conv_filters=conv_filters,
            kernel_size=kernel_size,
            pool_size=pool_size,
            lstm_units=lstm_units,
            dropout_rate=dropout_rate,
            dense_units=num_classes,
            learning_rate=learning_rate,
        )
        model.fit(self.xtrain, self.ytrain, epochs=10, batch_size=32, verbose=0)
        y_pred = model.predict(self.xtest)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.ytest, axis=1)
        accuracy = accuracy_score(y_true, y_pred_classes)
        return accuracy

    def roaring(self, stags):
        for i in range(len(stags)):
            local_search_step = np.random.uniform(-0.05, 0.05, self.num_params)
            stags[i] = stags[i] + local_search_step
            stags[i] = np.clip(stags[i], self.lower_bounds, self.upper_bounds)
        return stags

    def move_hinds(self, hinds, best_stag):
        for i in range(len(hinds)):
            hind = hinds[i]
            move_step = np.random.uniform(-0.1, 0.1, self.num_params)
            hind += move_step * (best_stag - hinds[i])
            hind = np.clip(hind, self.lower_bounds, self.upper_bounds)
            hinds[i] = hind
        self.hinds = hinds
        return hinds

    def fight(self, stags, fitness_scores):
        for i in range(len(stags)):
            local_search_step = np.random.uniform(-0.1, 0.1, self.num_params)
            new_stag = stags[i] + local_search_step
            new_stag = np.clip(new_stag, self.lower_bounds, self.upper_bounds)
            new_fitness = self.fitness_function(new_stag)
            if new_fitness > fitness_scores[i]:
                stags[i] = new_stag
                fitness_scores[i] = new_fitness
        return stags, fitness_scores

    def form_harems(self):
        com_fitness = self.fitness(self.coms)
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

    def crossover(self, parent1, parent2):
        crossover_point = np.random.randint(1, self.num_params)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
        return child

    def mate(self, population):
        # mating of commander with hinds in his harem
        num_harem_mate = [int(x * self.alpha) for x in self.num_harems]
        self.population_pool = list(population)
        for i in range(self.num_coms):
            np.random.shuffle(self.harems[i])
            for j in range(num_harem_mate[i]):
                r = np.random.random()
                offspring = (self.coms[i] + self.harems[i][j]) / 2 + (self.UB - self.LB) * r
                for k in range(self.num_features):
                    trans_value = sigmoid(offspring[k])
                    if np.random.random() < trans_value:
                        offspring[k] = 1
                    else:
                        offspring[k] = 0
                offspring = self.mutate(list(offspring))
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
                            trans_value = sigmoid(offspring[o])
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
                        trans_value = sigmoid(offspring[j])
                        if np.random.random() < trans_value:
                            offspring[j] = 1
                        else:
                            offspring[j] = 0
                    self.population_pool.append(list(offspring))

                    break

    def fight_annex(self, stags, fitness_scores):
        for i in range(len(stags)):
            local_search_step = np.random.uniform(-0.1, 0.1, self.num_params)
            new_stag = stags[i] + local_search_step
            new_stag = np.clip(new_stag, self.lower_bounds, self.upper_bounds)
            new_fitness = self.fitness(new_stag)
            if new_fitness > fitness_scores[i]:
                stags[i] = new_stag
                fitness_scores[i] = new_fitness

        return stags, fitness_scores

    def evaluate_fitness(self, population):
        return np.array([self.fitness(ind) for ind in population])

    def optimize(self):
        population = self.initialize_population()
        best_individual = None
        best_fitness = -np.inf

        for generation in range(self.num_generations):
            fitness_scores = self.evaluate_fitness(population)
            if np.max(fitness_scores) > best_fitness:
                best_fitness = np.max(fitness_scores)
                best_individual = population[np.argmax(fitness_scores)]

            stags, hinds = self.select_stags_and_hinds(population, fitness_scores)
            harems = self.assign_hinds_to_stags(stags, hinds)
            stags = self.roaring(stags)
            stag_fitness_score = self.evaluate_fitness(stags)
            self.fight(stags, stag_fitness_score)
            self.form_harems()

            new_population = []

            for stag_index, stag in enumerate(stags):
                harem = harems[stag_index]
                for hind in harem:
                    child = self.crossover(stag, hind)
                    child = self.mutate(child)
                    new_population.append(child)

            hinds = self.move_hinds(hinds, best_individual)
            new_population.extend(stags)
            new_population.extend(hinds)
            population = np.array(new_population)

            print(f'Generation {generation + 1}: Best Fitness = {best_fitness}')

        return best_individual, best_fitness

if __name__ == "__main__":
    lower_bounds = [0, 0, 0]
    upper_bounds = [10, 10, 10]
    rda = RDA(pop_size=20, num_generations=50, lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    best_individual, best_fitness = rda.optimize()