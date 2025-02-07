import random
import math

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN


def _evaluate_solution(X, y, solution):
    # Use the selected feature to train a RandomForestClassifier
    X_selected = X.iloc[:, solution.astype(bool)]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model = KNN()
    # model = SVC(kernel='poly', random_state=42, coef0=2)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return accuracy


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def roar(num_males, males, UB, LB, dim, xdata, ydata):
    for i in range(num_males):
        r1 = np.random.random()
        r2 = np.random.random()
        r3 = np.random.random()
        new_male = males[i].copy()

        if r3 >= 0.5:
            new_male += r1 * (((UB - LB) * r2) + LB)
        else:
            new_male -= r1 * (((UB - LB) * r2) + LB)

        for j in range(dim):
            trans_value = sigmoid(new_male[j])
            if np.random.random() < trans_value:
                new_male[j] = 1
            else:
                new_male[j] = 0

        if _evaluate_solution(xdata, ydata, new_male) < _evaluate_solution(xdata, ydata, males[i]):
            males[i] = new_male

    return males


def sort_agents(agents, fitness):
    idx = np.argsort(fitness)
    return agents[idx], fitness[idx]


def form_harems(xdata, ydata, coms, num_coms, num_hinds, hinds, dim):
    # Form harems based on fitness
    fitness = np.apply_along_axis(_evaluate_solution, 1, xdata, ydata, coms)
    coms, fitness = sort_agents(coms, fitness=fitness)
    norm = np.linalg.norm(fitness)
    normal_fit = fitness / norm
    total = np.sum(normal_fit)
    power = normal_fit / total  # Eq. (9)
    num_harems = [int(x * num_hinds) for x in power]  # Eq.(10)
    max_harem_size = np.max(num_harems)
    harem = np.empty(shape=(num_coms, max_harem_size, dim))
    random.shuffle(hinds)
    itr = 0
    for i in range(num_coms):
        harem_size = num_harems[i]
        for j in range(harem_size):
            harem[i][j] = hinds[itr]
            itr += 1

    return coms, harem, num_harems


def cross_over():
    pass


def mutate():
    pass


def fight(xdata, ydata, dim, UB, LB, coms, num_coms, stags):
    for i in range(num_coms):
        chosen_com = coms[i].copy()
        chosen_stag = random.choice(stags)
        r1 = np.random.random()
        r2 = np.random.random()

        new_male_1 = (chosen_com + chosen_stag) / 2 + r1 * (((UB - LB) * r2) + LB)
        new_male_2 = (chosen_com + chosen_stag) / 2 - r1 * (((UB - LB) * r2) + LB)

        for j in range(dim):
            trans_value = sigmoid(new_male_1[j])
            trans_value_2 = sigmoid(new_male_2[j])
            if np.random.random() < trans_value:
                new_male_1[j] = 1
            else:
                new_male_1[j] = 0

            if np.random.random() < trans_value_2:
                new_male_2[j] = 1
            else:
                new_male_2[j] = 0

        fitness = np.zeros(4)
        fitness[0] = _evaluate_solution(xdata, ydata, chosen_com)
        fitness[1] = _evaluate_solution(xdata, ydata, chosen_stag)
        fitness[2] = _evaluate_solution(xdata, ydata, new_male_1)
        fitness[3] = _evaluate_solution(xdata, ydata, new_male_2)

        bestfit = np.max(fitness)
        if fitness[0] < fitness[1] == bestfit:
            coms[i] = chosen_stag.copy()
        elif fitness[0] < fitness[2] == bestfit:
            coms[i] = new_male_1.copy()
        elif fitness[0] < fitness[3] == bestfit:
            coms[i] = new_male_2.copy()
    return coms, stags


def select_male_commanders_and_coms(males, num_coms):
    coms = males[:num_coms, :]
    stags = males[num_coms:, :]
    return coms, stags


def _compute_fitness(agents, xdata, ydata):
    weight_acc = 0.9
    if len(agents.shape) == 1:
        # handles with zero-rank arrays
        agents = np.array([agents])

    weight_feat = 1 - weight_acc
    (num_agents, num_features) = agents.shape
    fitness = np.zeros(num_agents)
    feat = None
    for (i, agent) in enumerate(agents):
        acc = _evaluate_solution(xdata, ydata, agent)
        if np.sum(agents[i]) != 0:
            feat = (num_features - np.sum(agents[i])) / num_features
            fitness[i] = weight_acc * acc + weight_feat * feat

    return feat


def rda_gpt(xdata, ydata, lb, ub, dim, pop_size=30, max_iter=100):
    # pop_size= num of agents, dim= num_features
    gamma = 0.5
    alpha = 0.2
    beta = 0.1
    population = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb
    fitness = np.array([_evaluate_solution(xdata, ydata, ind) for ind in population])
    Leader_accuracy = float("-inf")
    Leader_agent, Leader_fitness = population[0], fitness[0]

    for _ in range(max_iter):
        num_males = int(np.ceil(gamma * dim))
        num_hinds = dim - num_males
        num_stags = int(np.ceil(alpha * num_males))
        num_coms = num_males - num_stags

        males = np.empty(shape=(num_males, dim))
        stags = np.empty(shape=(num_stags, dim))
        coms = np.empty(shape=(num_coms, dim))
        hinds = np.empty(shape=(num_hinds, dim))

        males = roar(num_males, males, len(ub), len(lb), dim, xdata, ydata)
        coms, stags = select_male_commanders_and_coms(males, num_coms)
        coms, stags = fight(xdata, ydata, dim, len(ub), len(lb), coms, num_coms, stags)
        coms, harem, num_harems = form_harems(xdata, ydata, coms, num_coms, num_hinds, hinds, dim)

        # mate
        num_harem_mate = [int(x * alpha) for x in num_harems]  # Eq. (11)
        population_pool = list(population)
        UB = len(ub)
        LB = len(lb)

        for i in range(num_coms):
            random.shuffle(harem[i])
            for j in range(num_harem_mate[i]):
                r = np.random.random()  # r is a random number in [0, 1]
                offspring = (coms[i] + harem[i][j]) / 2 + (UB - LB) * r  # Eq. (12)

                # apply transformation function on offspring
                for j in range(dim):
                    trans_value = sigmoid(offspring[j])
                    if np.random.random() < trans_value:
                        offspring[j] = 1
                    else:
                        offspring[j] = 0
                population_pool.append(list(offspring))

                # if number of commanders is greater than 1, inter-harem mating takes place
                if num_coms > 1:
                    # mating of commander with hinds in another harem
                    k = i
                    while k == i:
                        k = random.choice(range(num_coms))

                    num_mate = int(num_harems[k] * beta)  # Eq. (13)

                    np.random.shuffle(harem[k])
                    for m in range(num_mate):
                        r = np.random.random()
                        offspring = (coms[i] + harem[k][m]) / 2 + (UB - LB) * r
                        # apply transformation function on offspring
                        for f in range(dim):
                            trans_value = sigmoid(offspring[f])
                            if np.random.random() < trans_value:
                                offspring[f] = 1
                            else:
                                offspring[f] = 0
                        population_pool.append(list(offspring))

        for stag in stags:
            dist = np.zeros(num_hinds)
            for i in range(num_hinds):
                dist[i] = math.sqrt(np.sum((stag - hinds[i]) * (stag - hinds[i])))
            min_dist = np.min(dist)
            for i in range(num_hinds):
                distance = math.sqrt(np.sum((stag - hinds[i]) * (stag - hinds[i])))  # Eq. (14)
                if distance == min_dist:
                    r = np.random.random()  # r is a random number in [0, 1]
                    offspring = (stag + hinds[i]) / 2 + (UB - LB) * r

                    # apply transformation function on offspring
                    for j in range(dim):
                        trans_value = sigmoid(offspring[j])
                        if np.random.random() < trans_value:
                            offspring[j] = 1
                        else:
                            offspring[j] = 0
                    population_pool.append(list(offspring))

                    break

        # selection of next generation
        population_pool = np.array(population_pool)
        fitness = np.array([_evaluate_solution(xdata, ydata, ind) for ind in population_pool])
        population_pool, fitness = sort_agents(coms, fitness=fitness)
        maximum = sum([f for f in fitness])
        selection_probs = [f / maximum for f in fitness]
        indices = np.random.choice(len(population_pool), size=pop_size, replace=True, p=selection_probs)
        population = population_pool[indices]

    fitness = np.array([_evaluate_solution(xdata, ydata, ind) for ind in population])
    population, fitness = sort_agents(agents=population, fitness=fitness)
    accuracy = [_evaluate_solution(xdata, ydata, p) for p in population]
    if fitness[0] > Leader_fitness:
        Leader_fitness = fitness[0]
        Leader_agent = population[0, :]
        Leader_accuracy = accuracy[0]

    return Leader_accuracy, Leader_fitness, Leader_agent

