import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN


def compute_accuracy(agents, xtrain, ytrain, xtest, ytest):
    # compute classification
    if len(agents.shape) == 1:
        agents = np.array([agents])

    (num_agents, num_features) = agents.shape
    acc = np.zeros(num_agents)
    clf = KNN()

    for (i, agent) in enumerate(agents):
        cols = np.flatnonzero(agent)

        train_data = xtrain[:, cols]
        test_data = xtest[:, cols]

        clf.fit(train_data, ytrain)
        acc[i] = clf.score(test_data, ytest)

    return acc


def compute_fitness(weight_acc):
    def _compute_fitness(agents, xtrain, ytrain, xtest, ytest):
        if len(agents.shape) == 1:
            # handles with zero-rank arrays
            agents = np.array([agents])

        weight_feat = 1 - weight_acc
        (num_agents, num_features) = agents.shape
        fitness = np.zeros(num_agents)
        acc = compute_accuracy(agents, xtrain, ytrain, xtest, ytest)
        feat = None
        for (i, agent) in enumerate(agents):
            if np.sum(agents[i]) != 0:
                feat = (num_features - np.sum(agents[i])) / num_features
                fitness[i] = weight_acc * acc[i] + weight_feat * feat

        return feat

    return _compute_fitness


def call_counter(func):
    # meta function to count the number of calls to another function
    def helper(*args, **kwargs):
        helper.cur_evals += 1
        func_val = func(*args, **kwargs)
        return func_val

    helper.cur_evals = 0
    helper.__name__ = func.__name__
    return helper


def sort_agents(agents, fitness):
    # sort the agents according to fitness
    idx = np.argsort(-fitness)
    sorted_agents = agents[idx].copy()
    sorted_fitness = fitness[idx].copy()

    return sorted_agents, sorted_fitness


def initialize(num_agents, num_features, seed):
    np.random.seed(seed)
    min_features = int(0.3 * num_features)
    max_features = int(0.6 * num_features)

    agents = np.zeros((num_agents, num_features))

    for agent_no in range(num_agents):
        cur_count = np.random.randint(min_features, max_features)
        temp_vec = np.random.rand(1, num_features)
        temp_idx = np.argsort(temp_vec)[0][0:cur_count]

        # select the features with the random indices
        agents[agent_no][temp_idx] = 1

    return agents


def sigmoid(val):
    if val < 0:
        return 1 - 1 / (1 + np.exp(val))
    else:
        return 1 / (1 + np.exp(-val))