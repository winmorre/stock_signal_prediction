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


def woa_gpt(xdata, ydata, dim, lb, ub, pop_size=30, max_iter=100):
    """
    lb:  Lower bounds of the hyperparameters
    ub: Upper bounds of the hyperparameters
    dim: Number of hyperparameters to optimize
    """
    positions = np.random.uniform(0, 1, (pop_size, dim)) * (ub - lb) + lb
    leader_pos = np.zeros(dim)
    leader_score = float("inf")

    for iter in range(max_iter):
        for i in range(pop_size):
            fitness = _evaluate_solution(xdata, ydata, positions[i])
            if fitness < leader_score:
                leader_score = fitness
                leader_pos = positions[i].copy()

        a = 2 - iter * (2 / max_iter)
        for i in range(pop_size):
            r1 = np.random.rand()
            r2 = np.random.rand()
            A = 2 * a * r1 - a
            C = 2 * r2
            p = np.random.rand()

            if p < 0.5:
                if abs(A) < 1:
                    D_leader = abs(C * leader_pos - positions[i])
                    positions[i] = leader_pos - A * D_leader
                else:
                    rand_pos = positions[np.random.randint(pop_size)]
                    D_rand = abs(C * rand_pos - positions[i])
                    positions[i] = rand_pos - A * D_rand
            else:
                distance_to_leader = abs(leader_pos - positions[i])
                positions[i] = distance_to_leader * np.exp(1 - iter / max_iter) * np.cos(
                    2 * np.pi * np.random.rand()) + leader_pos

        positions = np.clip(positions, lb, ub)

    return leader_pos, leader_score
