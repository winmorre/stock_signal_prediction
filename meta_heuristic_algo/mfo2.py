import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN

# stock_data = pd.read_csv('data.csv')
# X = stock_data.drop("target", axis=1)
# y = stock_data["target"]
scaler_standard = StandardScaler()


def _evaluate_solution(X, y, solution):
    # Use the selected feature to train a RandomForestClassifier
    X_selected = X.iloc[:, solution.astype(bool)]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model = KNN()
    # model = SVC(kernel='poly', random_state=42, coef0=2)
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return accuracy


def _update_moth_position(moth, flames, flame_scores):
    # Update the moth position based on the position of the flames
    new_moth = moth.copy()
    for j in range(len(moth)):
        if np.random.rand() < 0.5:
            new_moth[j] = 1 - moth[j]
    return new_moth


def _update_flame_position(flame, moths, moth_scores):
    new_flame = flame.copy()
    for j in range(len(flame)):
        if np.random.rand() < 0.5:
            new_flame[j] = 1 - flame[j]
    return new_flame


def mfo(X, y, n_features=10, max_iter=100):
    # Initialize the moth and the flame populations
    n_moths = n_features
    n_flames = X.shape[1] - n_features
    moths = np.random.randint(0, 2, size=(n_moths, X.shape[1]))
    flames = np.random.randint(0, 2, size=(n_flames, X.shape[1]))

    best_solution = None
    best_score = 0
    best_idx = None
    # best_indices = set()

    for ii in range(max_iter):
        if ii + 1 % 10 == 0:
            print(f"Iteration {ii + 1}/{max_iter}, Best score: {best_score}, Best index: {best_idx}")
        # evaluate the fitness of the moths and flames
        moth_scores = [_evaluate_solution(X, y, solution) for solution in moths]
        flame_scores = [_evaluate_solution(X, y, solution) for solution in flames]

        # update the moth and flame positions
        for i in range(n_moths):
            moths[i] = _update_moth_position(moths[i], flames, flame_scores)
        for i in range(n_flames):
            flames[i] = _update_flame_position(flames[i], moths, moth_scores)

        # track the best solution
        all_solutions = np.concatenate((moths, flames), axis=0)
        all_scores = moth_scores + flame_scores
        best_idx = np.argmax(all_scores)
        # best_indices.add(best_idx)
        if all_scores[best_idx] > best_score:
            best_solution = all_solutions[best_idx]
            best_score = all_scores[best_idx]

    return best_solution
