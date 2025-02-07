import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


class MFO:
    def __init__(self, n_moths, n_flames, max_iter, n_features):
        self.n_moths = n_moths
        self.n_flames = n_flames
        self.max_iter = max_iter
        self.n_features = n_features

    def initialize_population(self, n):
        return np.random.randint(2, size=(self.n_moths, n))

    def fitness(self, X, y, moth):
        selected_features = [i for i in range(len(moth)) if moth[i] == 1]
        if len(selected_features) == 0:
            return 0
        X_selected = X[:, selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return accuracy_score(y_test, y_pred)

    def update_position(self, moth, flame, t, b=1):
        distance = np.abs(flame - moth)
        return distance * np.exp(b * t) * np.cos(2 * np.pi * t) + flame

    def optimize(self, X, y):
        n = X.shape[1]
        population = self.initialize_population(n)
        fitness_values = np.array([self.fitness(X, y, moth) for moth in population])
        best_moth = None
        for iteration in range(self.max_iter):
            sorted_indices = np.argsort(fitness_values)[::-1]
            flames = population[sorted_indices[:self.n_flames]]
            best_fitness = fitness_values[sorted_indices[0]]
            best_moth = population[sorted_indices[0]]

            for i in range(self.n_moths):
                t = np.random.rand()
                flame = flames[i % self.n_flames]
                population[i] = self.update_position(population[i], flame, t)
                population[i] = np.clip(population[i], 0, 1)  # ensure binary values
                fitness_values[i] = self.fitness(X, y, population[i])

            print(f'Iteration {iteration + 1}/{self.max_iter}, Best Fitness: {best_fitness}')

        return best_moth


def main():
    X, y = make_classification(n_samples=1000, n_features=84, n_informative=30, n_redundant=0, random_state=42)

    # MFO parameters
    n_moths = 20
    n_flames = 10
    max_iter = 100
    n_features_to_select = 30
    mfo = MFO(n_moths, n_flames, max_iter, n_features_to_select)
    best_features = mfo.optimize(X, y)

    # print selected features
    selected_features = [i for i in range(len(best_features)) if best_features[i] == 1]
    print("Best features  :  ", best_features, "\n")
    print("Selected features: ", selected_features)


if __name__ == "__main__":
    main()
