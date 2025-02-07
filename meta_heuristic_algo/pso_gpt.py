import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN


class PSOGpt:
    def __init__(self, n_particles, xtrain, xtest, ytrain, ytest, w=0.5, c1=2, c2=2, max_iter=50):
        self.n_particles = n_particles
        self.n_features = xtrain.shape[1]
        self.w = w  # Inertia weight
        self.c1 = c1  # Cognitive parameter
        self.c2 = c2  # Social parameter
        self.max_iter = max_iter
        self.swarm = []
        self.velocities = []
        self.pbest = []
        self.gbest = []
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest

    def initialize(self):
        self.swarm = np.random.randint(2, size=(self.n_particles, self.n_features))
        self.velocities = np.random.rand(self.n_particles, self.n_features)
        self.pbest = self.swarm.copy()
        self.gbest = self.swarm[np.argmax([self.fitness_function(p) for p in self.swarm])]

    def optimize(self):
        for iter in range(self.max_iter):
            fitness_values = np.array([self.fitness_function(p) for p in self.swarm])
            for i in range(self.n_particles):
                if fitness_values[i] > self.fitness_function(self.pbest[i]):
                    self.pbest[i] = self.swarm[i].copy()
            if np.max(fitness_values) > self.fitness_function(self.gbest):
                self.gbest = self.swarm[np.argmax(fitness_values)].copy()

            for i in range(self.n_particles):
                r1 = np.random.rand(self.n_features)
                r2 = np.random.rand(self.n_features)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.pbest[i] - self.swarm[i]) +
                                      self.c2 * r2 * (self.gbest - self.swarm[i]))
                self.swarm[i] = np.where(np.random.rand(self.n_features) < self.sigmoid(self.velocities[i]), 1, 0)

        return self.gbest

    def run(self):
        self.initialize()
        return self.optimize()

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def fitness_function(self, particle):
        X_train_fs = self.xtrain[:, particle.astype(bool)]
        X_test_fs = self.xtest[:, particle.astype(bool)]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_fs, self.ytrain)
        predictions = model.predict(X_test_fs)
        return accuracy_score(self.ytest, predictions)


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


def pso_gpt(n_particles, xdata, ydata, w=0.5, c1=2, c2=2, max_iter=50):
    n_features = xdata.shape[1]
    swarm = np.random.randint(2, size=(n_particles, n_features))
    velocities = np.random.rand(n_particles, n_features)
    pbest = swarm.copy()
    gbest = swarm[np.argmax([_evaluate_solution(xdata, ydata, p) for p in swarm])]

    for iter in range(max_iter):
        fitness_values = np.array([_evaluate_solution(xdata, ydata, p) for p in swarm])
        for i in range(n_particles):
            if fitness_values[i] > _evaluate_solution(xdata, ydata, pbest[i]):
                pbest[i] = swarm[i].copy()
        if np.max(fitness_values) > _evaluate_solution(xdata, ydata, gbest):
            gbest = swarm[np.argmax(fitness_values)].copy()

        for i in range(n_particles):
            r1 = np.random.rand(n_features)
            r2 = np.random.rand(n_features)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - swarm[i]) +
                             c2 * r2 * (gbest - swarm[i]))
            swarm[i] = np.where(np.random.rand(n_features) < sigmoid(velocities[i]), 1, 0)

    return gbest
