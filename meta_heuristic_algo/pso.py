import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Bidirectional, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN

from _utilities import sigmoid, initialize


class PSO:
    def __init__(self, num_agents, max_iter, xdata, ydata, seed=0):
        self.num_agents = num_agents
        self.max_iter = max_iter
        self.xdata = xdata
        self.ydata = ydata
        self.seed = seed
        self.num_features = 10

        self.trans_function = sigmoid
        self.global_best_particle = [0 for _ in range(self.num_features)]
        self.global_best_fitness = float("-inf")
        self.local_best_particle = [[0 for _ in range(self.num_features)] for _ in range(self.num_agents)]
        self.local_best_fitness = [float("-inf") for _ in range(self.num_agents)]
        self.weight = 1.0
        self.velocity = [[0.0 for _ in range(self.num_features)] for _ in range(self.num_agents)]
        self.population = None
        self.cur_iter = 0
        self.max_evals = np.float64("inf")
        self.solution = None
        self.fitness = None
        self.accuracy = None
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(self.xdata, self.ydata, test_size=0.2,
                                                                            random_state=42)
        # self.obj_function = call_counter(compute_fitness(self.weight))
        self.Leader_fitness = 0
        self.Leader_agent = 0
        self.Leader_accuracy = 0
        self.weight_acc = 0.9
        self.num_features = self.xtrain.shape[1]

    def initialize(self):
        self.population = initialize(num_agents=self.num_agents, num_features=self.num_features, seed=self.seed)
        self.fitness = self._compute_fitness(self.population)
        self.population, self.fitness = self._sort_agents(solutions=self.population, fitness=self.fitness)
        self.accuracy = self._evaluate_solution(self.population)
        self.Leader_agent, self.Leader_fitness = self.population[0], self.fitness[0]

    def next(self):
        print('\n================================================================================')
        print('                          Iteration - {}'.format(self.cur_iter + 1))
        print('================================================================================\n')

        self.weight = 1.0 - (self.cur_iter / self.max_iter)

        for i in range(self.num_agents):
            for j in range(self.num_features):
                self.velocity[i][j] = (self.weight * self.velocity[i][j])
                r1, r2 = np.random.random(2)
                self.velocity[i][j] = self.velocity[i][j] + (
                        r1 * (self.local_best_particle[i][j] - self.population[i][j]))
                self.velocity[i][j] = self.velocity[i][j] + (
                        r2 * (self.global_best_particle[j] - self.population[i][j]))

        # updating position of particles
        for i in range(self.num_agents):
            for j in range(self.num_features):
                trans_value = self.trans_function(self.velocity[i][j])
                if np.random.random() < trans_value:
                    self.population[i][j] = 1
                else:
                    self.population[i][j] = 0

        # updating fitness of particles
        self.fitness = self._compute_fitness(self.population)
        self.population, self.fitness = self._sort_agents(solutions=self.population, fitness=self.fitness)

        # updating the global best and local best particles
        for i in range(self.num_agents):
            if self.fitness[i] > self.local_best_fitness[i]:
                self.local_best_fitness[i] = self.fitness[i]
                self.local_best_particle[i] = self.population[i][:]

            if self.fitness[i] > self.global_best_fitness:
                self.global_best_fitness = self.fitness[i]
                self.global_best_particle = self.population[i][:]

        self.cur_iter += 1

    def check_end(self):
        return self.cur_iter >= self.max_iter

    def post_processing(self):
        self.fitness = self._compute_fitness(self.population)
        self.population, self.fitness = self._sort_agents(solutions=self.population, fitness=self.fitness)
        self.accuracy = self._evaluate_solution(solution=self.population)
        if self.fitness[0] > self.Leader_fitness:
            self.Leader_fitness = self.fitness[0]
            self.Leader_agent = self.population[0, :]
            self.Leader_accuracy = self.accuracy

    def run(self):
        self.initialize()
        while not self.check_end():
            self.next()
            self.post_processing()

        return self.Leader_agent, self.Leader_fitness, self.Leader_accuracy

    def create_model(self, hyperparams, input_shape):
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

    def compute_fitness(self, hyperparams):
        # xtrain, xval, ytrain, yval = train_test_split(xtrain, ytrain, test_size=0.2, random_state=42)
        model = self.create_model(hyperparams, self.xtrain.shape[1:])
        model.fit(self.xtrain, self.ytrain, epochs=10, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(self.xtest, self.ytest, verbose=0)
        return -accuracy  # Negative because we want to minimize the objective function

    def _evaluate_solution(self, solution):
        cols = np.flatnonzero(solution)
        if cols.shape[0] == 0:
            return 0

        # x_selected = self.xdata.iloc[:, solution.astype(bool)]
        x_train = self.xtrain[:, cols]
        x_test = self.xtest[:, cols]
        # x_train, x_test, y_train, y_test = train_test_split(x_selected, self.ydata, test_size=0.2, random_state=42)
        model = KNN()
        # model = SVC(kernel='poly', random_state=42, coef0=2)
        model.fit(x_train, self.ytrain)
        accuracy = accuracy_score(self.ytest, model.predict(x_test))
        return accuracy

    def _compute_fitness(self, solution):
        if self.weight_acc is None:
            self.weight_acc = 0.9

        weight_feat = 1 - self.weight_acc
        num_features = solution.shape[0]
        acc = self._evaluate_solution(solution)
        feat = (num_features - np.sum(solution)) / num_features
        fitness = self.weight_acc * acc + weight_feat * feat
        return fitness

    def _sort_agents(self, solutions, fitness=None):
        if fitness is None:
            if len(solutions.shape) == 1:
                fitness = self._compute_fitness(solutions)
                return solutions, fitness
        else:
            num_solutions = solutions.shape[0]
            fitness = np.zeros(num_solutions)
            for idx, solution in enumerate(solutions):
                fitness[idx] = self._compute_fitness(solution)

        idx = np.argsort(-fitness)
        sorted_solutions = solutions[idx].copy()
        sorted_fitness = fitness[idx].copy()

        return sorted_solutions, sorted_fitness


if __name__ == "__main__":
    algo = PSO(num_agents=20, max_iter=20)
