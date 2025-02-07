import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

num_moths = 30
# dim = 1  # Xtrain.shape[1]

max_iter = 20

# stock_data = pd.read_csv('data.csv')
# X = stock_data.drop("target", axis=1)
# y = stock_data["target"]
scaler = StandardScaler()


def _objective_function(selected_features, X, y):
    X_selected = X.iloc[:, selected_features.astype(bool)]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # X_train_selected = X_train[:, selected_features == 1]
    # X_test_selected = X_test[:, selected_features == 1]

    # Train classifier abd evaluate performance
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Use accuracy as the fitness value (can be any other metrics)
    fitness = accuracy_score(y_test, y_pred)
    return fitness


def update_flames(moths, flames, moth_fitness, flame_fitness):
    # Sort flames based on fitness
    sorted_indices = np.argsort(flame_fitness)[::-1]  # Desending order
    flames = flames[sorted_indices]
    flame_fitness = flame_fitness[sorted_indices]

    # Update flames based on moths fitness
    num_flames = len(flames)
    for i in range(num_flames):
        if moth_fitness[i] > flame_fitness[i]:
            flames[i] = moths[i]
            flame_fitness[i] = moth_fitness[i]

    return flames, flame_fitness


def spiral_updates(moths, flames, b, t):
    for i in range(len(moths)):
        for j in range(len(moths[i])):

            if np.random.rand() < 0.5:
                print("I ", i, " J : ", "\n")
                distance_of_flame = abs(flames[i, j] - moths[i, j])
                moths[i, j] = (distance_of_flame * np.exp(b * t) * np.cos(t * 2 * np.pi) + flames[i, j]).astype(float)
            else:
                moths[i, j] = flames[i, j]

    return moths


def mfo(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dim = X_train.shape[1]
    moths = np.random.randint(2, size=(num_moths, dim))
    flames = np.copy(moths)
    moth_fitness = np.array([_objective_function(moth, X, y) for moth in moths])
    flame_fitness = np.copy(moth_fitness)
    b = 1
    for i in range(max_iter):
        flames, flame_fitness = update_flames(moths, flames, moth_fitness, flame_fitness)
        t = (i / max_iter) * 2 * np.pi
        moths = spiral_updates(moths, flames, b, t)

        moth_fitness = np.array([_objective_function(moth, X, y) for moth in moths])
        best_fitness = np.argmax(flame_fitness)
        print(f"Iteration {i + 1}/{max_iter}, Best fitness: {best_fitness}")

    best_index = np.argmax(flame_fitness)
    best_features = flames[best_index]
    selected_feature_indices = np.where(best_features == 1)[0]
    return best_features, selected_feature_indices
