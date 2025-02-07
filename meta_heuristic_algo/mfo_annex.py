import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def F1(x):
    ''' F1 function as defined in the paper for the test '''
    return np.sum(np.power(x, 2), axis=1)


def F2(x):
    ''' F2 function as defined in the paper for the test '''
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)


def F3(x):
    ''' F3 function as defined in the paper for the test '''
    o = 0
    for i in range(x.shape[1]):
        o += np.power(np.sum(x[:, :i], axis=1), 2)
    return o


def F4(x):
    ''' F4 function as defined in the paper for the test '''
    return np.max(x, axis=1)


def F5(x):
    ''' F5 function as defined in the paper for the test '''
    o = 0
    for i in range(x.shape[1] - 1):
        o += 100 * np.power((x[:, i + 1] - np.power(x[:, i], 2)), 2) + np.power(x[:, i] - 1, 2)
    return o


def F6(x):
    ''' F6 function as defined in the paper for the test '''
    return np.sum(np.power(x + 0.5, 2), axis=1)


def F7(x):
    ''' F7 function as defined in the paper for the test '''
    n = np.arange(1, x.shape[1] + 1, 1)
    return np.sum(n * np.power(x, 4), axis=1) + np.random.rand(x.shape[0])


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


def MFO(search_agents, dim, upper_bound, lower_bound, max_iter):
    moth_pos = np.random.uniform(low=lower_bound, high=upper_bound, size=(search_agents, dim))
    convergence_curve = np.zeros(shape=max_iter)
    best_flames = None
    best_flames_fit = None
    best_flame_score = None
    best_flame_pos = None
    for iteration in range(max_iter):
        number_flame = int(np.ceil(search_agents - (iteration + 1) * ((search_agents - 1) / max_iter)))

        moth_pos = np.clip(moth_pos, lower_bound, upper_bound)
        moth_fit = F6(moth_pos)

        if iteration == 0:
            order = np.argsort(moth_fit, axis=0)
            moth_fit = moth_fit[order]
            moth_pos = moth_pos[order, :]

            # updates the flames
            best_flames = np.copy(moth_pos)
            best_flames_fit = np.copy(moth_fit)

        else:
            # Sort the moths
            double_pos = np.vstack((best_flames, moth_pos))
            double_fit = np.hstack((best_flames_fit, moth_fit))
            order = np.argsort(double_fit, axis=0)
            double_fit = double_fit[order]
            double_pos = double_pos[order, :]

            # Updates the flames
            best_flames = double_pos[:search_agents, :]
            best_flames_fit = double_fit[:search_agents]

        # Update the position best flame obtained so far
        best_flame_score = best_flames_fit[0]
        best_flame_pos = best_flames[0, :]

        # a linearly decrease from -1 to -2 to calculate t in Eq. (3.12)
        a = -1 + (iteration + 1) * ((-1) / max_iter)

        # D in Eq. (3.13)
        distance_to_flames = np.abs(best_flames - moth_pos)
        b = 1
        t = (a - 1) * np.random.rand(search_agents, dim) + 1
        temp1 = best_flames[:number_flame, :]
        temp2 = best_flames[number_flame - 1, :] * np.ones(shape=(search_agents - number_flame, dim))
        temp2 = np.vstack((temp1, temp2))
        moth_pos = distance_to_flames * np.exp(b * t) * np.cos(t * 2 * np.pi) + temp2
        convergence_curve[iteration] = best_flame_score
    return best_flame_score, best_flame_pos, convergence_curve
