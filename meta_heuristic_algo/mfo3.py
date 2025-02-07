import numpy as np
import pandas as pd
import random
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


def calc_acc(y, y_pred):
    total = len(y)
    c = 0
    for i in range(0, total):
        if y[i] == y_pred[i]:
            c += 1
    return c / total


def fitness(moth_pos, xtrain, ytrain, xtest, ytest):
    xtrain_selected_features = pd.DataFrame()
    xtest_selected_features = pd.DataFrame()

    for i in range(0, len(moth_pos)):
        xtrain_selected_features[str(moth_pos[i])] = xtrain.iloc[:, int(moth_pos[i])]
        xtest_selected_features[str(moth_pos[i])] = xtest.iloc[:, int[moth_pos[i]]]

    svc_classifier = SVC(kernel="poly", coef0=2)
    svc_classifier.fit(xtrain_selected_features, ytrain)
    ytest = ytest.to_numpy()
    ypred = svc_classifier.predict(xtest_selected_features)
    acc = calc_acc(ytest, ypred)

    return acc

def MFO(xtrain,ytrain,xtest,ytest,max_iter):
    length = len(xtrain.iloc[0])
    size1 = int(length)

    moth_population = np.zeros(shape=(20,size1))
    moth_fitness = np.zeros(20)
    for i in range(0,20):
        moth_population[i] = np.random.randint(length,size=size1)

    previous_population=0
    previous_fitness = 0

    for i in range(max_iter):
        print(str(i), end=" ")
        flame_no = round(20-i*((20-1)/max_iter))
        for i in range(20):
            moth_fitness[i] = fitness(moth_population[i],xtrain,ytrain,xtest,ytest)

        if i ==1:
            best_flame_fitness = np.sort(moth_fitness)[::-1]
            sorted_fitness = np.argsort(moth_fitness)[::-1]
            best_flames = moth_population[sorted_fitness]
        else:
            double_population = np.concatenate((previous_population,best_flames))
            double_fitness=np.concatenate((previous_fitness,best_flame_fitness))

            double_fitness_sorted = np.sort(double_fitness)[::-1]
            sorting = np.argsort(double_fitness)[::-1]
            double_sorted_population = double_population[sorting]

            best_flame_fitness = double_fitness_sorted[0,20]
            best_flames = double_sorted_population[0:20]

        best_flam_score = best_flame_fitness[1]
        best_flame_pos = best_flames[1]


