import numpy as np
from numpy.random import random_sample
from scipy.spatial.distance import mahalanobis
from sklearn.neighbors import NearestNeighbors


def mahalanobis_distance(X, y, minority_class_label):
    minority_class_samples = X[y==minority_class_label]
    covariance_matrix=np.cov(minority_class_samples.T)
    inv_cov_matrix = np.linalg.inv(covariance_matrix)

    distances = []
    for sample in minority_class_samples:
        distances.append(mahalanobis(sample, minority_class_samples.mean(axis=0), inv_cov_matrix))

    return np.array(distances)


def get_minority_class_samples(X,y,minority_class_label):

    return X[y==minority_class_label]

def get_nearest_neighbors(X,neighbors=5):
    nn = NearestNeighbors(n_neighbors=neighbors)
    nn.fit(X)
    return nn

def generate_synthetic_samples(minority_class_samples,nn_model,n_synthetic_samples):
    synthetic_samples = []
    for _ in range(n_synthetic_samples):
        random_sample = minority_class_samples[np.random.randint(0,len(minority_class_samples))]
        neighbors = nn_model.kneighbors([random_sample],return_distance=False)
        neighbor_sample = minority_class_samples[neighbors[0][np.random.randint(1,len(neighbors[0]))]]

        # Interpolate using Mahalanobis-based technique
        synthetic_sample = random_sample + np.random.rand()*(neighbor_sample-random_sample)
        synthetic_samples.append(synthetic_sample)

    return np.array(synthetic_samples)


def oversample_with_mahakel(X,y,minority_class_label,n_synthetic_samples):
    minority_class_samples= get_minority_class_samples(X,y,minority_class_label)
    nn_model = get_nearest_neighbors(X,n_synthetic_samples)
    synthetic_samples = generate_synthetic_samples(minority_class_samples,nn_model,n_synthetic_samples)

    # Append the synthetic samples to the original data
    X_balanced = np.vstack([X, synthetic_samples])
    y_balanced = np.hstack([y,[minority_class_label]*n_synthetic_samples])
    return X_balanced,y_balanced

