from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd


# Function to compute Mahalanobis distance and generate synthetic samples
def generate_synthetic_samples(X, n_neighbors=5, n_synthetic_samples=100):
    # Fit Nearest Neighbors to find nearest points
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(X)

    synthetic_samples = []

    for _ in range(n_synthetic_samples):
        # Randomly choose a minority sample
        random_sample = X[np.random.randint(0, len(X))]

        # Find neighbors of the random sample
        neighbors = nn.kneighbors([random_sample], return_distance=False)

        # Randomly pick one of the neighbors to interpolate between
        neighbor_sample = X[neighbors[0][np.random.randint(1, n_neighbors)]]

        # Interpolate between random_sample and neighbor_sample
        synthetic_sample = random_sample + np.random.rand() * (neighbor_sample - random_sample)

        synthetic_samples.append(synthetic_sample)

    return np.array(synthetic_samples)


df = pd.read_csv("./all_features.csv")
# Extract the minority class (Target = 0)
minority_class_samples = df[df['Target'] == 0].drop(columns=['Target']).values

# Generate synthetic samples for the minority class
n_synthetic_samples = 2274 - 1251  # Balance the number of samples
synthetic_samples = generate_synthetic_samples(minority_class_samples, n_synthetic_samples=n_synthetic_samples)

# Create a DataFrame for the synthetic samples
synthetic_df = pd.DataFrame(synthetic_samples, columns=df.drop(columns=['Target']).columns)

# Add the 'Target' column for the synthetic samples
synthetic_df['Target'] = 0

# Combine the original dataset with the synthetic samples
balanced_df = pd.concat([df, synthetic_df], ignore_index=True)

# Check the new distribution of the "Target" column
balanced_target_distribution = balanced_df['Target'].value_counts()

balanced_target_distribution
