"""
K-means Clustering Using a Naive Algorithm and Particle Swarm Optimization.

Copyright (c) 2021 Gabriele Gilardi


data                (n_samples, n_features)     Dataset to cluster
theta               (nPop, nVar)                Variables in PSO
means               (K, n_features)             Cluster centers
idx                 (n_samples, )               Data cluster number
SSE                 scalar, (nPop, )            Sum of squared errors
cluster             (nel, )                     Data in a cluster
SC                  (n_samples, )               Silhoutte coefficient
DBindex             scalar                      Davies–Bouldin index


n_samples           Number of samples (rows) in the input dataset
n_features          Number of features (columns) in the input dataset
nPop                Number of agents in the PSO
nVar                Number of variables in the PSO
K                   Number of desired cluster
n_rep               Number of repetitions (re-starts) in the naive algorithm
max_iter            Max. number of iterations in the naive algorithm
nel                 Number of data in a cluster


Notes:
- clusters are numbered from 0 to K-1.
- idx[i] is the cluster number for data[i,:].
- the distance from the cluster centers is assumed as clustering error;
- the function minimized is the sum of squared errors;
- the silhouette coefficient and Davies–Bouldin index are available metrics;
- the function "assign_data" can be used to classify new data.
"""

import numpy as np


def PSO_Kmeans(theta, args):
    """
    Returns the objective function (the Sum of Squared Errors) to be minimized
    by the PSO.

    Notes:
    - each PSO agent represents one possible set of cluster centers.
    - no early stop criteria is used (i.e. the PSO always reach the max. number
      of allowed iterations).
    """
    data = args[0]
    n_features = data.shape[1]
    nPop, nVar = theta.shape
    K = nVar // n_features
    SSE = np.zeros(nPop)

    # Loop over all agents
    for i in range(nPop):

        # Reshape the cluster centers
        means = theta[i, :].reshape(K, n_features)

        # Assign data to clusters
        idx = assign_data(data, means)

        # Determine the total squared error
        SSE[i] = calc_SSE(data, means, idx)

    return SSE


def naive_Kmeans(data, K=2, n_rep=10, max_iter=100):
    """
    Returns the best solution using the naive algorithm.

    Note: a solution is found when array <idx> does not change in two
          consecutive iterations.
    """
    n_samples = data.shape[0]
    SSE_best = np.inf
    means_best = None

    # Shuffle the data indexes
    data_idx = np.arange(n_samples)
    np.random.shuffle(data_idx)

    # Loop over the number of repetitions
    for rep in range(n_rep):

        # Pick the initial cluster centers randomly from <data>
        init_idx = np.random.choice(data_idx, size=K, replace=False)
        means = data[init_idx, :]

        # Minimize the SSE
        idx_previous = np.zeros(n_samples, dtype=int)
        for i in range(max_iter):

            # Assign data to clusters
            idx = assign_data(data, means)

            # Determine the cluster centers
            means = calc_means(data, idx, K)

            # Skip this repetition if there is an empty cluster
            if (means is None):
                break

            # Iterate again if the clusters changed
            flag = (idx != idx_previous)
            if (np.any(flag)):
                idx_previous = idx

            # Stop and exit if the clusters did not change
            else:
                break

        # Skip this repetition if there is an empty cluster
        if (means is None):
            continue

        # Determine the total squared error
        SSE = calc_SSE(data, means, idx)

        # Save if it is the best
        if (SSE < SSE_best):
            SSE_best = SSE
            means_best = means

    return SSE_best, means_best


def assign_data(data, means):
    """
    Returns the cluster number of each data.
    """
    n_samples = data.shape[0]
    K = means.shape[0]
    d = np.zeros((n_samples, K))

    # Use quadratic distance instead of actual distance
    for j in range(K):
        d[:, j] = ((data - means[j, :]) ** 2).sum(axis=1)

    # Minimum distance define cluster membership
    idx = np.argmin(d, axis=1)

    return idx


def calc_means(data, idx, K):
    """
    Returns the cluster centers. Returns <None> if a cluster has zero data
    in it.
    """
    n_features = data.shape[1]
    means = np.zeros((K, n_features))

    # Loop over all clusters
    for j in range(K):

        # Data belonging to the j-th cluster
        cluster = (idx==j)

        # Number of data in the j-th cluster
        nel = cluster.sum()
        if (nel == 0):
            return None

        # Determine the cluster center
        means[j, :] = data[cluster, :].sum(axis=0) / nel

    return means


def calc_SSE(data, means, idx):
    """
    Returns the total squared error.
    """
    K = means.shape[0]
    SSE = 0.0

    # Loop over all clusters
    for j in range(K):

        # Squared distance from the cluster center
        d = (data[idx==j, :] - means[j, :]) ** 2

        # Total squared error
        SSE += d.sum()

    return SSE


def calc_SC(data, means, idx):
    """
    Returns the silhouette coefficient.
    """
    n_samples = data.shape[0]
    K = means.shape[0]
    SC = np.zeros(n_samples)

    # Loop over all data
    for i in range(n_samples):

        # Cluster of the current data
        cluster = (idx==idx[i])
        nel = cluster.sum()

        # If only one data in the cluster the coefficient is zero
        if (nel == 1):
            SC[i] = 0
            continue

        # Average distance of the current data from all other data in the
        # same cluster
        diff2 = (data[cluster, :] - data[i, :]) ** 2
        d = np.sqrt(diff2.sum(axis=1))
        ai = d.sum() / (nel - 1)

        # Search for the closest cluster
        d_min = np.inf
        for j in range(K):

            # Skip if same cluster
            if (j == idx[i]):
                continue

            # Use squared distance
            d = ((means[j, :] - data[i, :]) ** 2).sum()

            # Save if it is the closest
            if (d < d_min):
                d_min = d
                j_min = j

        # Data in the closest cluster
        cluster = (idx==j_min)
        nel = cluster.sum()

        # Average distance of the current data from all data in the closest
        # cluster
        diff2 = (data[cluster, :] - data[i, :]) ** 2
        d = np.sqrt(diff2.sum(axis=1))
        bi = d.sum() / nel

        # Silhouette coefficient
        SC[i] = (bi - ai) / max(ai, bi)

    return SC


def calc_DBI(data, means, idx):
    """
    Returns the Davies–Bouldin index.
    """
    K = means.shape[0]
    S = np.zeros(K)
    R = np.zeros((K, K))

    # Loop over all clusters
    for j in range(K):

        # Current cluster
        cluster = (idx==j)
        nel = cluster.sum()

        # Squared distance from the cluster center
        diff2 = (data[cluster, :] - means[j, :]) ** 2
        d = np.sqrt(diff2.sum(axis=1))

        # Cluster diameter
        S[j] = d.sum() / nel

    # Build the symmetric array R
    for i in range(K-1):

        for j in range(i+1, K):

            # Distance between the two clusters
            diff2 = (means[i, :] - means[j, :]) ** 2
            d = np.sqrt(diff2.sum())

            # Factor R
            R[i, j] = (S[i] + S[j]) / d
            R[j, i] = R[i, j]

    # Davies-Bouldin index
    DBI = np.max(R, axis=1).sum() / K

    return DBI
