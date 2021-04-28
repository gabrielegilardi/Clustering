"""
Fuzzy C-means Clustering Using a Naive Algorithm and Particle Swarm
Optimization.

Copyright (c) 2021 Gabriele Gilardi


data            (n_samples, n_features)     Dataset to cluster
theta           (nPop, nVar)                Variables in PSO
means           (K, n_features)             Cluster centers
SSE             scalar, (nPop, )            Sum of weighted squared errors
m               scalar                      Fuzziness coefficient
U               (n_samples, K)              Membership (weights) array
tol             scalar                      Convergency tolerance
FPC             scalar                      Fuzzy partition coefficient
FPCn            scalar                      Normalized fuzzy partition coefficient
idx             (n_samples, )               Closest K-means index array

n_samples       Number of samples (rows) in the input dataset
n_features      Number of features (columns) in the input dataset
nPop            Number of agents in the PSO
nVar            Number of variables in the PSO
K               Number of desired cluster
n_rep           Number of repetitions (re-starts) in the naive algorithm
max_iter        Max. number of iterations in the naive algorithm


Notes:
- clusters are numbered from 0 to K-1.
- idx[i] is closest K-means cluster number for data[i,:].
- the weighted distance from the cluster centers is assumed as clustering error;
- the function minimized is the sum of (weighted) squared errors;
- the Dunn's and Kaufman's fuzzy partition coefficients are available metrics;
- the function "calc_U" can be used to classify new data.
"""

import numpy as np


def PSO_FCmeans(theta, args):
    """
    Returns the objective function (the Sum of Weighted Squared Errors) to be
    minimized by the PSO.

    Notes:
    - each PSO agent represents one possible set of cluster centers.
    - no early stop criteria is used (i.e. the PSO always reach the max. number
      of allowed iterations).
    """
    data = args[0]
    m = args[1]
    n_samples, n_features = data.shape
    nPop, nVar = theta.shape
    K = nVar // n_features
    SSE = np.zeros(nPop)

    # Loop over all agents
    for i in range(nPop):

        # Reshape the cluster centers
        means = theta[i, :].reshape(K, n_features)

        # Determine the membership array
        U = calc_U(data, means, m)

        # Determine the total (weighted) squared error
        SSE[i] = calc_SSE(data, means, U, m)

    return SSE


def naive_FCmeans(data, K=2, n_rep=10, m=2, max_iter=100, tol=1.e-2):
    """
    Returns the best solution using the naive algorithm.

    Note: a solution is found when the change in array <U> in two consecutive
          iterations is smaller than <tol>.
    """
    n_samples = data.shape[0]
    SSE_best = np.inf

    # Loop over the number of repetitions
    for rep in range(n_rep):

        # Initialize the membership array (the sum along the rows must be 1)
        U = np.random.uniform(size=(n_samples, K))
        U_sum = U.sum(axis=1)
        U = U / U_sum.reshape(-1, 1)

        # Minimize the (weighted) SSE
        U_previous = U
        for i in range(max_iter):

            # Determine the cluster centers
            means = calc_means(data, U, m)

            # Determine the membership array
            U = calc_U(data, means, m)

            # Iterate again if the stop condition is not verified
            delta = np.max(np.abs(U - U_previous))
            if (delta >= tol):
                U_previous = U

            # Exit if the stop condition is verified
            else:
                break

        # Determine the total sum of (weighted) squared errors
        SSE = calc_SSE(data, means, U, m)

        # Save if it is the best
        if (SSE < SSE_best):
            SSE_best = SSE
            means_best = means

    return SSE_best, means_best


def calc_U(data, means, m):
    """
    Returns the membership value of each data.
    """
    n_samples = data.shape[0]
    K = means.shape[0]
    U = np.zeros((n_samples, K))
    d = np.zeros((n_samples, K))
    fact = 2.0 / (m - 1.0)

    # Calculate all distances
    for j in range(K):
        diff2 = (data - means[j, :]) ** 2
        d[:, j] = np.sqrt(diff2.sum(axis=1))

    # Coefficients of the membership array
    for j in range(K):
        cc = (d[:, j].reshape(-1, 1) / d) ** fact
        U[:, j] = 1.0 / cc.sum(axis=1)

    return U


def calc_means(data, U, m):
    """
    Returns the cluster centers.
    """
    n_features = data.shape[1]
    K = U.shape[1]
    means = np.zeros((K, n_features))
    Um = U ** m

    # Loop over all clusters
    for j in range(K):

        uj = Um[:, j].reshape(-1, 1)
        means[j, :] = (data * uj).sum(axis=0) / uj.sum()

    return means


def calc_SSE(data, means, U, m):
    """
    Returns the total (weighted) squared error.
    """
    K = means.shape[0]
    SSE = 0.0
    Um = U ** m

    # Loop over all clusters
    for j in range(K):

        # Weights for this cluster
        uj = Um[:, j].reshape(-1, 1)

        # Weighted squared distance from the cluster center
        d = uj * (data - means[j, :]) ** 2

        # Total weighted squared distance
        SSE += d.sum()

    return SSE


def calc_FPC_Dunn(U):
    """
    Returns the fuzzy partition coefficient and the normalized fuzzy partition
    coefficient using Dunn's definition.
    """
    n_samples, K = U.shape

    # Dunn's coefficient (values between 1/K and 1)
    FPC = (U ** 2).sum() / n_samples

    # Normalized Dunn's coefficient (values between 0 and 1)
    FPCn = (K * FPC - 1) / (K - 1)

    return FPC, FPCn


def calc_FPC_Kaufman(U):
    """
    Returns the fuzzy partition coefficient and the normalized fuzzy partition
    coefficient using Kaufman's definition.
    """
    n_samples, K = U.shape

    # Kaufman's coefficient (values between 0 and 1-1/K)
    idx = np.argmax(U, axis=1)
    for j in range(K):
        cluster = (idx == j)
        U[cluster, j] = 1.0 - U[cluster, j]
    FPC = (U ** 2).sum() / n_samples

    # Normalized Kaufman's coefficient (values between 0 and 1)
    FPCn = K * FPC / (K - 1)

    return FPC, FPCn
