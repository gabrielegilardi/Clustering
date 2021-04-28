"""
K-means and Fuzzy C-means Clustering Using a Naive Algorithm and Particle
Swarm Optimization.

Copyright (c) 2021 Gabriele Gilardi


Features
--------
- The code has been written and tested in Python 3.8.8.
- Two clustering methods (K-means and fuzzy C-means) and two solvers (naive
  algorithm and PSO).
- For the K-means clustering method:
  the distance from the cluster centers is assumed as clustering error;
  the function minimized is the sum of squared errors;
  the silhouette coefficient and Davies–Bouldin index are available metrics;
  the function "assign_data" can be used to classify new data.
- For the fuzzy C-means clustering method:
  the weighted distance from the cluster centers is assumed as clustering error;
  the function minimized is the sum of (weighted) squared errors;
  the Dunn's and Kaufman's fuzzy partition coefficients are available metrics;
  the function "calc_U" can be used to classify new data.
- Usage: python test.py <example>.


Main Parameters
---------------
example = g2, dim2, unbalance, s3
    Name of the example to run
nPop >=1, epochs >= 1
    Number of agents (population) and number of iterations.
K >= 2, K_list
    Number of clusters.
n_rep > 0
    Number of repetitions (re-starts) in the naive algorithm.
max_iter > 0
    Max. number of iterations in the naive algorithm.
func = PSO_Kmeans, PSO_FCmeans
    Name of the interface function for the PSO.
1 < m < inf
    Fuzziness coefficient in the fuzzy C-means method.
tol
    Convergency tolerance in the fuzzy C-means method.

The other PSO parameters are used with their default values (see pso.py).


Examples
--------
Example 1: K-means using PSO, 2 clusters, 16 features, 2048 samples.

Example 2: K-means using naive algorithm, 2 to 15 clusters, 2 features, 1351
           samples, siluhouette coefficient and Davies–Bouldin index as metrics.

Example 3: Fuzzy C-means using PSO, 8 clusters (unbalanced), 2 features, 6500
           samples.

Example 4: Fuzzy C-means using naive algorithm, 2 to 20 clusters, 2 features,
           5000 samples, Dunn's and Kaufman's fuzzy partition coefficients as
           metrics.


References
----------
- K-means @ https://en.wikipedia.org/wiki/K-means_clustering
- Fuzzy C-means @ https://en.wikipedia.org/wiki/Fuzzy_clustering
- PSO @ https://github.com/gabrielegilardi/PSO.git.
- Datasets @ http://cs.joensuu.fi/sipu/datasets/.


To do list
----------
- add readme
- add in projects docs
- add on github
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from pso import PSO
from Kmeans import *
from FCmeans import *

# Read example to run
if len(sys.argv) != 2:
    print("Usage: python test.py <example>")
    sys.exit(1)
example = sys.argv[1]

np.random.seed(1294404794)

# Example 1: K-means using PSO and with a fixed number of clusters
if (example == 'g2'):

    # Dataset: 2 clusters, 16 features, 2048 samples
    # Cluster centers: [[500, 500, 500, ....., 500],
    #                   [600, 600, 600, ....., 600]]
    # File g2-16-50.txt from the G2 set
    # http://cs.joensuu.fi/sipu/datasets/g2-txt.zip

    # Parameters
    K = 2
    func = PSO_Kmeans
    nPop = 40
    epochs = 100

    # Load data
    X = np.loadtxt('g2-16-50.csv', delimiter=',')
    n_samples, n_features = X.shape

    # Normalize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    Xn = (X - mu) / sigma

    # Run solver
    LB = np.tile(np.amin(Xn, axis=0), K)
    UB = np.tile(np.amax(Xn, axis=0), K)
    args = [Xn]
    theta, info = PSO(func, LB, UB, nPop=nPop, epochs=epochs, args=args)
    print(info[0])
    means = theta.reshape(K, n_features)

    # Solution (SSE = 16259.491)
    print("\nSSE:", np.around(info[0], 3))
    means = mu + sigma * means
    print("\nCluster centers:")
    print(np.around(means, 2))

    # Max. error with respect to actual solution (max. error = 0.798 %)
    sol = np.tile(np.array([[500], [600]]), (1, 16))
    max_err = 100 * np.max(np.abs((means - sol) / sol))
    print("\nMax. error [%]:", np.around(max_err, 3))

# Example 2: K-means using naive algorithm and a variable number of clusters
elif (example == 'dim2'):

    # Dataset: 9 clusters, 2 features, 1351 samples
    # File dim2.txt from the DIM2 set (low)
    # Metrics: siluhouette coefficient and Davies–Bouldin index
    # http://cs.joensuu.fi/sipu/datasets/data_dim_txt.zip

    # Parameters
    K_list = np.arange(2, 16)
    n_rep = 10
    max_iter = 1000

    # Load data
    X = np.loadtxt('dim2.csv', delimiter=',')
    n_samples, n_features = X.shape

    # Normalize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    Xn = (X - mu) / sigma

    # Find the optimal K in the interval [2,15] using 10 repetitions
    print()
    for K in K_list:

        # Solve
        SSE, means = naive_Kmeans(Xn, K=K, n_rep=n_rep, max_iter=max_iter)

        # Cluster and metrics
        idx = assign_data(Xn, means)
        SC = calc_SC(Xn, means, idx).mean()
        DBI = calc_DBI(Xn, means, idx)

        # Show best results for each K
        print("K = {0:2d}    SSE = {1:e}    SC = {2:.3f}    DBI = {3:.3f}"
              .format(K, SSE, SC, DBI))

    # Solution (normalized) for K = 9 (best solution based on both SC and DBI)
    # K = 9, SSE = 3.740477, SC = 0.948, DBI = 0.073
    K = 9
    SSE, means = naive_Kmeans(Xn, K=K, n_rep=n_rep, max_iter=max_iter)
    print("\nCluster centers (K = 9):")
    print(np.around(means, 4))

    # Plot (normalized) clusters for K = 9
    idx = assign_data(Xn, means)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, K)))
    for j in range(K):
        c = next(color)
        plt.plot(Xn[idx==j, 0], Xn[idx==j, 1], 'o', c=c, ms=2)
    plt.plot(means[:, 0], means[:, 1], 'X', c='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Cluster centers (normalized values) for K = 9")
    plt.xlim(-2, 1.5)
    plt.ylim(-2.5, 1.5)
    plt.grid(b=True)
    plt.show()

# Example 3: fuzzy C-means using PSO and with a fixed number of clusters
elif (example == 'unbalance'):

    # Dataset: 8 clusters, 2 features, 6500 samples, unbalanced
    # File unbalance.txt from the Umbalance set
    # There are 5 clusters with 100 points each and 3 with 2000 points each
    # http://cs.joensuu.fi/sipu/datasets/unbalance.txt

    # Parameters
    K = 8
    m = 2
    func = PSO_FCmeans
    nPop = 200
    epochs = 100

    # Load data
    X = np.loadtxt('unbalance.csv', delimiter=',')
    n_samples, n_features = X.shape

    # Normalize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    Xn = (X - mu) / sigma

    # Run solver
    LB = np.tile(np.amin(Xn, axis=0), K)
    UB = np.tile(np.amax(Xn, axis=0), K)
    args = [Xn, m]
    theta, info = PSO(func, LB, UB, nPop=nPop, epochs=epochs, args=args)
    means = theta.reshape(K, n_features)

    # Solution (normalized)
    print("\nSSE:", np.around(info[0], 3))      # SSE = 231.763
    print("\nCluster centers:")
    print(np.around(means, 4))

    # Plot the (normalized) clusters with the highest probabilities
    U = calc_U(Xn, means, m)
    idx = np.argmax(U, axis=1)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, K)))
    for j in range(K):
        c = next(color)
        plt.plot(Xn[idx==j, 0], Xn[idx==j, 1], 'o', c=c, ms=2)
    plt.plot(means[:, 0], means[:, 1], 'X', color='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Cluster centers (highest probabilities)")
    plt.xlim(-1, 5)
    plt.ylim(-5, 5)
    plt.grid(b=True)
    plt.show()

# Example 4: fuzzy C-means using naive algorithm and a variable number of
# clusters
elif (example == 's3'):

    # Dataset: 15 clusters, 2 features, 5000 samples
    # File s3.txt from the S-sets
    # Metrics: Dunn's and Kaufman's fuzzy partition coefficients
    # http://cs.joensuu.fi/sipu/datasets/s3.txt

    # Parameters
    K_list = np.arange(2, 21)
    n_rep = 10
    max_iter = 1000
    m = 2
    tol = 1.e-3

    # Load data
    X = np.loadtxt('s3.csv', delimiter=',')
    n_samples, n_features = X.shape

    # Normalize
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=0)
    Xn = (X - mu) / sigma

    # Find the optimal K in the interval [2,20] using 10 repetitions
    print()
    for K in K_list:

        # Solve
        SSE, means = naive_FCmeans(Xn, K=K, n_rep=n_rep, max_iter=max_iter,
                                   m=m, tol=tol)

        # Metrics
        U = calc_U(Xn, means, m)
        idx = np.argmax(U, axis=1)
        _, FPCn_D = calc_FPC_Dunn(U)
        _, FPCn_K = calc_FPC_Kaufman(U)

        # Show best results for each K
        print("K = {0:2d}    SSE = {1:e}    FPCn_D = {2:.3f}    FPCn_K = {3:.3f}"
              .format(K, SSE, FPCn_D, FPCn_K))

    # # Means (normalized) for K = 15 (best solution based on both FPCs)
    # K = 15, SSE = 204.2418, FPCn_D = 0.538, FPCn_K = 0.183
    K = 15
    SSE, means = naive_FCmeans(Xn, K=K, n_rep=n_rep, max_iter=max_iter,
                             m=m, tol=tol)
    print("\nCluster centers (K = 15):")
    print(np.around(means, 4))

    # Plot (normalized) clusters for K = 15
    U = calc_U(Xn, means, m)
    idx = np.argmax(U, axis=1)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, K)))
    for j in range(K):
        c = next(color)
        plt.plot(Xn[idx==j, 0], Xn[idx==j, 1], 'o', c=c, ms=2)
    plt.plot(means[:, 0], means[:, 1], 'X', color='k')
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Cluster centers (normalized values) for K = 15")
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.grid(b=True)
    plt.show()

else:

    print("Example not found")
    sys.exit(1)
