import numpy as np
"""This is the implementation of the revised Gelman-Rubin Diagnostic
with use of the presented calculations given in the paper
https://arxiv.org/pdf/1812.09384.pdf
written by Dootika Vats and Christina Knudson
"""

def k_th_batch_mean(k, b, chains):
    """calculation of a special mean for each chain respectively"""

    return (1.0/b) * chains[int((k-1)*b+1):int(k*b),:].sum(axis=0)


def combined_estimator_of_mu(m, chains):
    """The mean of means for different chains"""

    return 1.0/m * (chains.mean(axis=1)).sum(axis=0)


def replicated_batch_means_estimator(b, a, m, p, chains):
    """intermediate step for the following calculation"""

    mu = combined_estimator_of_mu(m, chains)
    x = np.zeros((p,p))

    for i in range(m):
        for k in range(a):
            k_th_batch_m = k_th_batch_mean(k, b, chains[i])
            x += np.outer((k_th_batch_m - mu),(k_th_batch_m - mu))

    return (b/(a*m-1)) * x


def replicated_lugsail_batch_mean_sestimator(b, a, m, p, chains):
    """see equation (9) in the paper named at the beginning"""

    return (2 * replicated_batch_means_estimator(float(b), a, m, p, chains)
           - replicated_batch_means_estimator(np.floor(b/3), a, m, p, chains))


#def biased_from_above_estimator(b, a, m, n, chains):

#    return (float(n-1.0))/n * average_of_the_m_sample_covariances(m, n, chains)
#           + (replicated_lugsail_batch_mean_sestimator(b, a, m, chains))/float(n)


def improved_estimator_PSRF(b, a, m, n, p, chains):
    """The final revised gelman-rubin statistic receiving b,a as additional
    parameters for the calculation of the k_th_batch_mean.
    They are chosen in the way, that n=a*b, whereas n is the chain length
    (number of samples in one chain). Usual choices for b include round off
    n^(1/2) or round off n^(1/3) according to the authers.
    m is the number of chains, n their length, p the dimension of the target
    distribution and chains the np.ndarray containing the chains in order [m,n,p]
    """

    t_hat = replicated_lugsail_batch_mean_sestimator(b, a, m, p, chains)
    sigma_hat = average_of_the_m_sample_covariances(m, n, p, chains)

    return (((n-1.)/n
           + (np.linalg.det(t_hat)
           / np.linalg.det(sigma_hat))**(1./p)/n)**(1.0/2))


def average_of_the_m_sample_covariances(m, n, p, chains):
    """Auxiliary function"""

    x = np.zeros((p,p))

    for i in range(m):
        x += np.cov(chains[i].T, ddof=1)

    return 1./m * x
