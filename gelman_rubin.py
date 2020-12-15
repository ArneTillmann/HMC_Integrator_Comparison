import numpy as np

def k_th_batch_mean(k, b, chains):
    return (1.0/b) * chains[int((k-1)*b+1):int(k*b),:].sum(axis=0)

def combined_estimator_of_mu(m, chains):
    return 1.0/m * (chains.mean(axis=1)).sum(axis=0)

def replicated_batch_means_estimator(b, a, m, chains):
    mu = combined_estimator_of_mu(m, chains)
    s = len(chains[0,0])
    x = np.zeros((s,s))
    for i in range(m):
        for k in range(a):
            k_th_batch_m = k_th_batch_mean(k, b, chains[i])
            #print(k_th_batch_m)
            x += np.outer((k_th_batch_m - mu),(k_th_batch_m - mu))
    return (b/(a*m -1))*x

def replicated_lugsail_batch_mean_sestimator(b, a, m, chains):
    return 2*replicated_batch_means_estimator(float(b), a, m, chains) - replicated_batch_means_estimator(np.floor(float(b/3)), a, m, chains)

#def biased_from_above_estimator(b, a, m, n, chains):
#    return (float(n-1.0))/n * average_of_the_m_sample_variances(m, n, chains) + (replicated_lugsail_batch_mean_sestimator(b, a, m, chains))/float(n)

def improved_estimator_PSRF(b, a, m, n, chains):
    s = len(chains[0,0])
    #print(replicated_batch_means_estimator(b,a,m,chains))
    #print(np.linalg.det(replicated_lugsail_batch_mean_sestimator(b, a, m, chains)))
    #print(np.linalg.det(average_of_the_m_sample_variances(m, n, chains)))
    return ((n-1.)/n+(np.linalg.det(replicated_lugsail_batch_mean_sestimator(b, a, m, chains)/np.linalg.det(average_of_the_m_sample_variances(m, n, chains))))**(1./s)/n)**(1.0/2)

def average_of_the_m_sample_variances(m, n, chains):
    s= len(chains[0,0])
    x = np.zeros((s,s))
    for i in range(m):
        x += np.cov(chains[i].T, ddof=1)
    return 1./m*x
