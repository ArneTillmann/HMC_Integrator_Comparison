from itertools import combinations
from mpl_toolkits import mplot3d
import autograd.numpy as np  # Thinly-wrapped numpy
# import numpy as np
from autograd import grad
from autograd import jacobian
import matplotlib.pyplot as plt
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC
import gelman_rubin
from write_pdb import write_ensemble, write_VMD_script
import matplotlib.pyplot as plt

mu_prior=3
sigma_squared=0.01
sigma_squared_likelihood=0.2



def log_prior(x):
    x = x.reshape(-1, 3)
    chain_log_prob = np.sum(-(np.sqrt(np.sum((x[1:]-x[:-1])**2, axis=1))-mu_prior)**2/(2*sigma_squared))
    dm = np.sum((x[None, :] - x[:, None]) ** 2, axis=2)
    dm2 = dm[inds_x, inds_y]
    dm2 = dm2[dm2 < mu_prior * mu_prior]
    volume_exclusion_log_prob = -np.sum((dm2 - mu_prior * mu_prior)**2 / (2 * sigma_squared))
    return chain_log_prob + volume_exclusion_log_prob


pairs = np.array([[0, 49],
                  [12, 37]])
distances = np.array([mu_prior, mu_prior])


def log_likelihood(x):
    # mu = np.linalg.norm(x[None,:] - x[:, None], axis=2)
    x = x.reshape(-1, 3)
    mu = np.linalg.norm(x[pairs[:,0]]-x[pairs[:,1]], axis=1)
    return -np.sum((mu - distances) ** 2/2.0 / sigma_squared_likelihood)
    return np.sum(-(np.log(distances/mu)**2/(2*sigma_squared_likelihood)))


def log_posterior(x):
    return log_prior(x) + log_likelihood(x)

gradient_log_posterior = grad(log_posterior)
hessian_log_posterior = jacobian(gradient_log_posterior)



number_of_monomere=50


pair_combinations = np.array(list(combinations(range(number_of_monomere), 2)))
inds_x = pair_combinations[:, 0]
inds_y = pair_combinations[:, 1]

x=np.abs(np.random.normal(0,5,(number_of_monomere,3))).ravel()


# b = int(np.floor(chain_length ** (1.0 / 3)))
# a = int(np.floor(chain_length / b))

def plot_hist_all_dist(chain):
    dist = np.triu(np.linalg.norm(chain[:,None,:]-chain[:,:,None,:], axis=3))
    dist = dist[np.nonzero(dist)]
    plt.hist(dist, bins = 100)
    plt.show()


def plot_hist_data(chain, pair):
    dist = np.linalg.norm(chain[:,pairs[pair,0]]-chain[:,pairs[pair,1]], axis = 1)



if __name__ == "__main__":

    chain_length = 500
    time = 10
    trajectory_length = 10
    stepsize = 0.005

    leapfrog = LeapfrogHMC(log_posterior, gradient_log_posterior, stepsize, trajectory_length)
    chain_leapfrog , acceptancerate = leapfrog.build_chain(x, chain_length)
    print(acceptancerate)

    write_ensemble(chain_leapfrog.reshape(-1, number_of_monomere, 3), "leapfrog_samples.pdb")
    write_VMD_script("leapfrog_samples.pdb", mu_prior/2, number_of_monomere, "leapfrog_script.rc")