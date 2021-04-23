from mpl_toolkits import mplot3d
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
from autograd import jacobian
import matplotlib.pyplot as plt
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC
import gelman_rubin
from write_pdb import write_ensemble, write_VMD_script

mu_prior=3
sigma_squared=0.1
sigma_squared_likelihood=1
number_of_monomere=30
dimension=3

def log_prior(x):
    x = x.reshape(-1, 3)
    return np.sum(-(np.sqrt(np.sum((x[1:]-x[:-1])**2, axis=1))-mu_prior)**2/(2*sigma_squared))

pairs = np.array([[0, 2],
                  [0, 3]])
distances = np.array([1.0, 1.5])

def log_likelihood(x):
    # mu = np.linalg.norm(x[None,:] - x[:, None], axis=2)
    x = x.reshape(-1, 3)
    mu = np.linalg.norm(x[pairs[:,0]]-x[pairs[:,1]], axis=1)
    return np.sum(-(np.log(distances/mu)**2/(2*sigma_squared_likelihood)))


def log_posterior(x):
    return log_prior(x) #+ log_likelihood(x)

gradient_log_posterior = grad(log_posterior)
hessian_log_posterior = jacobian(gradient_log_posterior)


x=np.abs(np.random.normal(0,1,(number_of_monomere,3))).ravel()

# b = int(np.floor(chain_length ** (1.0 / 3)))
# a = int(np.floor(chain_length / b))


if __name__ == "__main__":
    chain_length = 500
    time = 10
    trajectory_length = 10
    stepsize = 0.2
    leapfrog = LeapfrogHMC(log_posterior, gradient_log_posterior, stepsize, trajectory_length)
    chain_leapfrog , acceptancerate = leapfrog.build_chain(x, chain_length)
    print(acceptancerate)

    # u7 = U7HMC(log_posterior, gradient_log_posterior, hessian_log_posterior, stepsize, trajectory_length)
    # chain_u7, acceptancerate = u7.build_chain(x, chain_length)
    # print(acceptancerate)


    chain_leapfrog.reshape(chain_length +1, number_of_monomere, dimension)
    chain_leapfrog = chain_leapfrog[:].reshape(chain_length+1, number_of_monomere, dimension)

    # chain_u7.reshape(-1, number_of_monomere, dimension)
    # chain_u7 = chain_u7[:].reshape(chain_length+1, number_of_monomere, dimension)


    write_ensemble(chain_leapfrog, "leapfrog_samples.pdb", center=False)
    write_VMD_script("leapfrog_samples.pdb", mu_prior/2, chain_leapfrog.shape[1], "leapfrog_script.rc")


    # write_ensemble(chain_u7, "u7_samples.pdb", center=False)
    # write_VMD_script("u7_samples.pdb", mu_prior/2, chain_u7.shape[1], "u7_script.rc")
