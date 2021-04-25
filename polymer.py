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

l=3
sigma_squared=0.05
sigma_squared_likelihood=1


def log_prior(x):
    x = x.reshape(-1, 3)
    chain_log_prob = np.sum(-(np.sqrt(np.sum((x[1:]-x[:-1])**2, axis=1))-l)**2/(2*sigma_squared))
    dm = np.sum((x[None, :] - x[:, None]) ** 2, axis=2)
    dm2 = np.sqrt(dm[inds_x, inds_y])
    volume_exclusion_log_prob = -np.sum((dm2 - l)**2 / (2 * sigma_squared))
    return chain_log_prob + volume_exclusion_log_prob

pairs = np.array([[0, 2],
                  [0, 3]])
distances = np.array([1.0, 1.5])

def log_likelihood(x):
    # mu = np.linalg.norm(x[None,:] - x[:, None], axis=2)
    x = x.reshape(-1, 3)
    mu = np.linalg.norm(x[pairs[:,0]]-x[pairs[:,1]], axis=1)
    return np.sum((np.log(distances/mu)**2/(2*sigma_squared_likelihood)))


def log_posterior(x):
    return log_prior(x)# + log_likelihood(x)

gradient_log_posterior = grad(log_posterior)
hessian_log_posterior = jacobian(gradient_log_posterior)


number_of_monomere=50

pair_combinations = np.array(list(combinations(range(number_of_monomere), 2)))
inds_x = pair_combinations[:, 0]
inds_y = pair_combinations[:, 1]

dimension=3
x=np.abs(np.random.normal(0,1,(number_of_monomere,3))).ravel()

print(log_posterior(x))
print(gradient_log_posterior(x))
print(hessian_log_posterior(x))

chain_length = 500
time = 10
#trajectory_length = np.arange(1, 50, 3)
#stepsize = time/trajectory_length
trajectory_length = 10 
stepsize = 0.05

x_0 = np.random.uniform(low=-5, high=5, size=number_of_monomere*dimension)
b = int(np.floor(chain_length ** (1.0 / 3)))
a = int(np.floor(chain_length / b))

leapfrog = LeapfrogHMC(log_posterior, gradient_log_posterior, stepsize, trajectory_length)
chain , acceptancerate = leapfrog.build_chain(x, chain_length)
print(acceptancerate)
from write_pdb import write_ensemble, write_VMD_script
write_ensemble(chain, "out2.pdb")
write_VMD_script("out2.pdb", 1.5, number_of_monomere, "out2.rc")

# u7 = U7HMC(log_posterior, gradient_log_posterior, hessian_log_posterior, stepsize, trajectory_length)
# chain, acceptancerate = u7.build_chain(x, chain_length)
# print(acceptancerate)
