import numpy as np
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC
import gelman_rubin

number_of_chains = 5
chain_length = 50
# stepsize_u7 = 0.5
# stepsize_leapfrog = 0.3
time = 20
trajectory_length_u7 = 15
stepsize_u7 = trajectory_length_u7/time

trajectory_length_leapfrog = 20
stepsize_leapfrog = 1.0

dimension = 100

x_0 = np.random.uniform(low=-5, high=5, size=dimension)
# x_0 = np.zeros(dimension)   #may be chosen randomly for different chains
mu = np.random.normal(size=dimension)
# A = np.random.rand(dimension, dimension)
# cov = np.dot(A, A.transpose())
# inv_cov = np.linalg.inv(cov)


inv_cov = np.eye(dimension) * 0.75

def log_prob(x):

    return (-0.5 * (x-mu)@inv_cov@(x-mu))


def gradient_log_prob(x):

    return -inv_cov@(x-mu)


def hessian_log_prog(x):

    return -inv_cov




"""creating the chains for the different integrators respectively"""
u7_HMCs = [U7HMC(log_prob, gradient_log_prob, hessian_log_prog,
                 stepsize_u7, trajectory_length_u7) for _ in range(number_of_chains)]
chainu7_HMCs = []
acceptance_rate_list_u7 = []

for sampler in u7_HMCs:
    chain, acceptance_rate = sampler.build_chain(x_0, chain_length)
    chainu7_HMCs.append(chain)
    acceptance_rate_list_u7.append(acceptance_rate)

chainu7_HMCs = np.array(chainu7_HMCs)


Leapfrog_HMCs = [LeapfrogHMC(log_prob, gradient_log_prob, stepsize_leapfrog,
                             trajectory_length_leapfrog)
                             for i in range(number_of_chains)]
chainleapfrog_HMCs = []
acceptance_rate_list_Leapfrog = []

for sampler in Leapfrog_HMCs:
    chain, acceptance_rate = sampler.build_chain(x_0, chain_length)
    chainleapfrog_HMCs.append(chain)
    acceptance_rate_list_Leapfrog.append(acceptance_rate)

chainleapfrog_HMCs = np.array(chainleapfrog_HMCs)
print(chainleapfrog_HMCs.shape)

"""Evaluation of the inte by using the revised Gelman-Rubin Diagnostic
see https://arxiv.org/pdf/1812.09384.pdf for further notice
"""
b = int(np.floor(chain_length ** (1.0 / 3)))
a = int(np.floor(chain_length / b))

print("GR U7", gelman_rubin.improved_estimator_PSRF(b, a, number_of_chains, chain_length,
                                           dimension, chainu7_HMCs))
print("GR LF", gelman_rubin.improved_estimator_PSRF(b, a, number_of_chains, chain_length,
                                           dimension, chainleapfrog_HMCs))
print(acceptance_rate_list_u7)
print(acceptance_rate_list_Leapfrog)
