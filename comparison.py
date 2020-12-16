import numpy as np
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC
import gelman_rubin


number_of_chains = 2
chain_length = 5000
stepsize = 1.5
trajectory_length = 10
dimension = 2

x_0 = np.zeros(dimension)   #may be chosen randomly for different chains


def log_prob(x):

    return -0.5 * np.sum(x**2)


def gradient_log_prob(x):

    return -x


def hessian_log_prog(x):

    return -np.identity(2)


"""creating the chains for the different integrators respectively"""
u7_HMCs = [U7HMC(log_prob, gradient_log_prob, hessian_log_prog,
                 stepsize, trajectory_length) for i in range(number_of_chains)]
chainu7_HMCs = []

for i in range(number_of_chains):
    chain, acceptance_rate = u7_HMCs[i].build_chain(x_0, chain_length)
    chainu7_HMCs.append(chain)

chainu7_HMCs = np.array(chainu7_HMCs)


Leapfrog_HMCs = [LeapfrogHMC(log_prob, gradient_log_prob, stepsize,
                             trajectory_length)
                             for i in range(number_of_chains)]
chainleapfrog_HMCs = []

for i in range(number_of_chains):
    chain, acceptance_rate = Leapfrog_HMCs[i].build_chain(x_0, chain_length)
    chainleapfrog_HMCs.append(chain)

chainleapfrog_HMCs = np.array(chainleapfrog_HMCs)


"""Evaluation of the inte by using the revised Gelman-Rubin Diagnostic
see https://arxiv.org/pdf/1812.09384.pdf for further notice
"""
b = int(np.floor(chain_length**(1.0/3)))
a = int(np.floor(chain_length/b))

print(gelman_rubin.improved_estimator_PSRF(b, a, number_of_chains, chain_length,
                                           dimension, chainu7_HMCs))
print(gelman_rubin.improved_estimator_PSRF(b, a, number_of_chains, chain_length,
                                           dimension, chainleapfrog_HMCs))
