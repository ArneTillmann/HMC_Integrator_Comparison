import numpy as np
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC
import gelman_rubin
from autograd import elementwise_grad as egrad
from autograd import grad
from autograd import jacobian
import pdb
import matplotlib.pyplot as plt
np.seterr(all='raise')
#np.random.seed(39)


number_of_chains = 1
chain_length = 1
stepsize_u7 = 0.2
stepsize_leap = 1.5
trajectory_length_u7 = 10
trajectory_length_leap = 10
dimension = 2

x_0 = 1+np.zeros(dimension)   #may be chosen randomly for different chains
mu = np.random.normal(size=dimension)
print(mu)
cov=np.array([[1., 0.],[0., 10.]])
inv_cov = np.linalg.inv(cov)
def log_prob(x):

    return (-0.5 * (x-mu)@inv_cov@(x-mu))


def gradient_log_prob(x):

    return (-inv_cov@(x-mu))


def hessian_log_prog(x):

    return -inv_cov

#def log_prob(X):
#    return (2+np.cos(X[0]/2)*np.cos(X[1]/2))**5*np.where((X[0]>0)&(10*np.pi>X[0]),1,0)*np.where((X[1]>0)&(10*np.pi>X[1]),1,0)

#gradient_log_prob = grad(log_prob)

#hessian_log_prog = jacobian(gradient_log_prob)

"""creating the chains for the different integrators respectively"""
"""U7"""


u7_HMCs = [U7HMC(log_prob, gradient_log_prob, hessian_log_prog,
                 stepsize_u7, trajectory_length_u7) for _ in range(number_of_chains)]
chainu7_HMCs = []
acceptance_rate_list_u7 = []

for sampler in u7_HMCs:
    x_0 = 10*np.pi*np.random.uniform(-10,10,dimension)
    print(x_0)
    chain, acceptance_rate, simulationx, simulationv = sampler.build_chain(x_0, chain_length)
    chainu7_HMCs.append(chain)
    acceptance_rate_list_u7.append(acceptance_rate)

chainu7_HMCs = np.array(chainu7_HMCs)

"""Leapfrog"""


Leapfrog_HMCs = [LeapfrogHMC(log_prob, gradient_log_prob, stepsize_leap,
                             trajectory_length_leap)
                             for _ in range(number_of_chains)]
chainleapfrog_HMCs = []
acceptance_rate_list_Leapfrog = []

for sampler in Leapfrog_HMCs:
    x_0 = 10*np.pi*np.random.uniform(-10,10,dimension)
    chain, acceptance_rate = sampler.build_chain(x_0, chain_length)
    chainleapfrog_HMCs.append(chain)
    acceptance_rate_list_Leapfrog.append(acceptance_rate)

chainleapfrog_HMCs = np.array(chainleapfrog_HMCs)


"""Evaluation of the inte by using the revised Gelman-Rubin Diagnostic
see https://arxiv.org/pdf/1812.09384.pdf for further notice"""


b = int(np.floor(chain_length ** (1.0 / 3)))
a = int(np.floor(chain_length / b))

print("U7 acceptance_rate :" + str(acceptance_rate_list_u7))
print("Leapfrog acceptance_rate:" + str(acceptance_rate_list_Leapfrog))

# print(chainleapfrog_HMCs)
# print(chainu7_HMCs[:,:10])
# print(chainu7_HMCs[:,chain_length-10:chain_length])

# print("U7 gelman_rubin statistic :" + str(gelman_rubin.improved_estimator_PSRF(
#                                         b, a, number_of_chains, chain_length,
#                                            dimension, chainu7_HMCs)))
# print("Leapfrog gelman_rubin statistic :" + str(gelman_rubin.improved_estimator_PSRF(b, a, number_of_chains, chain_length,
#                                            dimension, chainleapfrog_HMCs)))
