import numpy as np
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC

chain_length = 5000
#Länge der Schrittweite im Leapfrog
stepsize = 1.5
#Anzahl der Schritte im Leapfrog
trajectory_length = 10

dimension = 2

x_0 = np.zeros(dimension)
def log_prob(x):
    return -0.5* np.sum(x**2 )
#gradient der log Dichte funktion
def gradient_log_prob(x):
    return -1*x

def hessian_log_prog(x):
    return -1 *np.identity(2)

leapfrog_HMC = LeapfrogHMC(x_0, chain_length, log_prob, gradient_log_prob, stepsize, trajectory_length)
chain, acceptance_rate = leapfrog_HMC.build_chain()
print(chain[:10])

u7_HMC = U7HMC(x_0, chain_length, log_prob, gradient_log_prob, hessian_log_prog, stepsize, trajectory_length)
chain2,  acceptance_rate2 = u7_HMC.build_chain()
print(chain2[:10])
