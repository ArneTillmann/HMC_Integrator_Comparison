from mpl_toolkits import mplot3d
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
from autograd import jacobian
import matplotlib.pyplot as plt
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC
import gelman_rubin

l=3
sigma_squared=1
sigma_squared_likelihood=1

def log_prior(x):
    return np.sum(-(np.sqrt(np.sum((x[1:]-x[:-1])**2, axis=1))-l)**2/(2*sigma_squared))


def log_likelihood(x):
    mu = np.linalg.norm(x[None,:] - x[:, None], axis=2)
    return np.sum(-np.triu(np.log(np.triu(D,1)/mu)**2/(2*sigma_squared_likelihood),1))


def log_posterior(x):
    return log_prior(x) + log_likelihood(x)

gradient_log_posterior = grad(log_posterior)
hessian_log_posterior = jacobian(gradient_log_posterior)


number_of_monomere=4
D=np.abs(np.random.normal(0,1,(4,4)))

dimension=3
x=np.abs(np.random.normal(0,1,(4,3)))

mu = np.linalg.norm(x[None,:] - x[:, None], axis=2) + np.eye(number_of_monomere)

print(log_posterior(x))
print(gradient_log_posterior(x))
#print(hessian_log_posterior(x))

chain_length = 500
time = 10
trajectory_length = np.arange(1, 50, 3)
stepsize = time/trajectory_length

x_0 = np.random.uniform(low=-5, high=5, size=number_of_monomere*dimension)
b = int(np.floor(chain_length ** (1.0 / 3)))
a = int(np.floor(chain_length / b))
