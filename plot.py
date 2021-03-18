from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC
import gelman_rubin


# number_of_chains = 5
chain_length = 5000
# stepsize_u7 = 0.5
# stepsize_leapfrog = 0.3
dimension = 100
time = 10
trajectory_length = np.arange(1, 50, 3)
stepsize = trajectory_length/time

x_0 = np.random.uniform(low=-5, high=5, size=dimension)
# x_0 = np.zeros(dimension)   #may be chosen randomly for different chains
mu = np.random.normal(size=dimension)
# A = np.random.rand(dimension, dimension)
# cov = np.dot(A, A.transpose())
# inv_cov = np.linalg.inv(cov)

b = int(np.floor(chain_length ** (1.0 / 3)))
a = int(np.floor(chain_length / b))

inv_cov = np.eye(dimension) * 0.75

def log_prob(x):

    return (-0.5 * (x-mu)@inv_cov@(x-mu))


def gradient_log_prob(x):

    return -inv_cov@(x-mu)


def hessian_log_prog(x):

    return -inv_cov


def f(x, y):
   return np.sin(np.sqrt(x ** 2 + y ** 2))


X, Y= np.meshgrid(trajectory_length, stepsize)
Z=np.zeros((len(trajectory_length),len(trajectory_length)))
Z2=np.zeros((len(trajectory_length),len(trajectory_length)))
Z3=np.zeros((len(trajectory_length),len(trajectory_length)))
Z4=np.zeros((len(trajectory_length),len(trajectory_length)))

for i in range(len(trajectory_length)):
    for j in range(len(stepsize)):
        chain, acceptance_rate = LeapfrogHMC(log_prob, gradient_log_prob, stepsize[j], trajectory_length[i]).build_chain(x_0, chain_length)
        chain = np.expand_dims(chain, axis=0)
        Z[i,j] = gelman_rubin.improved_estimator_PSRF(b, a, 1, chain_length,
                                                   dimension, chain)
        Z3[i,j] = acceptance_rate

for i in range(len(trajectory_length)):
    for j in range(len(stepsize)):
        chain, acceptance_rate = U7HMC(log_prob, gradient_log_prob, hessian_log_prog, stepsize[j], trajectory_length[i]).build_chain(x_0, chain_length)
        chain = np.expand_dims(chain, axis=0)
        Z2[i,j] = gelman_rubin.improved_estimator_PSRF(b, a, 1, chain_length,
                                                       dimension, chain)
        Z4[i,j] = acceptance_rate

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.plot_wireframe(X, Y, Z2, color='red')
ax.set_title('wireframe')
ax.set_xlabel('trajectory_length')
ax.set_ylabel('stepsize')
ax.set_zlabel('acceptance_rate')

fig2 = plt.figure()
ax2 = plt.axes(projection='3d')
ax2.plot_wireframe(X, Y, Z3, color='black')
ax2.plot_wireframe(X, Y, Z4, color='red')
ax2.set_title('wireframe')
ax2.set_xlabel('trajectory_length')
ax2.set_ylabel('stepsize')
ax2.set_zlabel('acceptance_rate')
plt.show()
