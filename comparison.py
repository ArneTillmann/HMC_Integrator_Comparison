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
# inv_cov = np.array([[ 1.84984923e+01, -1.21639515e+01,  1.47480361e+00, -7.86359700e+00,
#    2.98541181e+00, -8.00269519e+00,  3.80178183e+00, -2.02577223e+00,
#  -4.55767762e+00, -1.82009378e+00],
# [-1.21639515e+01,  9.85955053e+00, -1.62779823e+00,  4.42647356e+00,
#  -2.67779674e+00,  6.23774995e+00, -2.43038967e+00,  2.22521335e-01,
#   3.21783093e+00,  1.44878664e+00],
#  [ 1.47480361e+00, -1.62779823e+00,  3.82846261e+00, -2.07459077e+00,
#    1.60828746e+00, -2.65368838e+00,  2.17959295e+00,  1.83343081e-02,
#   -1.83255287e+00, -2.72634719e+00],
#  [-7.86359700e+00,  4.42647356e+00, -2.07459077e+00,  6.33730234e+00,
#   -1.87212227e+00,  3.99115470e+00, -2.99816665e+00,  1.49772896e+00,
#    1.91916969e+00,  1.43560595e+00],
#  [ 2.98541181e+00, -2.67779674e+00,  1.60828746e+00, -1.87212227e+00,
#    2.90869360e+00, -3.01118314e+00,  1.07050951e+00,  3.76801490e-01,
#   -1.65233028e+00, -1.51410822e+00],
#  [-8.00269519e+00,  6.23774995e+00, -2.65368838e+00,  3.99115470e+00,
#   -3.01118314e+00,  7.74813834e+00, -4.28478185e+00, -6.33559837e-01,
#    3.23189621e+00,  2.01324342e+00],
#  [ 3.80178183e+00, -2.43038967e+00,  2.17959295e+00, -2.99816665e+00,
#    1.07050951e+00, -4.28478185e+00,  4.21531856e+00,  7.40632917e-03,
#   -2.24965883e+00, -1.75615177e+00],
#  [-2.02577223e+00,  2.22521335e-01,  1.83343081e-02,  1.49772896e+00,
#    3.76801490e-01, -6.33559837e-01,  7.40632917e-03,  2.12860451e+00,
#    6.39614682e-03, -2.93000619e-01],
#  [-4.55767762e+00,  3.21783093e+00, -1.83255287e+00,  1.91916969e+00,
#   -1.65233028e+00,  3.23189621e+00, -2.24965883e+00,  6.39614682e-03,
#    3.23128437e+00,  1.60886584e+00],
#  [-1.82009378e+00,  1.44878664e+00, -2.72634719e+00,  1.43560595e+00,
#   -1.51410822e+00,  2.01324342e+00, -1.75615177e+00, -2.93000619e-01,
#    1.60886584e+00,  3.05125845e+00]])


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
