from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC
import gelman_rubin
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
from sklearn.linear_model import LinearRegression





# number_of_chains = 5
chain_length = 5000
# stepsize_u7 = 0.5
# stepsize_leapfrog = 0.3
dimension = 100
time = 10
trajectory_length = np.arange(1, 50, 3)
stepsize = time/trajectory_length

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
# Z=np.zeros((len(trajectory_length)))
# Z2=np.zeros((len(trajectory_length)))
Z3=np.zeros((len(trajectory_length)))
Z4=np.zeros((len(trajectory_length)))
# Z5=np.zeros((len(trajectory_length)))
# Z6=np.zeros((len(trajectory_length)))
test1=[]
test2=[]
# from rpy2.robjects.packages import importr
# import rpy2.robjects as robjects
# from rpy2.robjects import numpy2ri
# GR = importr('stableGR')
for i in range(len(trajectory_length)):
    chain, acceptance_rate = LeapfrogHMC(log_prob, gradient_log_prob, stepsize[i], trajectory_length[i]).build_chain(x_0, chain_length)

    chain = np.expand_dims(chain, axis=0)

    # Z[i] = gelman_rubin.improved_estimator_PSRF(b, a, 1, chain_length,
    #                                             dimension, chain)
    Z3[i] = acceptance_rate
    # chain = chain.reshape((chain_length+1,dimension))
    # numpy2ri.activate()
    # nr,nc = chain.shape
    # Br = robjects.r.matrix(chain, nrow=nr, ncol=nc)
    #
    # robjects.r.assign("chain", Br)
    # print(robjects.r('dim(chain)'))
    #
    # GR = importr('stableGR')
    #
    # try:
    #     Z5[i] = robjects.r('stable.GR(chain)$mpsrf')
    #     test1.append(True)
    # except:
    #     test1.append(False)

for i in range(len(trajectory_length)):
    chain, acceptance_rate = U7HMC(log_prob, gradient_log_prob, hessian_log_prog, stepsize[i], trajectory_length[i]).build_chain(x_0, chain_length)
    chain = np.expand_dims(chain, axis=0)
    # Z2[i] = gelman_rubin.improved_estimator_PSRF(b, a, 1, chain_length,
    #                                                 dimension, chain)
    Z4[i] = acceptance_rate
    # chain = chain.reshape((chain_length+1,dimension))
    # numpy2ri.activate()
    # nr,nc = chain.shape
    # Br = robjects.r.matrix(chain, nrow=nr, ncol=nc)
    #
    # robjects.r.assign("chain", Br)
    # print(robjects.r('dim(chain)'))
    #
    # GR = importr('stableGR')
    #
    # try:
    #     Z6[i] = robjects.r('stable.GR(chain)$mpsrf')
    #     test2.append(True)
    # except:
    #     test2.append(False)
#
# print(Z6)
# print(Z5)
f=15
f2=20


'''linear regression'''
linear_regressor1 = LinearRegression()
linear_regressor2 = LinearRegression()

linear_regressor1.fit(stepsize[2:].reshape(-1, 1), Z3[2:].reshape(-1, 1))
linear_regressor2.fit(stepsize[2:].reshape(-1, 1), Z4[2:].reshape(-1, 1))
Z3_pred = linear_regressor1.predict(stepsize[2:].reshape(-1, 1))
Z4_pred = linear_regressor2.predict(stepsize[2:].reshape(-1, 1))


plot1 = plt.figure(1)
plt.tick_params(axis='both', labelsize=15)
plt.plot(stepsize[2:], Z3[2:], 's', label='Leapfrog')
plt.plot(stepsize[2:], Z4[2:], '^', label='U7')
plt.plot(stepsize[2:], Z3_pred, '--', color='#1f77b4')
plt.plot(stepsize[2:], Z4_pred, '--', color='#ff7f0e')

plt.xlabel('stepsize = time / trajectory length, time = 10', fontsize=f)
plt.ylabel('acceptance rate', fontsize=f)
plt.title('acceptance rate for normal distribution, n='+str(chain_length), fontsize=f2)
plt.legend(fontsize=f)
# plot2 = plt.figure(2)
# plt.tick_params(axis='both', labelsize=15)
# plt.plot(stepsize, Z, 's', label='Leapfrog')
# plt.plot(stepsize, Z2, '^', label='U7')
# plt.xlabel('stepsize = time / trajectory length, time = 10', fontsize=f)
# plt.ylabel('Gelman-Rubin-statistic', fontsize=f)
# plt.title('Gelman-Rubin-statistic for normal distribution, n='+str(chain_length), fontsize=f2)
# plt.legend(fontsize=f)
# plot3 = plt.figure(3)
# plt.tick_params(axis='both', labelsize=15)
# plt.plot(stepsize[test1], Z5[test1], 's', label='Leapfrog')
# plt.plot(stepsize[test2], Z6[test2], '^', label='U7')
# plt.xlabel('stepsize = time / trajectory length, time = 10', fontsize=f)
# plt.ylabel('old Gelman-Rubin-statistic', fontsize=f)
# plt.title('old Gelman-Rubin-statistic for normal distribution, n='+str(chain_length), fontsize=f2)
# plt.legend(fontsize=f)

plt.show()
