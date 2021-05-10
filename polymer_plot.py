from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC
from polymer import log_posterior, gradient_log_posterior, hessian_log_posterior, x, mu_prior, sigma_squared, sigma_squared_likelihood, dimension, number_of_monomere, pair_combinations, inds_x, inds_y, pairs, distances
import polymer
import gelman_rubin
#from rpy2.robjects.packages import importr
#import rpy2.robjects as robjects
#from rpy2.robjects import numpy2ri
from sklearn.linear_model import LinearRegression



chain_length = 50

time = 0.1
trajectory_length = np.arange(10, 20, 2)
stepsize = time/trajectory_length
print(stepsize)


b = int(np.floor(chain_length ** (1.0 / 3)))
a = int(np.floor(chain_length / b))


X, Y= np.meshgrid(trajectory_length, stepsize)

# Z=np.zeros((len(trajectory_length)))
# Z2=np.zeros((len(trajectory_length)))
Z3=np.zeros((len(trajectory_length)))
Z4=np.zeros((len(trajectory_length)))
# Z5=np.zeros((len(trajectory_length)))
# Z6=np.zeros((len(trajectory_length)))



# from rpy2.robjects.packages import importr
# import rpy2.robjects as robjects
# from rpy2.robjects import numpy2ri
# GR = importr('stableGR')
test1=[]
test2=[]


for i in range(len(trajectory_length)):
    chain, acceptance_rate = LeapfrogHMC(log_posterior, gradient_log_posterior, stepsize[i], trajectory_length[i]).build_chain(x, chain_length)

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


# for i in range(len(trajectory_length)):
    # chain, acceptance_rate = U7HMC(log_posterior, gradient_log_posterior, hessian_log_posterior, stepsize[i], trajectory_length[i]).build_chain(x, chain_length)
    # chain = np.expand_dims(chain, axis=0)
    # Z2[i] = gelman_rubin.improved_estimator_PSRF(b, a, 1, chain_length,
    #                                                 dimension, chain)
    # Z4[i] = acceptance_rate
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

f=15
f2=20

#
# '''linear regression'''
# linear_regressor1 = LinearRegression()
# linear_regressor2 = LinearRegression()
#
# linear_regressor1.fit(stepsize[2:].reshape(-1, 1), Z3[2:].reshape(-1, 1))
# linear_regressor2.fit(stepsize[2:].reshape(-1, 1), Z4[2:].reshape(-1, 1))
# Z3_pred = linear_regressor1.predict(stepsize[2:].reshape(-1, 1))
# Z4_pred = linear_regressor2.predict(stepsize[2:].reshape(-1, 1))


plot1 = plt.figure(1)
plt.tick_params(axis='both', labelsize=15)
plt.plot(stepsize[:], Z3[:], 's--', label='Leapfrog')
plt.plot(stepsize[:], Z4[:], '^--', label='U7')
#plt.plot(stepsize[2:], Z3_pred, '--', color='#1f77b4')
#plt.plot(stepsize[2:], Z4_pred, '--', color='#ff7f0e')

plt.xlabel('stepsize = time / trajectory length, time = 10', fontsize=f)
plt.ylabel('acceptance rate', fontsize=f)
plt.title('normal distribution', fontsize=f2)
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
