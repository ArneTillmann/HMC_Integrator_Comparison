from itertools import combinations
from mpl_toolkits import mplot3d
import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import grad
from autograd import jacobian
import matplotlib.pyplot as plt
from hmc_sampler import LeapfrogHMC
from hmc_sampler import U7HMC
import gelman_rubin
from write_pdb import write_ensemble, write_VMD_script
import matplotlib.pyplot as plt


mu_prior=3
sigma_squared=0.01
sigma_squared_likelihood=0.2
dimension=3
number_of_monomere=50


def log_prior(x):
    x = x.reshape(-1, 3)
    chain_log_prob = np.sum(-(np.sqrt(np.sum((x[1:]-x[:-1])**2, axis=1))-mu_prior)**2/(2*sigma_squared))
    dm = np.sum((x[None, :] - x[:, None]) ** 2, axis=2)
    dm2 = dm[inds_x, inds_y]
    dm2 = dm2[dm2 < mu_prior * mu_prior]
    volume_exclusion_log_prob = -np.sum((dm2 - mu_prior * mu_prior)**2 / (2 * sigma_squared))
    return chain_log_prob + volume_exclusion_log_prob

pairs = np.array([[0, 6],
                  [12, 37],
                  [0, 25],
                  [4, 30],
                  [38, 43]])
distances = np.array([mu_prior, mu_prior, mu_prior, mu_prior, mu_prior])

def log_likelihood(x):
    # mu = np.linalg.norm(x[None,:] - x[:, None], axis=2)
    x = x.reshape(-1, 3)
    mu = np.linalg.norm(x[pairs[:,0]]-x[pairs[:,1]], axis=1)
    return -np.sum((mu - distances) ** 2/2.0 / sigma_squared_likelihood)
    return np.sum(-(np.log(distances/mu)**2/(2*sigma_squared_likelihood)))


def log_posterior(x):
    return log_prior(x) + log_likelihood(x)

gradient_log_posterior = grad(log_posterior)
hessian_log_posterior = jacobian(gradient_log_posterior)



pair_combinations = np.array(list(combinations(range(number_of_monomere), 2)))
inds_x = pair_combinations[:, 0]
inds_y = pair_combinations[:, 1]

x=np.abs(np.random.normal(0,5,(number_of_monomere,3))).ravel()
# x = [[3.92776565e+00 6.32914833e+00 8.63926663e+00]
#  [5.33246358e+00 5.42266052e-01 1.55669378e+00]
#  [5.44660036e+00 1.09703276e+01 1.05734325e+00]
#  [8.63752113e+00 9.60936126e-01 1.90726416e+00]
#  [5.67858017e+00 3.85204630e+00 9.15805571e-02]
#  [4.64714567e+00 4.35682451e-01 1.82413817e+00]
#  [6.83294286e-01 4.32142721e+00 4.21013684e+00]
#  [2.21348668e+00 1.40653129e+01 1.51088640e+00]
#  [1.85387996e+00 2.88681163e+00 4.54883598e+00]
#  [5.37156475e+00 1.19575502e+00 4.25272978e-01]
#  [8.60731654e+00 8.00828616e-01 8.03548460e-01]
#  [4.65184060e+00 5.88934330e+00 3.03460103e+00]
#  [6.42895080e-01 6.49252673e+00 3.80216703e-01]
#  [5.48043155e+00 4.89535516e+00 1.07421782e+00]
#  [3.07661264e+00 8.03807008e+00 1.99541983e+00]
#  [5.92486367e-01 1.53079377e+00 1.26607763e+00]
#  [1.11341239e+00 7.36603801e-01 1.06952492e+00]
#  [1.18866044e+01 7.87257456e-01 5.90299372e+00]
#  [8.76532364e-01 1.40035243e+00 9.87259771e-01]
#  [6.04135420e+00 3.39145124e+00 4.65621534e+00]
#  [2.85741811e+00 3.29231799e+00 5.31105324e+00]
#  [8.81458790e+00 7.43288641e+00 1.02910110e+00]
#  [4.57376938e+00 1.29654011e+00 1.89026748e+00]
#  [8.17672189e-01 1.99679876e+00 2.42272475e-01]
#  [1.46560207e+00 8.77259394e+00 9.97868061e-01]
#  [1.19392743e+00 3.30708078e+00 5.77382804e+00]
#  [5.69758023e-01 2.45072772e+00 7.88091860e+00]
#  [4.62748948e+00 2.29120419e+00 3.39046968e+00]
#  [4.81775080e-01 1.05984238e+00 8.24891211e+00]
#  [5.71660079e+00 1.09221609e+00 1.72650746e+01]
#  [6.44625087e+00 5.39379576e+00 3.85436134e+00]
#  [5.12586663e+00 6.93656203e+00 8.26549037e+00]
#  [2.22317336e+00 1.42235941e+00 6.94234505e+00]
#  [3.68946623e+00 3.77182456e+00 5.68195401e+00]
#  [1.40703550e-01 2.75161557e+00 1.99356427e+00]
#  [1.19841927e+00 3.07413076e+00 2.57494588e+00]
#  [1.20884430e+01 4.19861829e+00 4.45632468e-01]
#  [1.58224182e+00 8.53127564e+00 9.75336790e+00]
#  [4.54942487e-01 2.97664447e+00 2.22136823e+00]
#  [6.69620848e+00 6.40594252e+00 1.83654549e+00]
#  [5.50747542e+00 4.22510544e+00 2.21709337e+00]
#  [2.16275061e+00 7.52900493e+00 7.29132334e+00]
#  [2.42701047e+00 1.09372696e+01 8.38293674e+00]
#  [2.95808628e+00 1.20743271e+01 1.57284773e+00]
#  [1.39118841e+01 3.51939701e+00 7.30769638e-01]
#  [2.57571577e+00 6.27225514e+00 1.26420472e+00]
#  [7.79529665e+00 4.07475887e+00 4.22849829e+00]
#  [1.99550753e+00 2.74376430e+00 1.39083633e+00]
#  [3.81853455e+00 6.51884933e+00 1.88719107e-03]
#  [4.66021715e+00 7.89687751e+00 6.38243078e+00]]

# b = int(np.floor(chain_length ** (1.0 / 3)))
# a = int(np.floor(chain_length / b))

def plot_hist_all_dist(chain):
    dist = np.triu(np.linalg.norm(chain[:,None,:]-chain[:,:,None,:], axis=3))
    dist = dist[np.nonzero(dist)]
    plt.hist(dist, bins = 100)
    plt.show()


def plot_hist_data(chain, pair):
    dist = np.linalg.norm(chain[:,pairs[pair,0]]-chain[:,pairs[pair,1]], axis = 1)
    plt.hist(dist, bins = 100)
    plt.show()



if __name__ == "__main__":
    chain_length = 1000
    time = 10
    trajectory_length = 10
    stepsize = 0.01

    leapfrog = LeapfrogHMC(log_posterior, gradient_log_posterior, stepsize, trajectory_length)
    chain_leapfrog , acceptancerate = leapfrog.build_chain(x, chain_length)
    print(acceptancerate)


    u7 = U7HMC(log_posterior, gradient_log_posterior, hessian_log_posterior, stepsize, trajectory_length)
    chain_u7, acceptancerate = u7.build_chain(x, chain_length)
    print(acceptancerate)



    chain_leapfrog.reshape(chain_length +1, number_of_monomere, dimension)
    chain_leapfrog = chain_leapfrog[:].reshape(chain_length+1, number_of_monomere, dimension)
    print(chain_leapfrog.shape)
    # plot_hist_all_dist(chain_leapfrog)
    # plot_hist_data(chain_leapfrog[:], 0)

    # chain_u7.reshape(-1, number_of_monomere, dimension)
    # chain_u7 = chain_u7[:].reshape(chain_length+1, number_of_monomere, dimension)
    #
    # write_ensemble(chain_leapfrog, "leapfrog_samples.pdb", center=False)
    # write_VMD_script("leapfrog_samples.pdb", mu_prior/2, chain_leapfrog.shape[1], "leapfrog_script.rc")
    #

    # write_ensemble(chain_u7, "u7_samples.pdb", center=False)
    # write_VMD_script("u7_samples.pdb", mu_prior/2, chain_u7.shape[1], "u7_script.rc")
    print(chain_leapfrog[0])
### vmd -e vmd_script.rc
