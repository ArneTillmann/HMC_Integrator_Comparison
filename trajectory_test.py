import numpy as np
from only_integrators import U7
from only_integrators import Leapfrog
import matplotlib.pyplot as plt

np.random.seed(42)


n=100
stepsize_u7=1.5
stepsize_leap=1.5
trajectory_length_u7=10
trajectory_length_leap=10

mu=0
dimension=1
#cov=np.array([[1., 0.],[0., 10.]])
#inv_cov = np.linalg.inv(cov)
cov=np.array([2.])
inv_cov=cov**-1
def log_prob(x):

    return (-0.5 * (x-mu)@inv_cov@(x-mu))

def gradient_log_prob(x):

    return (-inv_cov*(x-mu))


def hessian_log_prog(x):

    return -inv_cov

def log_p_acc(x_old, v_old, x_new, v_new):

    def H(x,v):
        return log_prob(x) + 0.5*np.sum(v**2)

    r = min(0, -(H(x_new, v_new) - H(x_old, v_old)))
    #pdb.set_trace()
    return r



start_points = 100*np.random.uniform(-1,1,n)
start_impuls = np.random.normal(0,1,n)

u7s = [U7(start_points[0], start_impuls[i], stepsize_u7, trajectory_length_u7,
         gradient_log_prob, hessian_log_prog) for i in range(n)]

end_points_u7 = []
end_impuls_u7 = []
for integrator in u7s:
    simulationx, simulationv = integrator.integrate()
    end_points_u7 = np.append(end_points_u7, simulationx[-1], axis=0)
    end_impuls_u7 = np.append(end_impuls_u7, simulationv[-1], axis=0)


leapfrogs = [Leapfrog(start_points[0], start_impuls[i], stepsize_leap,
                      trajectory_length_leap,
                      gradient_log_prob) for i in range(n)]

end_points_leapfrog = []
end_impuls_leapfrog = []
for integrator in leapfrogs:
    simulationx, simulationv = integrator.integrate()

    end_points_leapfrog = np.append(end_points_leapfrog, simulationx[-1], axis=0)
    end_impuls_leapfrog = np.append(end_impuls_leapfrog, simulationv[-1], axis=0)

plt.hist(end_points_u7, bins=20, color="red")
plt.hist(end_points_leapfrog, bins=20)
plt.show()
print(start_points[0])
print(sum(end_points_u7-end_points_u7.mean()))
print(sum(end_points_leapfrog-end_points_leapfrog.mean()))
