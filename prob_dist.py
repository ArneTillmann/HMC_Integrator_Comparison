import autograd.numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from autograd import elementwise_grad as egrad
from autograd import grad
from autograd import jacobian
import scipy.stats
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
np.random.seed(42)

def egg_box_likelihood(x):
    return (2+np.cos(x[1]/2)*np.cos(x[1]/2))**5*np.where(10*np.pi>x[1]>0,1,0)*np.where(10*np.pi>x[2]>0,1,0)

X = np.arange(-5, 50, 0.25)
Y = np.arange(-5, 50, 0.25)
X, Y = np.meshgrid(X, Y)
Z = (2+np.cos(X/2)*np.cos(Y/2))**5*np.where((X>0)&(10*np.pi>X),1,0)*np.where((Y>0)&(10*np.pi>Y),1,0)

#print(Z)

surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.show()


class Prob_dist:

    def __init__(self, likelihood, prior, dimension):
        self.likelihood = likelihood
        self.prior = prior
        self.dimension = dimension

    def posterior(self):
        return self.likelihood

    def grad_posterior(self):
        return grad(self.posterior)

    def hessian_posterior(self):
        return jacobian(self.grad_posterior)

    def get_starting_point(self):
        return self.prior.rvs(size=1)

def likelihood(X):
    return (2+np.cos(X[0]/2)*np.cos(X[1]/2))**5

def prior(X):
    return np.where((X[0]>0)&(10*np.pi>X[0]),1,0)*np.where((X[1]>0)&(10*np.pi>X[1]),1,0)

toy_model_1 = Prob_dist(likelihood, prior, 2)

x = np.array([1., 1.])
print(x)
print(likelihood(x))
print(grad(likelihood)(x))
print(jacobian(grad(likelihood))(x))
#print(toy_model_1.grad_posterior()(x))
x1 = 10*np.pi*np.random.uniform(0,1,1)
x2 = 10*np.pi*np.random.uniform(0,1,1)

print(10*np.pi*np.random.uniform(0,1,2))
