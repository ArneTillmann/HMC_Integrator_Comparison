from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import itertools
l=3
sigma_squared=1
sigma_squared_likelihood=1

def log_prior(x):
    return np.sum(-(np.sqrt(np.sum((x[1:]-x[:-1])**2, axis=1))-l)**2/(2*sigma_squared))
def log_likelihood(x,D):
    mu = np.linalg.norm(x[None,:] - x[:, None], axis=2)
    return np.nansum(np.triu(-np.log(D/mu)**2/(2*sigma_squared_likelihood),1))

D=np.arange(64.0).reshape((4,4,4))
D[1,0,1] = np.nan

x=np.arange(12.0).reshape((4,3))
mu = np.linalg.norm(x[None,:] - x[:, None], axis=2)
print(-np.log(D/mu)**2/(2*sigma_squared_likelihood))
print(log_likelihood(x,D))

# x = np.linspace(-6, 6, 30)
# y = np.linspace(-6, 6, 30)
#
# X, Y = np.meshgrid(x, y)
# Z = log_prior(X,Y)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot_wireframe(X, Y, Z, color='black')
# ax.set_title('wireframe')
# plt.show()
