import numpy as np
import math
import matplotlib.pyplot as plt
np.random.seed(42)
import copy


class HMC:
    def __init__(self, x_0, chain_length, log_prob, gradient_log_prob, stepsize, trajectory_length):
        self.x_0 = x_0
        self.chain_length = chain_length
        self.stepsize = stepsize
        self.trajectory_length = trajectory_length
        self.log_prob = log_prob
        self.gradient_log_prob = gradient_log_prob



    def build_chain(self):
        n_accepted = 0
        chain = [self.x_0]

        for i in range(self.chain_length):
            state, accept = self.sample(chain[len(chain)-1])
            state = np.array(state)
            chain = np.append(chain, [state], axis = 0)
            n_accepted += accept

        self.acceptance_rate= n_accepted/float(self.chain_length)

        return chain

    def get_acceptance_rate(self):
        return self.acceptance_rate

    def log_p_acc(self, x_old, v_old, x_new, v_new):
        def H(x,v): return -self.log_prob(x) + 0.5*np.sum(v**2)
        return min(0,-(H(x_new, v_new) - H(x_old, v_old)))

    def sample(self, x_old):
        v_old = np.random.normal(size = len(x_old))
        x_new, v_new = self.integrate(copy.copy(x_old), copy.copy(v_old))

        accept = np.log(np.random.random()) < self.log_p_acc(x_old, v_old, x_new, v_new)
        if accept:
            return x_new, accept
        else:
            return x_old, accept


    def integrate(self, x, v):
        ...
class U7HMC (HMC):
    def __init__(self, x_0, chain_length, log_prob, gradient_log_prob, hessian_log_prog, stepsize, trajectory_length):
        super().__init__(x_0, chain_length, log_prob, gradient_log_prob, stepsize, trajectory_length)
        self.hessian_log_prog = hessian_log_prog

    def integrate(self, x, v):
        v += 1/6 * self.stepsize * self.gradient_log_prob(x)
        for i in range(self.trajectory_length -1):
            x += 1/2 * v * self.stepsize
            v += 2/3 * self.stepsize * (self.gradient_log_prob(x) + self.stepsize**2/24 * np.matmul(self.hessian_log_prog(x),self.gradient_log_prob(x)))
            x = x + 1/2 * v * self.stepsize
            v += 1/3 * self.stepsize * self.gradient_log_prob(x)
        x += 1/2 * v * self.stepsize
        v += 2/3 * self.stepsize * (self.gradient_log_prob(x) + self.stepsize**2/24 *np.matmul(self.hessian_log_prog(x),self.gradient_log_prob(x)))
        x += 1/2 * v * self.stepsize
        v += 1/6 * self.stepsize * self.gradient_log_prob(x)
        return x, v

class LeapfrogHMC (HMC):


    def integrate(self, x, v):
        #I am using the gradient of the density function and not
        #the gradient of the energy potential, which is used usually
        #in Hamiltonien methodes (just a matter of sign)
        v += 0.5 * self.stepsize * self.gradient_log_prob(x)
        for i in range(self.trajectory_length-1):
            x = x + self.stepsize*v
            v = v + self.stepsize*self.gradient_log_prob(x)
        x += self.stepsize * v
        v += 0.5 * self.stepsize * self.gradient_log_prob(x)
        return x, v
