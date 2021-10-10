import numpy as np
import math
import matplotlib.pyplot as plt
import copy


class HMC:

    def __init__(self, log_prob, gradient_log_prob,
                 stepsize, trajectory_length):
        """Class to create a HMC object, from which we can build chains.
        Therefore we need a targeted distribution given as the log probabillity
        (log density function), its log gradient and a stepsize and
        trajectory length for the integrator simulating the Hamiltonian dynamics

        IMPORTANT!!!
        I am using the gradient of the log-density function and not
        the gradient of the energy potential, which is usually used
        in Hamiltonien methods (just a matter of sign)

        Arguments:
        - log_prob (algebraic function R^n -> R)
        - gradient_log_prob (algebraic function R^n -> R^n)
        - stepsize (float)
        - trajectory_length (int)
        """

        self.stepsize = stepsize
        self.trajectory_length = trajectory_length
        self.log_prob = log_prob
        self.gradient_log_prob = gradient_log_prob


    def build_chain(self, x_0, chain_length):
        """To build a chain we need a starting point x_0 and tell the HMC
        algorythm how long the chain should be.
        """

        n_accepted = 0
        chain = [x_0]

        for i in range(chain_length):
            state, accept = self.sample(chain[len(chain)-1])
            state = np.array(state)
            chain = np.append(chain, [state], axis=0)
            n_accepted += accept

        acceptance_rate = n_accepted/float(chain_length)

        return chain, acceptance_rate


    def log_p_acc(self, x_old, v_old, x_new, v_new):

        def H(x,v):
            return -self.log_prob(x) + 0.5*np.sum(v**2)

        return min(0, -(H(x_new, v_new) - H(x_old, v_old)))


    def sample(self, x_old):

        v_old = np.random.normal(size=len(x_old))
        x_new, v_new = self.integrate(copy.copy(x_old), copy.copy(v_old))
        accept = (np.log(np.random.random())
                 < self.log_p_acc(x_old, v_old, x_new, v_new))

        if accept:
            return x_new, accept
        else:
            return x_old, accept


    def integrate(self, x, v):
        ...

class U7HMC(HMC):

    def __init__(self, log_prob, gradient_log_prob,
                 hessian_log_prog, stepsize, trajectory_length):
        """For our use, we want to compare two different integrator for
        simulating the Hamiltonian dynamics. One is presented in this paper
        https://arxiv.org/pdf/2007.05308.pdf and is just simply called u7.
        As it is a fourth order intgrator it requires the hessian matrix of the
        targeted log density as well.

        Arguments:
        - log_prob (algebraic function R^n -> R)
        - gradient_log_prob (algebraic function R^n -> R^n)
        - hessian_log_prog (algebraic function R^n -> R^(n x n))
        - stepsize (float)
        - trajectory_length (int)
        """

        super().__init__(log_prob, gradient_log_prob,
        stepsize, trajectory_length)

        self.hessian_log_prog = hessian_log_prog


    def integrate(self, x, v):
        """Note the different calculations to propose a new state of the
        Hamiltonian dynamics. For further information see
        https://arxiv.org/pdf/2007.05308.pdf.
        """

        v += 1./6 * self.stepsize * self.gradient_log_prob(x)

        for i in range(self.trajectory_length-1):
            x += 1./2 * v * self.stepsize
            tmp_grad = self.gradient_log_prob(x)
            v += (2./3 * self.stepsize * (tmp_grad
                + self.stepsize**2/24
                * np.matmul(self.hessian_log_prog(x),tmp_grad)))

            x += 1./2 * v * self.stepsize
            v += 1./3 * self.stepsize * self.gradient_log_prob(x)

        x += 1./2 * v * self.stepsize
        tmp_grad = self.gradient_log_prob(x)
        v += (2./3 * self.stepsize * (tmp_grad
            + self.stepsize**2./24
            * np.matmul(self.hessian_log_prog(x),tmp_grad)))
        x += 1./2 * v * self.stepsize
        v += 1./6 * self.stepsize * self.gradient_log_prob(x)

        return x, v


class LeapfrogHMC(HMC):

    def integrate(self, x, v):
        """The second integrator to which we compare the new u7 is the classical
        Leapfrog.
        """

        v += 1./2 * self.stepsize * self.gradient_log_prob(x)

        for i in range(self.trajectory_length-1):
            x += self.stepsize * v
            v += self.stepsize * self.gradient_log_prob(x)

        x += self.stepsize * v
        v += 1./2 * self.stepsize * self.gradient_log_prob(x)

        return x, v
