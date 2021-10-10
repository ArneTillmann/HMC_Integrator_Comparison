import numpy as np


class Integrator:

    def __init__(self, start_point, start_impuls, stepsize, trajectory_length,
                 gradient_log_prob):
        self.start_point = start_point
        self.start_impuls = start_impuls
        self.stepsize = stepsize
        self.trajectory_length = trajectory_length
        self.gradient_log_prob = gradient_log_prob


    def integrate(self):
        ...

class U7(Integrator):

    def __init__(self, start_point, start_impuls, stepsize, trajectory_length,
                 gradient_log_prob, hessian_log_prog):
        super().__init__(start_point, start_impuls, stepsize, trajectory_length,
                         gradient_log_prob)
        self.hessian_log_prog = hessian_log_prog

    def integrate(self):
          """Note the different calculations to propose a new state of the
          Hamiltonian dynamics. For further information see
          https://arxiv.org/pdf/2007.05308.pdf.
          """

          x=self.start_point
          v=self.start_impuls
          simulationx = np.array([[x]])
          simulationv = np.array([[v]])

          v += 1./6 * self.stepsize * self.gradient_log_prob(x)

          simulationv = np.append(simulationv, [v], axis=0)

          for i in range(self.trajectory_length-1):
              x += 1./2 * v * self.stepsize
              simulationx = np.append(simulationx, [x], axis=0)
              v += (2./3 * self.stepsize * (self.gradient_log_prob(x)
                  + self.stepsize**2/24
                  * np.matmul(self.hessian_log_prog(x),self.gradient_log_prob(x))))

              simulationv = np.append(simulationv, [v], axis=0)
              x += 1./2 * v * self.stepsize
              simulationx = np.append(simulationx, [x], axis=0)
              v += 1./3 * self.stepsize * self.gradient_log_prob(x)
              simulationv = np.append(simulationv, [v], axis=0)

          x += 1./2 * v * self.stepsize
          simulationx = np.append(simulationx, [x], axis=0)
          v += (2./3 * self.stepsize * (self.gradient_log_prob(x)
              + self.stepsize**2./24
              * np.matmul(self.hessian_log_prog(x),self.gradient_log_prob(x))))
          simulationv = np.append(simulationv, [v], axis=0)
          x += 1./2 * v * self.stepsize
          simulationx = np.append(simulationx, [x], axis=0)
          v += 1./6 * self.stepsize * self.gradient_log_prob(x)
          simulationv = np.append(simulationv, [v], axis=0)

          return simulationx, simulationv


class Leapfrog(Integrator):

    def __init__(self, start_point, start_impuls, stepsize, trajectory_length,
                 gradient_log_prob):
        super().__init__(start_point, start_impuls, stepsize, trajectory_length,
                         gradient_log_prob)


    def integrate(self):
        """The second integrator to which we compare the new u7 is the classical
        Leapfrog.
        """
        x=self.start_point
        v=self.start_impuls
        simulationx = np.array([[x]])
        simulationv = np.array([[v]])

        v += 1./2 * self.stepsize * self.gradient_log_prob(x)
        simulationv = np.append(simulationv, [v], axis=0)

        for i in range(self.trajectory_length-1):
            x += self.stepsize * v
            simulationx = np.append(simulationx, [x], axis=0)
            v += self.stepsize * self.gradient_log_prob(x)
            simulationv = np.append(simulationv, [v], axis=0)

        x += self.stepsize * v
        simulationx = np.append(simulationx, [x], axis=0)
        v += 1./2 * self.stepsize * self.gradient_log_prob(x)
        simulationv = np.append(simulationv, [v], axis=0)

        return simulationx, simulationv
