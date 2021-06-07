# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Comparing two integrators in the usecase of HMC
# In this blog post we will compare the performance of two different integrators, sampling with Hamiltonian Monte Carlo (HMC). In HMC, the classical Leapfrog is still the state of the art method to generate new proposal states for the Metropolis Hastings algorythm. In [this paper](https://arxiv.org/pdf/2007.05308.pdf) Jun Hao Hue et al. benchmark the performance of the Leapfrog and U7, which was first presented by [Chau et al](https://iopscience.iop.org/article/10.1088/1367-2630/aacde1/pdf), against various classical and quantum systems, but not against sampling methods used in HMC.
# This blog post aims to give you first a brief introduction to HMC, then describes formally the two integrators and then numerically evaluates their performance. In case you are new to HMC or MCMC methods, a good starting point might be the [series of blog posts](https://www.tweag.io/blog/2019-10-25-mcmc-intro1/) by Simeon Carstens. 

# # Introduction
# Broadly speaking, the idea of HMC is, that given a previous state $x$ of our markov chain, we draw a random momentum $v$ from a normal distribution and simulate the behaviour of a fictive particle with starting point $(x,v)$. This intuition makes sense, because our probability density gives rise to a potential energy and vice versa \fotenote Simeons_blog_post. This deterministic behaviour is simulated for some fixed time $t$. The final state $(x^\star, v^\star)$ of the particle after some time t will then serve as the new proposal state of the Metropolis-Hastings algorithm.
#
# You might wonder, why we can draw the initial momentum $v$ from a normal distribution. You can check yourself, if you have a closer look at the joint probability distribution
# $$
# \begin{align}
#  p(x,v) \propto & \,\exp \{-H(x,v)\} \\
# = & \,\exp \{-K(v)\} \times \exp \{-E(x)\}\\
#  =& \,\exp \{-\frac{|v|^2}{2} \} \times p(x).
# \end{align}
# $$
# Since we just multiply the two densities for $x$ and $v$, by definition, the two variables are independent from each other, therefore we can just sample $v$ independently of $x$. Moreover, you can see that the marginal distribution of $v$ is just the normal distribution up to some constant ($\propto$ means, that $A\propto B \iff A = kB$ for an abritrary $k$).
#
# The Hamiltonian is then given by $$H(x,v) = K(v) + E(x).$$ Usually the Hamiltonian does not need to be of the form kinetic energy plus potential energy (seperable), but it certainly makes things a lot easier and is a reasonable assumption for most usecases arising from Bayesian statistics and beyond. In fact, you can actually apply splitting methods, which we will do in the following, to nonseperable Hamiltonians as well, though some more math will be necessary. 
#
# As mentioned aboved, the key concept of the HMC method is to simulate the fictive particle's dynamics. In terms of the Hamiltonian formalism, this is done by solving a set of differential equations, namely
# $$
# \frac{dx}{dt} = \frac{\partial H }{\partial v}, \hspace{15pt} \frac{dv}{dt}= -\frac{\partial H }{\partial x}
# $$
# The notation can be simplified by introducing $z=(x,v)$. Then the Hamiltonian equations of motion can be rewritten in a single expression as 
# $$
# \dot{z} = \begin{pmatrix}\dot x \\ \dot v\end{pmatrix} = \begin{pmatrix}\frac{\partial H }{\partial v} \\ -\frac{\partial H }{\partial x}\end{pmatrix} = \{z,H(z)\}
# $$
# where $\{\cdot, \cdot \}$ is the [Poisson bracket](https://en.wikipedia.org/wiki/Poisson_bracket). What the poisson bracket essentially does is to compactly connect the two derivatives of $H$ with respect to $x$ and $v$ in one expression. The real strength lies in the compatibility of generalized coordinates. In that case you want to derive with respect to a new axis which could point in a new direction, but this is not of concern for our purposes. Furthermore, by introducing an operator $D_{H\cdot} = \{ \cdot, H \}$, the equation can be further simplified to 
# $$
# \dot z = D_H z
# $$
# and is solved by
# $$
# z(t) = \exp ( t D_H) z(0) = \exp ( t(D_K + D_E)) z(0).
# $$ 
# Note that $D_K$ and $D_E$ stand respectivly for the operators $\{ \cdot, K \}$ and $\{ \cdot, E \}.$
# In general, the solution depends highly on the potential energy (which translates to the posterior distribution in the Bayesian setting) and can therefore be infeasable. 

# <!-- 
# This system of coupled differential equations is formally solved by
#
# \begin{equation}
# \begin{pmatrix}x \\v\end{pmatrix} = e^{-tH_v} \begin{pmatrix}x \\v\end{pmatrix} ,
# \end{equation}
#
# where we define $H_v=\frac{\partial H}{\partial x}\frac{\partial}{\partial v}-\frac{\partial H}{\partial v}\frac{\partial}{\partial x}.$
#
# -->
# comment

# <!--
# #### But wait!
#
# I dont want to loose any of you right on the spot, just because I used the term differential equation. It is actually much simpler than that, so don't worry. Just let me show you one simple remarkable result.
#
# In our case of the seperable Hamiltonian, we can rewrite the differential equations, by just using the terms, that are actually dependend on the variable, we want to take the derivativ of, so
# $$\frac{dx}{dt} = \frac{\partial}{\partial v} K(v) = \frac{v}{m}, \hspace{15pt} \frac{dv}{dt}= -\frac{\partial}{\partial x} E(x).$$
#
# This might look already much more familiar to you. Just multiply the left equation with m and take once more the derivate with respect to $t$ gives us $m\frac{d^2x}{dt^2}  = \frac{dv}{dt}.$ This then plugt into the second equation 
# $$m\frac{d^2x}{dt^2}  = \frac{dv}{dt} = -\frac{\partial}{\partial x} E(x) = F$$
# leaves us with Newton's beautiful second law of motion.
#
# -->
# footnote

# ## Splitting methods
# In 1995 [Suzuki](http://people.ucalgary.ca/~dfeder/535/suzuki.pdf) proposed a new way to approximate expressions such as the solution of the Hamiltonian equations,
# $$
# \exp ( t(D_K + D_E)) = \prod_{i=1}^{k/2} \exp (c_i t D_K) \exp (d_i t D_E) + \cal{O}(t^{k+1}),
# $$
# where $\Sigma_{i=1}^k c_i = \Sigma_{i=1}^k d_i =1.$ You can think of this fomular as a generalization of the identity $b^{m+n} = b ^m \cdot b^n.$ The error term is a result of the fact, that operators, such as matrices, generally don't commute. 
#
# Each of the factors $\exp (c_i t D_K)$ will correspond to an update of the state $x$ and similarly $\exp (c_i t D_E)$ can be translated to an update of the momentum $v.$ [^1]
#
# Imagine $(x^\star,v^\star)$ would be the exact solution after time $t$ and $(x_{t},v_{t})$ an approximation, then we would say, that the approximation is of n-th order and write $\mathcal{O}(t^n)$, if $||(x^\star,v^\star)-(x_{t},v_{t})||\leq C * t^n$ and $C$ is independend of the $t.$
#
# Now, that we know how to come up with an approximation of the solution of the Hamiltonian equations, lets give a first example of one approximative algorythm:

# ## The Leapfrog
#
# <!-- 
# For now, let's come back to our fictive particle. As so often the differential eqaution is actually not always analytically tractable and so we need a way to approximate the behaviour of our fictive particle. And this is where the leapfrog method comes into play. 
# --> 
# The intuition is, that we update the space coordinate $x$ and the momentum variable $v$ one after an other multiple times, which is the behaviour, where the name comes from.
#
# ![](Leapfrog.gif)
#
# More rigorously, the updates look like the following,
# $$
# x_{i+1}= x_n +   v_{i + 1/2} \Delta t
# $$
# $$
# v_{i + 3/2} = v_{i+1/2} + \left(-\frac{\partial}{\partial x} E(x_{i+1})\right) \Delta t
# $$
#
# As you might have noticed, you need to perform half a step for the momentum in the beginning and the end. 
# So in terms of Suzuki, the Leapfrog looks like this
# $$
# \text{Leapfrog} =  U_3 U_3 \cdots U_3, 
# $$
# where 
# $$
# U_3 = \exp (\frac {1}{2}\Delta t D_E)\exp (\Delta t D_K)\exp (\frac {1}{2}\Delta t D_E).
# $$
# The coefficients are $c_1 = 0,\, c_2 = 1,\, d_1=d_2 = \frac{1}{2}.$
#
# If we further devide our time $t$ into $t = stepsize * trajectory\_length$ and apply the Suzuki aproximation $U3$ $trajectory\_length$-times, then a small function, approximating the desired behaviour for some time $t$, would have the following look:

def integrate(x, v):

    v += 1./2 * stepsize * -gradient_pot_energy(x)

    for i in range(trajectory_length-1):
        x += stepsize * v
        v += stepsize * gradient_pot_energy(x)

    x += stepsize * v
    v += 1./2 * stepsize * gradient_pot_energy(x)

    return x, v


# You might wonder, why should we look further, if we have found a sufficiently exact approximation of what we desired. We can always deminish the error by shortening the $stepsize$ and increasing the $trajectory\_length.$
#
# Well, one answer might simply be, what if there is a more efficient way. But secondly, the Leapfrog is already not exact even for constant force $- \nabla E = \text{const.}$
#
# Evolutionary, that is why first a five factor approximation was considered by [Chau et al](https://iopscience.iop.org/article/10.1088/1367-2630/aacde1/pdf). Unfortunately this approximation has still error terms of third order. And this in turn was reason for the development of:

# ## The $U_7$
# The novelty of the $U_7$ consists of the usage of the second order derivative of the potential energy. This comes along with a few more updates of $x$ and $v$ per step and yields to a lower error. 
#
# Concretly, the $U_7$, as the name suggests, is originally a seven factor approximation, but uses a special [property](https://en.wikipedia.org/wiki/Baker%E2%80%93Campbell%E2%80%93Hausdorff_formula), which reduces two factors to one and is applied twice, making it a five factor approximation. This newly formed term includs the second order derivative as mentioned above. One way to think about this is, that when we want to apply $e ^A \cdot e^B= e^{A+B}$ to operators we have to take into account, that they don't commute. So basically this identity does not hold for the general case, but what we can do is, that we can use a Taylor-expansion like approximation and take the first few terms of that series $e ^A \cdot e^B= e^{A+B + f_1(A,B) + f_2(A,B) + \cdots}$. Like the Taylor expansion, this approximation also uses the derivative in the higher terms.
#
# Eventhought we could reduce the number of factors the approximation remains an accuracy with error order $\cal{O}(t^5)$. To verify this check out [Siu A. Chin work](https://www.sciencedirect.com/science/article/pii/S0375960197000030). Consequently the $U_7$ is exact up to fourth order and is therfore said to be a fourth-order approximation.[^2] 
#
# The Suzuki formula for the $U_7$ is given by
#
# $$U_7 = \exp (\frac {1}{6}t D_E)\exp (\frac {1}{2}t D_K)\exp (\frac {2}{3}t D_\tilde{V})\exp ( \frac {1}{2}t D_K)\exp (\frac {1}{6}t D_E),$$
#
# whereas $D_\tilde V = \{\cdot , V + \frac{1}{48}[t\nabla V ]^2\}.$ 
#
# A simplified python method of the algorythm described above would look like this:

def integrate(x, v):

    v += 1./6 * stepsize * gradient_pot_energy(x)

    for i in range(trajectory_length-1):
        x += 1./2 * v * stepsize
        v += (2./3 * stepsize * (gradient_pot_energy(x)
            + stepsize**2./24
            * np.matmul(hessian_log_prog(x),gradient_pot_energy(x))))
        x += 1./2 * v * stepsize
        v += 1./3 * stepsize * gradient_pot_energy(x)

    x += 1./2 * v * stepsize
    v += (2./3 * stepsize * (gradient_pot_energy(x)
        + stepsize**2./24
        * np.matmul(hessian_log_prog(x),gradient_pot_energy(x))))
    x += 1./2 * v * stepsize
    v += 1./6 * stepsize * gradient_pot_energy(x)

    return x, v

# Bare in mind, that the higher accuracy, achieved with the $U_7$, comes along with a non-negalectable, additional computational cost. 
#
# Note also, the longer the stepsize, the larger the error and the less likely the acceptance of the proposed state for the MCMC algorythm.

# ## Comparison 
# As mentioned in the beginning, our goal was, to compare the leapfrog with the U7 to draw samples from a probability distribution. For this purpose we chose two different toy examples and two methods of convergence diagnostics to measure the fit of the drawn samples. 
#
# The first example is a 100 dimensional standard normal distribution. We fixed the time to 10 and plotted the acceptance rate for different stepsizes against each other. The expected higher acceptance rate, due to a better approximation of the deterministic behaviour, can easily be observed. 

# ![title](Figure_2.png)

# Unexpected peak at stepsize 1,5 might be explained by the hidden hamiltonian, but we could not figure out for sure, why this is the case. 

# summary: hessian sparse Zeitgewinn

# ## Footnotes
# 1. This algorythm in operator notation has a quite simple representation in canonical coordinates. Since $D_K^2 z = \{\{z,K\}K\} = \{(\dot  x, 0), K  \} = (0,0)$, the taylor expansion of $\exp (a D_K) $ reduces to $\Sigma_{n=0}^{\infty} \frac{(a D_K)^n}{n!} = 1+ aD_K$. This in turn makes $\exp(\beta_i t D_K)$ the mapping $$\begin{pmatrix}x \\v\end{pmatrix} \mapsto \begin{pmatrix}x + t \beta_i \frac{\partial K}{\partial v} (v)\\v\end{pmatrix}$$ and $\exp(\beta_i t D_K)$ gives $$\begin{pmatrix}x \\v\end{pmatrix} \mapsto \begin{pmatrix}x \\v - t \beta_i \frac{\partial E}{\partial x} (x)\end{pmatrix}.$$ 
# 2. One interesting remark is that, for symmetric approximations, $U(t)U(-t) = 1$, the error terms can't be of even order since then, intuitively speaking, the error would point in the same direction, because $t^{2n} = (-t)^{2n}$. 

#
