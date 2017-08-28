import numpy
from scipy.special import kv

C = None

def run_VB(S_flips, flip_intervals, flip_neuron, gamma, J_mu_prior, lmbda=1.,
           lmbda_theta=1.):
    """ Variational Bayes algorithm for getting the full approximate
    posterior over couplings.

    :param S_flips: numpy.ndarray [Num. of flips + 1 x N+1]
        Binary data with only flips
    :param flip_intervals: numpy.ndarray [Num. of flips + 1]
        Durations between flips
    :param flip_neuron: numpy.ndarray [Num. of flips + 1]
        Index of which spin flipped
    :param gamma: float
        Rate of a spins being updated
    :param J_mu_prior: numpy.ndarray [N+1, N]
        Mean of the prior
    param lmbda: float
        Scaling parameter of the coupling prior
    param lmbda_theta: float
        Inverse std. of the external field prior

    :return:

        J_mu: numpy.ndarray [N+1 x N]
            Mean of the posterior estimate
        J_Sigma: numpy.ndarray [N+1 x N+1 x N]
            Covariance matrices of the posterior
        F_list: numpy.ndarray [Num. of VB iterations]
            Free energy at each VB iteration

    """
    global C
    C = S_flips[:, numpy.newaxis, :] * S_flips[:, :, numpy.newaxis]
    N = S_flips.shape[1] - 1
    T = numpy.sum(flip_intervals)

    # Initialization
    J_mu = J_mu_prior
    diag_idx = numpy.diag_indices(N + 1)
    J_Sigma = numpy.zeros([N + 1, N + 1, N])
    J_Sigma[diag_idx[0], diag_idx[1], :] = .001
    J_Sigma[0, 0, :] = lmbda_theta
    H_mu = numpy.empty([S_flips.shape[0], N])
    H2_mu = numpy.empty([S_flips.shape[0], N])

    for ineuron in range(N):
        H_mu[:, ineuron] = numpy.dot(S_flips, J_mu[:, ineuron])
        J2_mu = J_Sigma[:, :, ineuron] + numpy.outer(J_mu[:, ineuron],
                                                     J_mu[:, ineuron])
        H2_mu[:, ineuron] = numpy.sum(S_flips * numpy.dot(S_flips, J2_mu),
                                      axis=1)

    F_list = []
    F = numpy.inf
    converged = False
    iteration = 0

    # First E step
    rho_mu, omega_mu, beta_mu = VB_estep(S_flips, J_mu, J_Sigma, H_mu, H2_mu,
                                         flip_intervals, gamma, lmbda)

    # VB loop
    while not converged:
        F_old = F
        J_mu, J_Sigma, H2_mu, H_mu = VB_mstep(S_flips, rho_mu, omega_mu,
                                              beta_mu, flip_neuron, lmbda,
                                              J_mu_prior, lmbda_theta)
        rho_mu, omega_mu, beta_mu = VB_estep(S_flips, J_mu, J_Sigma, H_mu,
                                             H2_mu, flip_intervals, gamma,
                                             lmbda)
        F = free_energy(S_flips, flip_intervals, flip_neuron, H_mu, H2_mu,
                        J_mu, J_Sigma, gamma, J_mu_prior, lmbda, lmbda_theta)
        converged = (F_old - F) / N / T < -1
        F_list.append(F)
        iteration += 1
        print(F)
        print('Iteration %d: Free Energy = %f' % (iteration, (F_old - F) / N / T))

    F_list = numpy.array(F_list)
    return J_mu, J_Sigma, F_list


def VB_estep(S_flips, J_mu, J_Sigma, H_mu, H2_mu, flip_intervals, gamma, lmbda):
    """ Variational expectation step (Updating q_2)

    :param S_flips: numpy.ndarray [Num. of flips + 1 x N+1]
        Binary data with only flips
    :param J_mu: numpy.ndarray [N+1 x N]
        Mean of the posterior estimate
    :param J_Sigma: numpy.ndarray [N+1 x N+1 x N]
        Covariance matrices of the posterior
    :param H_mu: numpy.ndarray [Num. of flips+1 x N]
        Expected values of fields.
    :param H2_mu: numpy.ndarray [Num. of flips+1 x N]
        Expected values of squared fields
    :param flip_intervals: numpy.ndarray [Num. of flips + 1]
        Durations between flips
    :param gamma: float
        Rate of a spins being updated
    :param lmbda: float
        Scaling parameter of the coupling prior
    :return:

        rho_mu: numpy.ndarray [Num. of flips + 1 x N]
            Expected value of Poisson variable
        omega_mu: numpy.ndarray [Num. of flips + 1 x N]
            Expected value of Polya-Gamma variable
        beta: numpy.ndarray [N x N]
            Expected value of sparsity variables

    """
    rho_mu = .5 * gamma * numpy.exp(S_flips[:, 1:] * H_mu) / numpy.cosh(
        numpy.sqrt(H2_mu)) * flip_intervals[:, None]
    omega_mu = 1. / 4. / numpy.sqrt(H2_mu) * numpy.tanh(numpy.sqrt(H2_mu))
    mu_J_squared = J_Sigma.diagonal().T + J_mu * J_mu
    beta_mu = 1. / (lmbda * numpy.sqrt(mu_J_squared[1:]))

    return rho_mu, omega_mu, beta_mu


def VB_mstep(S_flips, rho_mu, omega_mu, beta_mu, flip_neuron, lmbda, J_mu_prior,
             lmbda_theta):
    """

    :param S_flips: numpy.ndarray [Num. of flips + 1 x N+1]
        Binary data with only flips
    :param rho_mu: numpy.ndarray [Num. of flips + 1 x N]
        Expected value of Poisson variable
    :param omega_mu: numpy.ndarray [Num. of flips + 1 x N]
        Expected value of Polya-Gamma variable
    :param beta: numpy.ndarray [N x N]
        Expected value of sparsity variables
    :param flip_neuron: numpy.ndarray [Num. of flips + 1]
        Index of which spin flipped
    :param J_mu_prior: numpy.ndarray [N+1, N]
        Mean of the prior
    param lmbda: float
        Scaling parameter of the coupling prior
    param lmbda_theta: float
        Inverse std. of the external field prior
    :return:

        J_mu: numpy.ndarray [N+1 x N]
            Mean of the posterior
        J_Sigma: numpy.ndarray [N+1 x N+1 x N]
            Covariance matrices of posterior
        H2_mu: numpy.ndarray [Num. of flips+1 x N]
            Expected values of squared fields
        H_mu: numpy.ndarray [Num. of flips+1 x N]
            Expected values of fields.
    """
    global C
    N = C.shape[1] - 1
    J_mu = numpy.empty([N + 1, N])
    J_Sigma = numpy.empty([N + 1, N + 1, N])
    J_Sigma_inv_prior = numpy.empty([N + 1, N])
    J_Sigma_inv_prior[1:] = beta_mu * lmbda ** 2
    J_Sigma_inv_prior[0] = lmbda_theta ** 2
    diag_idx = numpy.diag_indices(N + 1)
    H_mu = numpy.empty([S_flips.shape[0], N])
    H2_mu = numpy.empty([S_flips.shape[0], N])
    for ineuron in range(N):
        flip_idx = numpy.where(flip_neuron == ineuron + 1)[0]
        b = -numpy.sum(C[flip_idx, ineuron + 1], axis=0)
        b += numpy.dot(rho_mu[:, ineuron], C[:, ineuron + 1])
        A = 4. * numpy.tensordot(omega_mu[flip_idx, ineuron], C[flip_idx, :],
                                 axes=[0, 0])
        A += 4. * numpy.tensordot((rho_mu[:, ineuron] * omega_mu[:, ineuron]),
                                  C, axes=[0, 0])
        A[diag_idx[0], diag_idx[1]] += J_Sigma_inv_prior[:, ineuron]
        J_Sigma[:, :, ineuron] = numpy.linalg.inv(A)
        J_mu[:, ineuron] = numpy.dot(J_Sigma[:, :, ineuron],
                                     b + J_Sigma_inv_prior[:,
                                         ineuron] * J_mu_prior[:, ineuron])
        H_mu[:, ineuron] = numpy.dot(S_flips, J_mu[:, ineuron])
        J2_mu = J_Sigma[:, :, ineuron] + numpy.outer(J_mu[:, ineuron],
                                                     J_mu[:, ineuron])
        H2_mu[:, ineuron] = numpy.sum(S_flips * numpy.dot(S_flips, J2_mu),
                                      axis=1)

    return J_mu, J_Sigma, H2_mu, H_mu


def free_energy(S_flips, flip_intervals, flip_neuron, H_mu, H2_mu, J_mu, \
                J_Sigma, gamma, J_mu_prior, lmbda, lmbda_theta):
    """ Calculates the free energy of the current posterior.

    :param S_flips: numpy.ndarray [Num. of flips + 1 x N+1]
        Binary data with only flips
    :param flip_intervals: numpy.ndarray [Num. of flips + 1]
        Durations between flips
    :param flip_neuron: numpy.ndarray [Num. of flips + 1]
        Index of which spin flipped
    :param H_mu: numpy.ndarray [Num. of flips+1 x N]
        Expected values of fields.
    :param H2_mu: numpy.ndarray [Num. of flips+1 x N]
        Expected values of squared fields
    :param J_mu: numpy.ndarray [N+1 x N]
        Mean of the posterior
    :param J_Sigma: numpy.ndarray [N+1 x N+1 x N]
        Covariance matrices of posterior
    :param gamma: float
        Rate of a spins being updated
    :param J_mu_prior: numpy.ndarray [N+1, N]
        Mean of the prior
    param lmbda: float
        Scaling parameter of the coupling prior
    param lmbda_theta: float
        Inverse std. of the external field prior

    :return:

        F: float
            Free energy
    """
    N = S_flips.shape[1] - 1
    F = 0
    # Flip parts
    F += numpy.sum(numpy.log(numpy.cosh(numpy.sqrt(
        H2_mu[range(len(flip_intervals) - 1), flip_neuron - 1]))) + numpy.log(
        2.))
    F += numpy.sum(S_flips[range(len(flip_intervals) - 1), flip_neuron] * H_mu[
        range(len(flip_intervals) - 1), flip_neuron - 1])
    F += gamma * numpy.sum(numpy.dot(flip_intervals, 1. - .5 * numpy.exp(
        S_flips[:, 1:] * H_mu) / numpy.cosh(numpy.sqrt(H2_mu))))

    # Sparsity part
    J2_mu = J_Sigma.diagonal().T + J_mu ** 2
    F += numpy.sum(-.25 * numpy.log(J2_mu[1:]) + .5 * numpy.log(
        2. * numpy.pi) - 3. / 2. * numpy.log(lmbda))
    F -= numpy.sum(numpy.log(8. * kv(-.5, numpy.sqrt(J2_mu[1:] * lmbda ** 2))))

    # Prior external field part
    F += .5 * N * numpy.log(
        2 * numpy.pi / lmbda_theta ** 2) + .5 * lmbda_theta ** 2 * numpy.sum(
        J2_mu[0] - 2. * J_mu[0] * J_mu_prior[0] + J_mu_prior[0] ** 2)

    # Coupling part J
    for ineuron in range(N):
        F -= .5 * numpy.linalg.slogdet(
            2. * numpy.pi * numpy.e * J_Sigma[:, :, ineuron])[1]

    return F