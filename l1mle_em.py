'''
This file contains the code for penalized maximum likelihood estimator.

Author: Christian Donner
'''

import numpy

C = None

def run_EM_l1(S_flips, flip_intervals, flip_neuron, gamma, lmbda):
    """ Expecation maximization algorithm for getting the L1 maximum likelihood
    estimate.

    :param S_flips: numpy.ndarray [Num. of flips + 1 x N+1]
            Binary data with only flips
    :param flip_intervals: numpy.ndarray [Num. of flips + 1]
            Durations between flips
    :param flip_neuron: numpy.ndarray [Num. of flips + 1]
            Index of which spin flipped
    :param gamma: float
        Rate of a spins being updated
    param lmbda: float
        Regularization parameter (>0)

    :return:

        J_est: numpy.ndarray [N+1 x N]
            Maximum likelihood estimate of couplings
        llk_list: numpy.ndarray [Num. of EM iterations]
            Penalized log-likelihood at each EM iteration

    """

    # Covariance
    global C
    C = S_flips[:, numpy.newaxis, :] * S_flips[:, :, numpy.newaxis]
    N = S_flips.shape[1] - 1
    T = numpy.sum(flip_intervals)

    # Initialization
    J_est = numpy.zeros([N + 1, N])
    J_est[0] = -.1
    converged = False
    llk = -numpy.inf
    llk_list = []
    iteration = 0

    # EM Loop (Min. iteration because sometimes in the beginning likelihood
    # changes slowly)
    while not converged:
        llk_old = llk
        rho_tau, omega_tau, H_tau, beta = estep_l1(S_flips, J_est, gamma, lmbda)
        J_est = mstep_l1(rho_tau, omega_tau, beta, flip_intervals, flip_neuron,
                     lmbda)
        llk = log_likelihood_l1(S_flips, J_est, flip_intervals, flip_neuron, \
            gamma, lmbda)
        converged = -(llk_old - llk) / N / T < 1e-4 and iteration > 10
        llk_list.append(llk)
        iteration += 1
        print('Iteration %d: Likelihood = %.1f' % (iteration, llk))
    llk_list = numpy.array(llk_list)
    return J_est, llk_list


def estep_l1(S_flips, J, gamma, lmbda):
    """ Computes the expectation step.

    :param S_flips: numpy.ndarray [Num. of flips + 1 x N+1]
        Binary data with only flips
    :param J_est: numpy.ndarray [N+1 x N]
        Current estimate of couplings
    :param gamma: float
        Rate of a spins being updated
    param lmbda: float
        Regularization parameter (>0)

    :return:

        rho_tau: numpy.ndarray [Num. of flips + 1 x N]
            Expected value of Poisson variable
        omega_tau: numpy.ndarray [Num. of flips + 1 x N]
            Expected value of Polya-Gamma variable
        H_tau: numpy.ndarray [Num. of flips + 1 x N]
            The values of the field
        beta: numpy.ndarray [N x N]
            Expected value of sparsity variables
    """
    H_tau = numpy.dot(S_flips, J)
    rho_tau = .5 * gamma * numpy.exp(S_flips[:, 1:] * H_tau) / numpy.cosh(H_tau)
    omega_tau = 1. / 4. / H_tau * numpy.tanh(H_tau)
    beta = numpy.empty(J[1:].shape)
    beta[J[1:] != 0] = 1. / (lmbda * numpy.absolute(J[1:][J[1:] != 0]))
    beta[J[1:] == 0] = 1e10

    return rho_tau, omega_tau, H_tau, beta


def mstep_l1(rho_tau, omega_tau, beta, flip_intervals, flip_neuron, lmbda):
    """ Computes the maximization step.

    :param rho_tau: numpy.ndarray [Num. of flips + 1 x N]
            Expected value of Poisson variable
    :param omega_tau: numpy.ndarray [Num. of flips + 1 x N]
            Expected value of Polya-Gamma variable
    :param beta: numpy.ndarray [N x N]
            Expected value of sparsity variables
    :param flip_intervals: numpy.ndarray [Num. of flips + 1]
            Durations between flips
    :param flip_neuron: numpy.ndarray [Num. of flips + 1]
            Index of which spin flipped
    :param lmbda: float
        Regularization parameter (>0)

    :return:

        J_new: numpy.ndarray [N+1, N]
            New estimate of couplings
    """
    global C
    N = C.shape[1] - 1
    J_new = numpy.empty([N + 1, N])
    diag_idx = numpy.diag_indices(N)
    for ineuron in range(N):
        flip_idx = numpy.where(flip_neuron == ineuron + 1)[0]
        b = -numpy.sum(C[flip_idx, ineuron + 1], axis=0)
        b += numpy.dot(flip_intervals * rho_tau[:, ineuron], C[:, ineuron + 1])
        A = 4. * numpy.tensordot(omega_tau[flip_idx, ineuron], C[flip_idx, :],
                                 axes=[0, 0])
        A += 4. * numpy.tensordot(
            (flip_intervals * omega_tau[:, ineuron] * rho_tau[:, ineuron]), C,
            axes=[0, 0])
        A[diag_idx[0] + 1, diag_idx[1] + 1] += beta[:, ineuron] * lmbda ** 2
        J_new[:, ineuron] = numpy.linalg.solve(A, b)

    return J_new


def log_likelihood_l1(S_flips, J, flip_intervals, flip_neuron, gamma, lmbda):
    """ Computes the penalized log likelihood for given couplings

    :param S_flips: numpy.ndarray [Num. of flips+1 x N+1]
        Binary data with only flips
    :param J: numpy.ndarray [N+1 x N]
        Given couplings
    :param flip_intervals: numpy.ndarray [Num. of flips + 1]
            Durations between flips
    :param flip_neuron: numpy.ndarray [Num. of flips + 1]
            Index of which spin flipped
    :param gamma: float
        Rate of a spins being updated
    :param lmbda: float
        Regularization parameter (>0)

    :return:

        llk: float
            Log-likelihood
    """
    H_tau = numpy.dot(S_flips, J)
    flip_prob = .5 * numpy.exp(-S_flips[:, 1:] * H_tau) / numpy.cosh(H_tau)
    llk = numpy.sum(
        numpy.log(flip_prob[range(len(flip_prob) - 1), flip_neuron - 1]))
    llk -= numpy.sum(numpy.dot(flip_intervals[:], gamma * flip_prob))
    llk += -lmbda * numpy.sum(numpy.absolute(J[1:]))
    return llk