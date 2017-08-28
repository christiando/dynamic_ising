'''
This file contains the code for sampling data for the Ising model in
continuous time and the maximum likelihood estimator.

Author: Christian Donner
'''

import numpy

C = None


def sample_continuous_time_model_Gillespie(N, g, theta, gamma, T, J=None,
                                           sparse_rate=0., seed=None):
    """ Samples continuous time data from kinetic Ising model.

    :param N: int
        Number of spins
    :param g: float
        Standard deviation of couplings will g/sqrt(N)
    :param theta: float
        Value of thetas
    :param gamma: float
        Rate of a spins being updated
    :param T: float
        Length of Data
    param J: numpy.ndarray [N+1 x N]
        Couplings data is sampled from. If None, new couplings are drawn (
        Default=None)
    :param sparse_rate: float
        Probability for zero couplings (Default=0)
    :param seed: int
        numpy seed (Default=None)
    :return:

        S: numpy.ndarray [Num. of updates + 1, N+1]
            Binary data matrix
        J: numpy.ndarray [N+1, N]
            Couplings with external fields
        update_times: numpy.ndarray [Num of updates]
            Update times
        update_idx: numpy.ndarray [Num of updates]
            Index of updated spin

    """

    if not seed == None:
        numpy.random.seed(seed)

    if J is None:
        J = numpy.empty([N + 1, N])
        J[0] = theta  # + .5*numpy.random.randn(N)
        J[1:] = g / numpy.sqrt(N) * numpy.random.randn(N, N)
        J[1:][numpy.random.rand(N, N) < sparse_rate] = 0.
        # J[1:][J[1:] < 1e-2] = 0.

    expected_updates = T * N * gamma
    update_intervals = numpy.random.exponential(1. / (N * gamma), size=[
        int(1.1 * expected_updates)])
    update_times = numpy.cumsum(update_intervals)
    num_updates = numpy.sum(update_times <= T)
    update_times = update_times[:num_updates]
    update_intervals = update_intervals[:num_updates]
    update_intervals = numpy.concatenate(
        [update_intervals, numpy.array([T - update_times[-1]])])
    update_idx = numpy.random.randint(1, N + 1, size=[num_updates])

    rand_numbers = numpy.random.rand(num_updates)
    S = -numpy.ones([num_updates + 1, N + 1])
    S[:, 0] = 1.

    for iupdate in range(num_updates):
        ispin = update_idx[iupdate]
        H = numpy.dot(J[:, ispin - 1], S[iupdate])
        p = numpy.exp(H) / numpy.cosh(H) / 2.
        S[iupdate + 1] = S[iupdate]
        S[iupdate + 1, ispin] = 2. * (rand_numbers[iupdate] < p) - 1.

    return S, J, update_times, update_idx


def prepare_data(S, T, update_times, update_idx):
    """ Function transforming data from sampling function to fitting data (
    removes all updates, where no flip appears).

    :param S: numpy.ndarray [Num. of updates + 1, N+1]
        Binary data array
    :param T: float
        Length of Data
    :param update_times: numpy.ndarray [Num of updates]
        Update times
    :param update_idx: numpy.ndarray [Num of updates]
        Index of updated spin

    :return:

        S_flips: numpy.ndarray [Num. of flips + 1 x N+1]
            Binary data with only flips
        flip_intervals: numpy.ndarray [Num. of flips + 1]
            Durations between flips
        flip_neuron: numpy.ndarray [Num. of flips + 1]
            Index of which spin flipped
    """
    flip_idx = numpy.where(
        numpy.absolute(numpy.sum(numpy.diff(.5 * S, axis=0), axis=1)) == 1)[0]
    S_flips = S[flip_idx + 1]
    S_flips = numpy.concatenate([numpy.array([S[0]]), S_flips])
    flip_times = update_times[flip_idx]
    flip_times = numpy.concatenate(
        [numpy.array([0]), flip_times, numpy.array([T])])
    flip_intervals = numpy.diff(flip_times)
    flip_neuron = update_idx[flip_idx]

    return S_flips, flip_intervals, flip_neuron


def run_EM(S_flips, flip_intervals, flip_neuron, gamma):
    """ Expecation maximization algorithm for getting the maximum likelihood
    estimate.

    :param S_flips: numpy.ndarray [Num. of flips + 1 x N+1]
            Binary data with only flips
    :param flip_intervals: numpy.ndarray [Num. of flips + 1]
            Durations between flips
    :param flip_neuron: numpy.ndarray [Num. of flips + 1]
            Index of which spin flipped
    :param gamma: float
        Rate of a spins being updated

    :return:

        J_est: numpy.ndarray [N+1 x N]
            Maximum likelihood estimate of couplings
        llk_list: numpy.ndarray [Num. of EM iterations]
            Log-likelihood at each EM iteration

    """

    T = numpy.sum(flip_intervals)
    # Compute covariance matrix
    global C
    C = S_flips[:, numpy.newaxis, :] * S_flips[:, :, numpy.newaxis]
    N = S_flips.shape[1] - 1

    # Initialize
    J_est = numpy.zeros([N + 1, N])
    J_est[0] = -1.
    converged = False
    llk = -numpy.inf
    llk_list = []
    iteration = 0

    # EM Loop
    while not converged:
        llk_old = llk
        rho_tau, omega_tau, H_tau = estep(S_flips, J_est, gamma)
        J_est = mstep(rho_tau, omega_tau, flip_intervals, flip_neuron)
        llk = log_likelihood(S_flips, J_est, flip_intervals, flip_neuron, gamma)
        converged = -(llk_old - llk)/T/N < 1e-4
        llk_list.append(llk)
        iteration += 1
        print('Iteration %d: Likelihood = %.1f' %(iteration, llk))

    llk_list = numpy.array(llk_list)
    return J_est, llk_list


def estep(S_flips, J_est, gamma):
    """ Computes the expectation step.

    :param S_flips: numpy.ndarray [Num. of flips + 1 x N+1]
        Binary data with only flips
    :param J_est: numpy.ndarray [N+1 x N]
        Current estimate of couplings
    :param gamma: float
        Rate of a spins being updated

    :return:

        rho_tau: numpy.ndarray [Num. of flips + 1 x N]
            Expected value of Poisson variable
        omega_tau: numpy.ndarray [Num. of flips + 1 x N]
            Expected value of Polya-Gamma variable
        H_tau: numpy.ndarray [Num. of flips + 1 x N]
            The values of the field
    """
    H_tau = numpy.dot(S_flips, J_est)
    rho_tau = .5 * gamma * numpy.exp(S_flips[:, 1:] * H_tau) / numpy.cosh(H_tau)
    omega_tau = 1. / 4. / H_tau * numpy.tanh(H_tau)

    return rho_tau, omega_tau, H_tau


def mstep(rho_tau, omega_tau, flip_intervals, flip_neuron):
    """ Computes the maximization step.

    :param rho_tau: numpy.ndarray [Num. of flips + 1 x N]
            Expected value of Poisson variable
    :param omega_tau: numpy.ndarray [Num. of flips + 1 x N]
            Expected value of Polya-Gamma variable
    :param flip_intervals: numpy.ndarray [Num. of flips + 1]
            Durations between flips
    :param flip_neuron: numpy.ndarray [Num. of flips + 1]
            Index of which spin flipped

    :return:

        J_new: numpy.ndarray [N+1, N]
            New estimate of couplings
    """
    global C
    N = C.shape[1] - 1
    J_new = numpy.empty([N + 1, N])
    for ineuron in range(N):
        flip_idx = numpy.where(flip_neuron == ineuron + 1)[0]
        b = -numpy.sum(C[flip_idx, ineuron + 1], axis=0)
        b += numpy.dot(flip_intervals * rho_tau[:, ineuron], C[:, ineuron + 1])
        A = 4. * numpy.tensordot(omega_tau[flip_idx, ineuron], C[flip_idx, :],
                                 axes=[0, 0])
        A += 4. * numpy.tensordot(
            (flip_intervals * omega_tau[:, ineuron] * rho_tau[:, ineuron]), C,
            axes=[0, 0])
        J_new[:, ineuron] = numpy.linalg.solve(A, b)

    return J_new


def log_likelihood(S_flips, J, flip_intervals, flip_neuron, gamma):
    """ Computes the log likelihood for given couplings

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

    :return:

        llk: float
            Log-likelihood
    """
    H_tau = numpy.dot(S_flips, J)
    flip_prob = .5 * numpy.exp(-S_flips[:, 1:] * H_tau) / numpy.cosh(H_tau)
    llk = numpy.sum(
        numpy.log(flip_prob[range(len(flip_prob) - 1), flip_neuron - 1]))
    llk -= numpy.sum(numpy.dot(flip_intervals[:], gamma * flip_prob))
    return llk