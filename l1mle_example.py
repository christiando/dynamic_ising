'''
Author: Christian Donner
'''

from matplotlib import pyplot
from mle_em import sample_continuous_time_model_Gillespie, prepare_data
from l1mle_em import run_EM_l1

N = 20
T = 100.
gamma = 100.
theta = 0.
g = .3
lmbda = 1.

print('Sample Data')
S, J, update_times, update_idx = sample_continuous_time_model_Gillespie(N, g,
                                        theta, gamma, T, sparse_rate=.5, seed=1)
print('Prepare Data')
S_flips, flip_intervals, flip_neuron = prepare_data(S, T, update_times,
                                                    update_idx)
print('Run penalized EM')
J_est, llk_list = run_EM_l1(S_flips, flip_intervals, flip_neuron, gamma, lmbda)



pyplot.rcParams['font.size'] = 5
pyplot.rcParams['xtick.labelsize'] = 5
pyplot.rcParams['ytick.labelsize'] = 5
pyplot.rcParams['lines.markersize'] = .5
pyplot.rcParams['lines.linewidth'] = .5
pyplot.rcParams['ytick.major.pad'] = 1.
pyplot.rcParams['xtick.major.pad'] = 1.
pyplot.rcParams['axes.labelpad'] = 2.
pyplot.rc('text', usetex=True)
pyplot.rc('font', family='serif')

fig = pyplot.figure('Fig1', figsize=(3+3/8,.5*(3+3/8)))
ax1 = fig.add_axes([.12,.15,.35,.7])
ax1.plot([-.25,.25],[-.25,.25],color=[.5,.5,.5])
ax1.plot(J[1:].flatten(),J_est[1:].flatten(),'ko')
ax1.plot(J[0],J_est[0],'rv')
ax1.set_xlabel('$J$')
ax1.set_ylabel('$J_{est}$')
ax2 = fig.add_axes([.595,.15,.35,.7])
ax2.plot(llk_list*1e-5,marker='s',color='k')
ax2.set_ylabel('$\\ln \mathcal{L}\ [10^{5}]$')
ax2.set_xlabel('EM Iteration')
ax1a = fig.add_axes([.01,.95,.05,.05], frameon=0)
ax1a.set_xticks([])
ax1a.set_yticks([])
ax1a.text(.0,.0,'\\textbf{(a)}')
ax2b = fig.add_axes([.475,.95,.05,.05], frameon=0)
ax2b.set_xticks([])
ax2b.set_yticks([])
ax2b.text(.0,.0,'\\textbf{(b)}')

pyplot.show()