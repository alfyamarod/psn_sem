"""
Local and population framework
are tested with delay on the synapses
for a 2D dynamical system
"""
from pop_pbn import *
from local_pbn import *

NT = 1  
dtStim = 0.1 / 1000 
time = np.arange(dtStim, NT + dtStim, dtStim)
nt = len(time)  

blocksize = 0.2
nblock = round(blocksize / dtStim)
stimrnge = [-25, 25]

stim1 = np.repeat(np.random.rand(nt // nblock), nblock) 
stim1 = stim1 * (stimrnge[1] - stimrnge[0]) + stimrnge[0]

stim2 = np.repeat(np.random.rand(nt // nblock), nblock)
stim2 = stim2 * (stimrnge[1] - stimrnge[0]) + stimrnge[0]

# Create the 2D stimulus
stim = np.vstack([stim1 - 0.5, -stim2 + 0.5])

A = np.linalg.inv(np.array([[-0.5, 1], [-1, -0.5]]))

# delays in ms
delays = [1, 5, 10]


fig, axs = plt.subplots(3, 1, figsize=(6, 6))  

for idx, dl in enumerate(delays):
    params_local = {
        'N': 200,
        'wmean': np.array([0.1]), 
        'wsig': 0.01,
        'fmax': 100, 
        'alpha': 100, 
        'fmin': 1, 
        'taud': 10,
        'A': A, 
        'mu': 0,
        'tdel': dl
    }

    params_pop = {
        'N': 200,  
        'wmean': np.array([0.1]),  
        'wsig': 0.01,  
        'kappa': 10,  
        'taud': 10, 
        'A': A, 
        'mu': 0, 
        'tdel': dl  
    }

    ol, xhl, xxl = local_framework(stim, dtStim, params_local)
    op, xhp, xxp = population_framework(stim, dtStim, params_pop)

    
    axs[idx].plot(time, xhl[0, :], 'r-', label="Local fram.")
    axs[idx].plot(time, xhl[1, :], 'r-', label="Local fram.")
    axs[idx].plot(time, xhp[0, :], 'b-', label="Population fram.")
    axs[idx].plot(time, xhp[1, :], 'b-', label="Population fram.")
    axs[idx].plot(time, xxp[0, :], 'y-', label="Target", linewidth=2.0)
    axs[idx].plot(time, xxp[1, :], 'y-', label="Target", linewidth=2.0)
    axs[idx].set_title(f'Delay {dl} ms')
    axs[idx].set_xlabel('Time (s)')
    axs[idx].legend(frameon=True)

fig.tight_layout()
plt.show()
