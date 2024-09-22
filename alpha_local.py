"""
Effect of alpha on the accuracy
of local model
For this example we want to aprrox
a 1D integrator dx/dt = c(t)
"""

from local_pbn import *

NT = 1  
dtStim = 0.1 / 1000 
time = np.arange(dtStim, NT + dtStim, dtStim)
nt = len(time)  

blocksize = 0.2
nblock = round(blocksize / dtStim)
stimrnge = [-10, 10]

stim1 = np.repeat(np.random.rand(nt // nblock), nblock) 
stim1 = stim1 * (stimrnge[1] - stimrnge[0]) + stimrnge[0]


alphas = [10, 100, 1000]

fig, axs = plt.subplots(3, 1, figsize=(6, 6))  

for idx, a in enumerate(alphas):
    params = {
        'N': 400,
        'wmean': np.array([0.1]), 
        'wsig': 0.01,
        'fmax': 100, 
        'alpha': a, 
        'fmin': 1, 
        'taud': 10,
        'A': 0, 
        'mu': 0,
        'tdel': 0
    }

    # Simulate using local_framework
    o, xh, xx = local_framework(stim1.reshape(1, -1), dtStim, params)
    
    axs[idx].plot(time, xh.flatten(), 'r-', label="Read-out")
    axs[idx].plot(time, xx.flatten(), 'b--', label="Target")
    axs[idx].set_title(f'Readout vs Target (alpha = {a})')
    axs[idx].set_xlabel('Time (s)')
    axs[idx].legend(frameon=False)

fig.tight_layout()

plt.show()
