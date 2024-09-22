from pop_pbn import *

# Set up parameters
NT = 3
dtStim = 0.1 / 1000 
time = np.arange(dtStim, NT + dtStim, dtStim) 
nt = len(time) 

# Set up stimulus
blocksize = 0.2  # length of stimulus blocks in seconds
nblock = round(blocksize / dtStim)
stimrnge = [-25, 25]

stim1 = np.repeat(np.random.rand(nt // nblock), nblock) 
stim1 = stim1 * (stimrnge[1] - stimrnge[0]) + stimrnge[0]

stim2 = np.repeat(np.random.rand(nt // nblock), nblock)
stim2 = stim2 * (stimrnge[1] - stimrnge[0]) + stimrnge[0]

# Create the 2D stimulus
stim = np.vstack([stim1 - 0.5, -stim2 + 0.5])

A = np.linalg.inv(np.array([[-0.5, 1], [-1, -0.5]]))
params = {
        'N': 400,  # Number of neurons
        'wmean': 0.2 * np.ones((1, A.shape[0])),  
        'wsig': 0.01,  # variance in weight value
        'kappa': 50,  
        'taud': 0.2,  # rate decay
        'A': A,  # dynamics of x
        'mu': 0,  # quadratic cost on spiking
        'tdel': 0  # time delay, in bins
}

o, xh, xx = population_framework(stim, dtStim, params)


# Plot stimulus
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax1.plot(time, np.zeros_like(time), 'k--', linewidth=2)
ax1.plot(time, stim[0, :], label="Stimulus 1", linewidth=2)
ax1.plot(time, stim[1, :], label="Stimulus 2", linewidth=2)
ax1.set_title('Stimulus')
ax1.set_ylabel('stimulus')
ax1.set_xticks([])

# Plot raster plot
N = params['N']  # number of neurons
iiinh, jjinh = np.where(o[:N//2, :])
iiexc, jjexc = np.where(o[N//2:, :])
ax2 = plt.subplot2grid((2, 2), (0, 1))
ax2.plot(jjinh*dtStim, iiinh, '.', jjexc*dtStim, iiexc + N//2, '.')
ax2.set_ylim([0, N])
ax2.set_title('Raster plot')
ax2.set_ylabel('neuron')

# Target and read-out
ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)
ax3.plot(time, np.zeros_like(time), 'k--', linewidth=2)
ax3.plot(time, xh[0, :], 'r-', label="Read-out 1")
ax3.plot(time, xx[0, :], 'b--', label="Target 1")
ax3.plot(time, xh[1, :], 'r-.', label="Read-out 2")  # Different line style for clarity
ax3.plot(time, xx[1, :], 'b:', label="Target 2")
ax3.legend(frameon=False)
ax3.set_title('Target and Read-out')

plt.show()
