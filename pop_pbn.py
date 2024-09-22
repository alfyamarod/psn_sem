import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def population_framework(stim, dt, params):
    """
    
    Parameters:
    ===========
    stim : np.array
        Stimulus 
    dt : float
        Time step size.
    params : dict
        Dictionary containing simulation parameters:
    Returns:
    ========
    ss : np.array
        Spike trains for all N neurons, shape (N, T).
    xh : np.array
        Network output (reconstruction of x), shape (D, T).
    xx : np.array
        True dynamic variable x, shape (D, T).
    """
    nd, nt = stim.shape  
    N = params['N']  
    tdel = params['tdel']  
    kappa = params['kappa']  

    # Dynamics parameters
    lambda_d = 1 / params['taud']  
    A = params['A'] 

    ww = np.outer(params['wmean'], np.ones(N // 2)) + np.random.randn(nd, N // 2) * params['wsig'] * np.mean(params['wmean'])  
    wpinv = np.linalg.pinv(ww)  

 
    ss = np.zeros((N, nt))  
    rr = np.zeros((N // 2, nt)) 
    vv = np.zeros((N // 2, nt))  
    zz = np.zeros((nd, nt))  
    xh = np.zeros((nd, nt)) 
    xx = np.zeros((nd, nt))  
    pspike = np.zeros((N // 2, nt))  

    
    Amult = expm(dt * A)   
    Amult_delay = expm(tdel * dt * A)  
    Rmult = np.exp(-lambda_d * dt)  
    Rmult_delay = np.exp(-lambda_d * dt * tdel) 
  
    for tt in range(1, nt):
        # Update true target variable xx
        xx[:, tt] = Amult @ xx[:, tt - 1] + dt * stim[:, tt]

        # Update filtered spike trains
        rr[:, tt] = rr[:, tt - 1] * Rmult + rr[:, tt] 
        xh[:, tt] = ww @ rr[:, tt]  
       
        zz[:, tt] = Amult @ zz[:, tt - 1] + dt * stim[:, tt]

       
        vv[:, tt] = wpinv @ (Amult_delay @ zz[:, tt] - Rmult_delay * (ww @ rr[:, tt]))

        # Compute spike probabilities
        pspike[:, tt] = (1 / kappa) * vv[:, tt]
        ppos = np.maximum(pspike[:, tt], 0)  # positive mirror neuron spikes
        pneg = -np.minimum(pspike[:, tt], 0)  # negative mirror neuron spikes

        sspos = (np.random.rand(N // 2) < ppos).astype(int)  # positive mirror neurons
        ssneg = (np.random.rand(N // 2) < pneg).astype(int)  # negative mirror neurons

        
        if tt < nt - tdel:  
            ss[:, tt + tdel] = np.concatenate((ssneg, sspos))  
            netsps = sspos - ssneg  
            rr[:, tt + tdel] += netsps  

        xh[:, tt] = ww @ rr[:, tt]

    return ss, xh, xx
