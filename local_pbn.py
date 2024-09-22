import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt

def local_framework(stim, dt, params):
    """
    Inputs:
    =======
    stim (numpy.ndarray): Stimulus time series (D x T)
    dt (float): Time step size
    params (dict): Parameters

    Outputs:
    ========
    ss : Spike trains (N x T)
    xh : Network read-out (D x T)
    xx : True dynamic variable (D x T)
    """
    nd, nt = stim.shape
    N = params['N']
    tdel = params['tdel']
    
    alpha = params['alpha']
    fmax = params['fmax']
    fmin = params['fmin']
    
    lambda_d = 1 / params['taud']
    A = params['A']
    
    mu = params['mu']
    rlambda = mu * lambda_d ** 2
    
    
    # Generate decoding weights
    wmean = params['wmean']
    wsig = params['wsig']
    ww = np.tile(wmean, (N, 1)).T * np.concatenate(([-1] * (N // 2), [1] * (N - N // 2))) + np.random.randn(nd, N) * wsig
    
    T = (mu * lambda_d ** 2 + np.linalg.norm(ww.T, axis=1)**2) / 2
    
    ss = np.zeros((N, nt))
    rr = np.zeros((N, nt))
    xx = np.zeros((nd, nt))
    xh = np.zeros((nd, nt))
    zz = np.zeros((nd, nt))
    vv = np.zeros((N, nt))
    pspike = np.zeros((N, nt))
    pen = np.zeros((N, 1))

    # Handle the case when A is zero
    if np.all(A == 0):
        Amult = np.eye(nd)  # Identity matrix
        Amult_delay = np.eye(nd)  # Identity matrix
    else:
        # Exponential integrators
        Amult = np.eye(nd) + dt * A  # x integration (first-order approximation)
        Amult_delay = np.eye(nd) + tdel * dt * A  # x integration with delay (first-order approximation)
        
    Rmult = np.exp(-lambda_d * dt)
    Rmult_delay = np.exp(-lambda_d * dt * tdel)

    
    for tt in range(1, nt):
        xx[:, tt] = Amult @ xx[:, tt-1] + dt * stim[:, tt]
        
        rr[:, tt] = rr[:, tt-1] * Rmult + rr[:, tt]
        xh[:, tt] = ww @ rr[:, tt]
        
        zz[:, tt] = Amult @ zz[:, tt-1] + dt * stim[:, tt]
        
        
        vv[:, tt] = ww.T @ (Amult_delay @ zz[:, tt] - Rmult_delay * (ww @ rr[:, tt])) 
        
        
        rt = alpha * (vv[:, tt] - T)
        cond = ((fmax - fmin) / (1 + np.exp(-rt))) + fmin
        pspike[:, tt] = 1 - np.exp(-cond * dt)
        
        iisp = np.where(np.random.rand(N) < pspike[:, tt])[0]
        if len(iisp) > 0:
            if tt < nt - tdel:
                ss[iisp, tt + tdel] = 1
                rr[iisp, tt + tdel] = rr[iisp, tt + tdel] + 1
        xh[:, tt] = ww @ rr[:, tt]
        
    return ss, xh, xx

