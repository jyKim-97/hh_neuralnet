import numpy as np
from numba import njit
from frites.core.gcmi_nd import cmi_nd_ggg
from typing import Annotated, Tuple, List
from numpy.linalg import LinAlgError


def compute_te(v_sample: np.ndarray,
               nmove: int=5,
               nmin_delay: int=1, nmax_delay: int=40, nstep_delay: int=1,
               nrel_points: Tuple|List=None,
               method="naive"):
    
    if nrel_points is None: nrel_points = [0]
    if np.any(np.array(nrel_points) > 0):
        raise ValueError("nrel_points must be negative")
    
    assert method in ("spo", "naive", "mit")
        
    if method == "spo":
        rollout_points = rollout_points_spo
    elif method == "naive":
        rollout_points = rollout_points_naive
    elif method == "mit":
        # rollout_points = rollout_points_te
        raise NotImplemented("momentary information transfer method have not yet implemted")
    
    # change data shape
    data = np.transpose(v_sample, (1, 2, 0))[..., np.newaxis, :]
    
    # target points to compue
    ndelays = -np.arange(nmin_delay, nmax_delay+1, nstep_delay)
    npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)

    pair = ((0, 1), (1, 0))
    te_pair = np.zeros((2, len(ndelays)))
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks] # source, (time, nvar, epochs)
        y = data[kd] # desination

        te, nte = 0, 0
        for n0 in npoint_pos:
            yt_curr, yt_prev, xt_prev = rollout_points(x, y, n0, ndelays, nrel_points)
            
            # compute TE with "num_points" steps history
            if yt_curr.shape[-1] < 5*len(nrel_points): continue
            try:
                te_sub = cmi_nd_ggg(yt_curr, xt_prev, yt_prev, mvaxis=-2, demeaned=False)
            except:
                continue
            
            te += te_sub
            nte += 1

        te_pair[ntp] = te / nte

    nlag = -ndelays - np.average(nrel_points)
        
    return te_pair, nlag


def compute_te_2d(v_sample: np.ndarray, 
                      nmove: int=5,
                      nmin_delay: int=1, nmax_delay: int=80, nstep_delay: int=1,
                      num_time_stack: int=50):
    """
    Compute TE without considering time components
    
    Inputs:
    v_sample (nsamples, sources, ntimes)
    
    Outputs:
    te_pair_2d (npairs, ndelays, ndelays): TE (src, dst)
    """
    assert not np.any(np.isnan(v_sample))
    data = np.transpose(v_sample, (1, 0, 2)) # (sources, nsamples, ntimes)
    
    ndelays = -np.arange(nmin_delay, nmax_delay+1, nstep_delay)
    num_delays = len(ndelays)
    
    npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)
    
    pair = ((0, 1), (1, 0))
    te_pair_2d = np.zeros((2, num_delays, num_delays))
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks] # (nsample, ntimes)
        y = data[kd]
        
        nstack = 0
        nte = np.zeros(num_delays)
        yt_curr, yt_prev, xt_prev = [], [], []
        for n0 in npoint_pos:
            # stack
            yt_curr.append(y[:, n0]) # (nsamples)
            yt_prev.append(y[:,n0+ndelays])
            xt_prev.append(x[:,n0+ndelays])
            nstack += 1
            
            if nstack == num_time_stack or n0 == npoint_pos[-1]:
                # reshape
                yt_curr = np.concatenate(yt_curr)
                yt_prev = np.vstack(yt_prev)
                xt_prev = np.vstack(xt_prev)
                
                yt_curr = np.tile(yt_curr[None,None,:], (num_delays, 1, 1))
                yt_prev = np.transpose(yt_prev)[:,np.newaxis,:]
                xt_prev = np.tile(np.transpose(xt_prev)[:,None,None,:], (1, num_delays, 1, 1))
                
                # compute TE
                for i in range(num_delays):
                    te = cmi_nd_ggg(yt_curr, xt_prev[i], yt_prev, mvaxis=1, demeaned=False)
                    te_pair_2d[ntp,i,:] += te
                    nte[i] += 1
                
                # reset
                nstack = 0
                yt_curr, yt_prev, xt_prev = [], [], []
            
        nte[nte==0] = 1
        te_pair_2d[ntp] /= nte[:, None]
        
    nlag = -ndelays
    
    return te_pair_2d, nlag


def compute_te_2d_reverse(v_sample: np.ndarray, 
                          nmove: int=5,
                          nmin_delay: int=1, nmax_delay: int=80, nstep_delay: int=1,
                          num_time_stack: int=50):
    """
    Compute TE without considering time components
    
    Inputs:
    v_sample (nsamples, sources, ntimes)
    
    Outputs:
    te_pair_2d (npairs, ndelays, ndelays): TE (src, dst)
    """
    assert not np.any(np.isnan(v_sample))
    data = np.transpose(v_sample, (1, 0, 2)) # (sources, nsamples, ntimes)
    
    ndelays = np.arange(nmin_delay, nmax_delay+1, nstep_delay)
    ndelays_pre = np.arange(nmin_delay-nstep_delay, nmax_delay, nstep_delay)
    num_delays = len(ndelays)
    
    npoint_pos = np.arange(0, v_sample.shape[-1]-nmax_delay, nmove)
    
    pair = ((0, 1), (1, 0))
    te_pair_2d = np.zeros((2, num_delays, num_delays)) # (pair, src, dst)
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks] # (nsample, ntimes)
        y = data[kd]
        
        nstack = 0
        nte = np.zeros(num_delays)
        yt_next, yt_curr, xt_curr = [], [], []
        for n0 in npoint_pos:
            # stack
            yt_next.append(y[:,n0+ndelays])
            yt_curr.append(y[:,n0+ndelays_pre])
            xt_curr.append(x[:,n0])
            nstack += 1
            
            if nstack == num_time_stack or n0 == npoint_pos[-1]:
                # reshape
                yt_next = np.vstack(yt_next)
                yt_curr = np.vstack(yt_curr)
                xt_curr = np.concatenate(xt_curr)
                
                yt_next = np.tile(np.transpose(yt_next)[:,None,None,:], (1, num_delays, 1, 1)) 
                yt_curr = np.transpose(yt_curr)[:,np.newaxis,:] # (ndelays, 1, nepochs)
                xt_curr = np.tile(xt_curr[None,None,:], (num_delays, 1, 1)) 
                
                # compute TE
                for i in range(num_delays):
                    te = cmi_nd_ggg(yt_next[i,:i+1], xt_curr[:i+1], yt_curr[:i+1], mvaxis=1, demeaned=False)
                    te_pair_2d[ntp,i,:i+1] += te
                    nte[i] += 1
                
                # reset
                nstack = 0
                yt_next, yt_curr, xt_curr = [], [], []
            
        nte[nte==0] = 1
        te_pair_2d[ntp] /= nte[:, None]
        
    nlag = ndelays
    
    return te_pair_2d, nlag    
            
        
# # TODO: oscillation peak 위에서 보내는 정보를 계산하려면 t-k, t를 고려하는 것보다 t, t+k고려가 낫지 않을까?
# def compute_te_2d(v_sample: np.ndarray,
#                   nmove: int=5,
#                   nmin_delay: int=1, nmax_delay: int=80, nstep_delay: int=1):
    
#     """
#     Inputs:
#     v_sample (nsamples, sources, ntimes)
    
#     Outputs:
#     te_pair_2d (npairs, ndelays, ndelays): TE (src, dst)
#     """
    
#     # change data shape
#     data = np.transpose(v_sample, (1, 2, 0))[..., np.newaxis, :]
    
#     # target points to compue
#     ndelays = -np.arange(nmin_delay, nmax_delay+1, nstep_delay)
#     npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)
#     num_delays = len(ndelays)

#     pair = ((0, 1), (1, 0))
#     te_pair_2d = np.zeros((2, num_delays, num_delays)) # \tau_{x-}, \tau_{y-}
#     for ntp, (ks, kd) in enumerate(pair):
#         x = data[ks] # source, (time, nvar, epochs)
#         y = data[kd] # desinatio
        
#         nte = np.zeros(num_delays)
#         for n0 in npoint_pos:
#             yt_curr = np.tile(y[n0,...], (num_delays, 1, 1))
#             yt_prev = y[n0+ndelays,...]
            
#             for i, n2 in enumerate(ndelays):
                
#                 xt_prev = np.tile(x[n0+n2,...], (num_delays, 1, 1))
#                 _x, _y, _z = clean_null_points(yt_curr, xt_prev, yt_prev)
                
#                 if _x.shape[-1] < 5: continue
#                 try:
#                     te = cmi_nd_ggg(_x, _y, _z, mvaxis=-2, demeaned=False)
#                 except:
#                     continue
                
#                 te_pair_2d[ntp,i,:] += te
#                 nte[i] += 1
        
#         nte[nte == 0] = 1
#         te_pair_2d[ntp] /= nte[:, None]

#     nlag = -ndelays
        
#     return te_pair_2d, nlag


def _cmi_nd_ggg(x, y, z, b=0):
    try:
        return cmi_nd_ggg(x, y, z, mvaxis=-2, demeaned=False)
    except LinAlgError as e:
        if b != 1e-9:
            b = 1e-9
            x -= x.mean(axis=-1, keepdims=True) + b
            y -= y.mean(axis=-1, keepdims=True) + b
            z -= z.mean(axis=-1, keepdims=True) + b
            return _cmi_nd_ggg(x, y, z, b=b)
        else:
            return np.zeros(len(x))*np.nan


def compute_te_full(v_sample: np.ndarray,
                    nmove: int=5,
                    nmin_delay: int=1, nmax_delay: int=40, nstep_delay: int=1, verbose=False):
    
    # change data shape
    data = np.transpose(v_sample, (1, 2, 0))[..., np.newaxis, :]
    
    # target points to compue
    ndelays = -np.arange(nmin_delay, nmax_delay+1, nstep_delay)
    npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)
    num_delays = len(ndelays)

    pair = ((0, 1), (1, 0))
    te_pair_full = np.zeros((2, num_delays))
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks] # source, (time, nvar, epochs)
        y = data[kd] # desinatio
        
        nte = np.zeros(num_delays)
        for n0 in npoint_pos:
            for i, nd in enumerate(ndelays):
                yt_curr = y[n0,...][np.newaxis,...]
                
                # d = -np.arange(1, -nd+1)[::-1]
                # nd_set = -np.arange(1, -nd+1)
                # xt_prev = np.hstack([x[[n0+d],...] for d in nd_set])
                # yt_prev = np.hstack([y[[n0+d],...] for d in nd_set])
                # _x, _y, _z = clean_null_points(yt_curr, xt_prev, yt_prev)
                
                # try:
                #     te = cmi_nd_ggg(_x, _y, _z, mvaxis=-2, demeaned=False)
                # except:
                #     continue
                
                # xt_prev = np.hstack([x[[n0+d],...] for d in nd_set])
                
                # if abs(te) > 1e5 or np.isnan(te):
                #     continue
                # nte[i] += 1
                
                xt_prev = x[n0+nd,...][np.newaxis,...]
                yt_prev = np.hstack([y[[n0+d],...] for d in ndelays])
                try:
                    te = cmi_nd_ggg(yt_curr, xt_prev, yt_prev, mvaxis=-2, demeaned=False)
                except LinAlgError as e:
                    pass
                
                if np.any(abs(te) > 1e5) or np.any(np.isnan(te)):
                    continue

                te_pair_full[ntp] += te
                nte += 1
                
        nte[nte == 0] = 1
        te_pair_full[ntp] /= nte

    nlag = -ndelays
        
    return te_pair_full, nlag


def compute_te_full2(v_sample: np.ndarray,
                    nmove: int=5,
                    nmin_delay: int=1, nmax_delay: int=80, nstep_delay: int=1):
    
    """
    Compute transfer entropy (TE) considering full history
    TE_{X->Y}(k) = I(Y(t), X(t-k) | {Y_{t-k_max},...,Y_{t-1}}\\Y_{t-k}})
    
    Parameters
    -----------
    v_sample : np.ndarray
        A 3D array of shape (n, m, t) where 'n' is the number of samples, 
        'm' is the number of nodes (2) and 't' is the number of time steps
    nmove : int
        The number of steps the window move
    nmin_delay : int (default 1)
        The minimal time step for lag
    nmax_delay : int (default 80)
        The maximal time step for lag
    nstep_delay : int (default 1)
    
    Returns
    -------
    te_pair_2d : np.ndarray
        A 2D array of shape (m, k) where 'k' is equal to (nmax_delay-nmin_delay)//2
    """
    
    # change data shape
    data = np.transpose(v_sample, (1, 2, 0))[..., np.newaxis, :]
    
    # target points to compue
    ndelays = -np.arange(nmin_delay, nmax_delay+1, nstep_delay)
    npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)
    num_delays = len(ndelays)

    pair = ((0, 1), (1, 0))
    te_pair_full = np.zeros((2, num_delays))
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks] # source, (time, nvar, epochs)
        y = data[kd] # desinatio
        
        nte = np.zeros(num_delays)
        for n0 in npoint_pos:
            # yt_curr = np.tile(y[n0,...], (len(ndelays), 1, 1))
            # xt_prev = y[n0+ndelays,...]
            # yt_prev = np.concatenate([
            #     np.hstack([y[[n0+d],...] for d in ndelays if d != nd]) for nd in ndelays
            # ])
            
            # yt_prev = np.hstack([
            #     [y[[n0+d],...] for d in ndelays if d != nd] for nd in ndelays
            # ])
            
            # print(yt_curr.shape, xt_prev.shape, yt_prev.shape)
            
            # te = _cmi_nd_ggg(yt_curr, xt_prev, yt_prev)
            
            # if np.any(abs(te) > 1e5) or np.any(np.isnan(te)):
            #         continue

            # te_pair_full[ntp] += te
            # nte += 1
            
            
            for i, nd in enumerate(ndelays):
                yt_curr = y[n0,...][np.newaxis,...]
                xt_prev = x[n0+nd,...][np.newaxis,...]
                yt_prev = np.hstack([y[[n0+d],...] for d in ndelays if d != nd])
                
                te = _cmi_nd_ggg(yt_curr, xt_prev, yt_prev)                
                if abs(te) > 1e5 or np.isnan(te):
                    continue
                
                te_pair_full[ntp, i] += te
                nte[i] += 1
                
        nte[nte == 0] = 1
        te_pair_full[ntp] /= nte

    nlag = -ndelays
        
    return te_pair_full, nlag


# def _cmi_nd_ggg_full(x, y, n0, ndelays):
#     yt_curr = np.tile(y[n0,...], (len(ndelays), 1, 1))
#     xt_prev = y[n0+ndelays,...]
#     yt_prev_tot = y[n0+ndelays,...]
    

    


def clean_null_points(x, y, z):
    # is_in = x[-1, 0, :] != 0
    is_in = ~np.isnan(x[-1, 0, :])
    return x[:,:,is_in], y[:,:,is_in], z[:,:,is_in]


def rollout_points_naive(x, y, n0, ndelays, nrel_points):
    yt_curr = np.tile(y[n0,...], (len(ndelays), 1, 1))
    yt_prev = roll_hstack(y, n0, ndelays, nrel_points)
    xt_prev = roll_hstack(x, n0, ndelays, nrel_points)

    return clean_null_points(yt_curr, yt_prev, xt_prev)


def rollout_points_spo(x, y, n0, ndelays, nrel_points):
    yt_curr = np.tile(y[n0,...], (len(ndelays), 1, 1))
    # yt_prev = roll_hstack(y, n0, np.array([-1]), [0])
    yt_prev = roll_hstack(y, n0, np.array([-1]), nrel_points)
    yt_prev = np.tile(yt_prev, (len(ndelays), 1, 1))
    xt_prev = roll_hstack(x, n0, ndelays, nrel_points)    
    
    return clean_null_points(yt_curr, yt_prev, xt_prev)


# def rollout_points_full2(x, y, n0, ndelays, nrel_points):
#     yt_curr = y[n0,...][np.newaxis,...]
#     xt_prev = x[n0+nd,...][np.newaxis,...]
#     yt_prev = np.hstack([y[[n0+d],...] for d in ndelays if d != nd])


def roll_hstack(data, ncurr, ndelays:int|np.ndarray, nrel_points):
    return np.hstack([data[ncurr+ndelays+nd,...] for nd in nrel_points])


# === surrogate test ==== #
# def sample_true(v_set: np.ndarray,
#                 nsamples: int,
#                 nadd: int):
    
#     idx = np.random.randint(0, len(v_set), nsamples)
#     return v_set[idx, :, nadd:]


def _sampling(f_sampling,
              v_set: np.ndarray,
              nchunks: int=1000,
              chunk_size: int=100,
              max_delay: int=0,
              nadd: int=0,
              reverse=False):
    """
    Sampling by chunking the signal.
    The first 'max_delay' number of signals will be oeverlapped
    """
    
    assert max_delay < nadd
    
    nlen = chunk_size + max_delay
    v_sample = np.zeros((nchunks, 2, nlen))
    
    n = 0
    refresh = True
    while n < nchunks:
        
        if refresh: # reload new dataset
            v_sel = f_sampling(v_set)

            nmax = v_sel.shape[1]
            n0 = np.random.randint(nlen)
            n1 = n0 + nlen
            refresh = False
        
        # print(n0, n1, nmax, v_sel.shape, n1-n0)
        if n1 <= nmax:
            v_sample[n] = v_sel[:,n0:n1]
            n0 = n1 - max_delay
            n1 = n0 + nlen
            
            # try:
            #     v_sample[n] = v_sel[:,n0:n1]
            # except:
            #     print(n0, n1, n1-n0, nmax, nlen, max_delay, chunk_size)
            #     raise ValueError("")
                
            # if not reverse:
            #     n0 = n1 - max_delay
            #     n1 = n0 + nlen
            # else:
            #     n0 = n1
            #     n1 = n0 + nlen
        else:
            n -= 1            
            refresh = True
        
        n += 1

    return v_sample


def sample_true(v_set: np.ndarray,
                nchunks: int=1000,
                chunk_size: int=100,
                nmax_delay: int=0,
                nadd: int=0, reverse: bool=False):
    
    def f_sampling(v_set):
        idx = np.random.randint(0, v_set.shape[0])
        if not reverse:
            v_sel = v_set[idx, :, nadd-nmax_delay:]
        else:
            v_sel = v_set[idx,:,:-(nadd-nmax_delay)]
        is_in = ~np.isnan(v_sel[0, :])
        v_sel = v_sel[:, is_in]
        return v_sel

    return _sampling(f_sampling, v_set, nchunks, chunk_size, nmax_delay, nadd, reverse=reverse)


def sample_surrogate(v_set: np.ndarray,
                     nchunks: int=1000,
                     chunk_size: int=100,
                     nmax_delay: int=0,
                     nadd: int=0,
                     warp_range=(0.8, 1.2),
                     reverse=False):
    
    dr = 0.05
    ratio_set = np.arange(warp_range[0],
                          warp_range[1]+dr//2, dr)
    
    def f_sampling(v_set):
        nt, ns = np.random.choice(v_set.shape[0], 2, replace=True)
        if nt == ns:
            if not reverse:
                vsub = v_set[nt,:,nadd-nmax_delay:]
            else:
                vsub = v_set[nt,:,:-(nadd-nmax_delay)]
            vsub = vsub[:, ~np.isnan(vsub[0])]
            return vsub
        
        idt = ~np.isnan(v_set[nt,0,:])
        ids = ~np.isnan(v_set[ns,0,:])
        
        if np.sum(idt) > np.sum(ids)*warp_range[1]-dr:
            nt, ns = ns, nt
            idt, ids = ids, idt
        
        v_sel = v_set[nt][:,idt]
        n0 = nadd-nmax_delay
        if not reverse:
            vf = v_sel[0,n0:]
            vs = v_sel[1,n0:]
        else:
            vf, vs = v_sel[0,:-n0], v_sel[1,:-n0]
        
        vs_surr = warp_surrogate_set(vs, v_set[ns,1,ids], ratio_set)
        assert not np.any(np.isnan(vf))
        assert not np.any(np.isnan(vs_surr))
        # vs_surr = vs
        
        return np.array([vf, vs_surr])

    return _sampling(f_sampling, v_set, nchunks, chunk_size, nmax_delay, nadd, reverse=reverse)


def sample_surrogate_iaaft(v_set: np.ndarray,
                           nchunks: int=1000,
                           chunk_size: int=100,
                           nmax_delay: int=0,
                           nadd: int=0,
                           reverse=False):
    
    def f_sampling(v_set):
        
        idx = np.random.randint(0, v_set.shape[0])
        if not reverse:
            v_sel = v_set[idx, :, nadd-nmax_delay:]
        else:
            v_sel = v_set[idx,:,:-(nadd-nmax_delay)]
        is_in = ~np.isnan(v_sel[0, :])
        v_sel = v_sel[:, is_in]
        
        vs_surr = bivariate_surrogates(v_sel[0], v_sel[1])
        return np.array(vs_surr)

    return _sampling(f_sampling, v_set, nchunks, chunk_size, nmax_delay, nadd, reverse=reverse)


def bivariate_surrogates(x1, x2, tol_pc=5., maxiter=1e3):
    """
    Returns bivariate IAAFT surrogates of given two time series.

    Parameters
    ----------
    x1, x2 : numpy.ndarray, with shape (N,)
        Input time series for which bivariate IAAFT surrogates are to be estimated.
    ns : int
        Number of surrogates to be generated.
    tol_pc : float
        Tolerance (in percent) level which decides the extent to which the
        difference in the power spectrum of the surrogates to the original
        power spectrum is allowed (default = 5).
    maxiter : int
        Maximum number of iterations before which the algorithm should
        converge. If the algorithm does not converge until this iteration
        number is reached, the while loop breaks.

    Returns
    -------
    xs1, xs2 : numpy.ndarray, with shape (ns, N)
        Arrays containing the bivariate IAAFT surrogates of `x1` and `x2` such that
        each row of `xs1` and `xs2` are individual surrogate time series.

    """
    nx = x1.shape[0]
    ii = np.arange(nx)

    # Get the FFT of the original arrays
    x1_amp = np.abs(np.fft.fft(x1))
    x2_amp = np.abs(np.fft.fft(x2))

    x1_srt = np.sort(x1)
    x2_srt = np.sort(x2)

    r_orig1 = np.argsort(x1)
    r_orig2 = np.argsort(x2)
    
    # 1) Generate random shuffle of the data
    count = 0
    r_prev1 = np.random.permutation(ii)
    r_prev2 = np.random.permutation(ii)
    r_curr1 = r_orig1
    r_curr2 = r_orig2
    z_n1 = x1[r_prev1]
    z_n2 = x2[r_prev2]
    percent_unequal = 100.
    
    phi1 = np.angle(np.fft.fft(x1))
    phi2 = np.angle(np.fft.fft(x2))   

    # Core iterative loop
    while (percent_unequal > tol_pc) and (count < maxiter):
        r_prev1 = r_curr1
        r_prev2 = r_curr2

        # 2) FFT current iteration yk, and then invert it but while
        # replacing the amplitudes with the original amplitudes but
        # keeping the angles from the FFT-ed version of the random
        y_prev1 = z_n1
        # y_prev2 = z_n2

        # Apply FFT to both series
        fft_prev1 = np.fft.fft(y_prev1)
        # fft_prev2 = np.fft.fft(y_prev2)

        # Get phase angles for both series
        phi_prev1 = np.angle(fft_prev1)
        # phi_prev2 = np.angle(fft_prev2)

        # Maintain cross-correlation by using the same phase relationship
        mean_phase_diff = phi2 - phi1
        
        # Adjust the phase of the surrogate with consistent phase difference
        e_i_phi1 = np.exp(phi_prev1 * 1j)
        e_i_phi2 = np.exp((phi_prev1 + mean_phase_diff) * 1j)

        z_n1 = np.fft.ifft(x1_amp * e_i_phi1)
        z_n2 = np.fft.ifft(x2_amp * e_i_phi2)

        # 3) Rescale zk to the original distribution of x1 and x2
        r_curr1 = np.argsort(z_n1, kind="quicksort")
        r_curr2 = np.argsort(z_n2, kind="quicksort")

        z_n1[r_curr1] = x1_srt.copy()
        z_n2[r_curr2] = x2_srt.copy()

        # percent_unequal = ((r_curr1 != r_prev1).sum() + (r_curr2 != r_prev2).sum()) * 50. / nx
        percent_unequal = (r_curr2 != r_prev2).sum() * 100. / nx

        # 4) Repeat until number of unequal entries between r_curr and 
        # r_prev is less than tol_pc percent
        count += 1

    if count >= (maxiter - 1):
        print("Maximum number of iterations reached!")

    xs1 = np.real(z_n1)
    xs2 = np.real(z_n2)

    return xs1, xs2


# def sample_true(v_set: np.ndarray,
#                 nchunks: int=1000,
#                 chunk_size: int=100,
#                 max_delay: int=0,
#                 nadd: int=0):
    
#     assert max_delay < nadd
    
#     N = len(v_set)
#     v_true = np.zeros((nchunks, 2, chunk_size+max_delay))
    
#     n = 0
#     refresh = True
#     while n < nchunks:
        
#         if refresh:
#             idx = np.random.randint(0, N)
#             v_sel = v_set[idx, :, nadd-max_delay:]
#             is_in = v_sel[0, :] != 0
#             v_sel = v_sel[:, is_in]

#             nmax = v_sel.shape[1]
#             n0, n1 = 0, chunk_size+max_delay
#             refresh = False

#         if n1 <= nmax:
#             v_true[n] = v_sel[:,n0:n1]
#             n0 = n1 - max_delay
#             n1 = n0 + chunk_size + max_delay
#         elif n0 < nmax:
#             v_true[n,:,:(nmax-n0)] = v_sel[:,n0:]
#             refresh = True
#         else:
#             refresh = True
        
#         n += 1

#     return v_true


    # idx = np.random.randint(0, len(v_set), nsamples)
    # return v_set[idx, :, nadd:]



# def sample_surrogate(v_set: np.ndarray,
#                      nsamples: int,
#                      nadd: int=0,
#                      warp_range=(0.8, 1.2)):
    
#     dr = 0.05
#     ratio_set = np.arange(warp_range[0],
#                           warp_range[1]+dr//2, dr)
    
#     N = len(v_set)
#     v_set_surr = np.zeros((nsamples, 2, v_set.shape[-1]-nadd))
#     for n in range(nsamples):
#         nt, ns = np.random.choice(N, 2, replace=True)
#         if nt == ns:
#             v_set_surr[n] = v_set[nt,:,nadd:]
#             continue
        
#         idt = v_set[nt,0,:] != 0
#         ids = v_set[ns,0,:] != 0
        
#         if np.sum(idt) > np.sum(ids)*warp_range[1]-dr:
#             nt, ns = ns, nt
#             idt, ids = ids, idt
        
#         # id1 = v_set[n1,0,:] != 0
#         # id2 = v_set[n2,0,:] != 0
        
#         # l1 = np.sum(id1)
#         # l2 = np.sum(id2)
        
#         # if l1 > l2:
#         #     nt, ns = n2, n1
#         #     idt, ids = id2, id1
#         # else:
#         #     nt, ns = n1, n2
#         #     idt, ids = id1, id2
        
#         vf = v_set[nt,0,nadd:]
#         vs = v_set[nt,1,nadd:]
        
#         vs_surr = warp_surrogate_set(vs[idt[nadd:]],
#                                      v_set[ns,1,ids], ratio_set)
        
#         v_set_surr[n,0,:] = vf
#         v_set_surr[n,1,:len(vs_surr)] = vs_surr
    
#     return v_set_surr


@njit
def warp_surrogate_set(v_template, v_surr, ratio_set):
    
    num = len(ratio_set)
    cmax_set = np.zeros(num)
    vw_set = np.zeros((num, len(v_template)))
    
    for n in range(num):
        
        if len(v_surr) * ratio_set[n] < len(v_template):
            continue
        
        cmax, vw_align = warp_surrogate(v_template,  v_surr, ratio_set[n])
        
        cmax_set[n] = cmax
        vw_set[n] = vw_align
        
    nid = np.argmax(cmax_set)
    return vw_set[nid]


@njit
def warp_surrogate(v_template, v_surr, ratio):
    N = len(v_surr)
    nmax = int(N * ratio)
    
    # stretch signal
    vw = np.interp(np.arange(0, nmax+1e-10),
                np.linspace(0, nmax, N),
                v_surr)
    
    # get cross-correlation
    vw = (vw - vw.mean())/vw.std()
    c = np.correlate(vw, v_template)/len(v_template)
    
    # get optimal point
    nc = np.argmax(c)
    cmax = c[nc]
    vw_align = vw[nc:nc+len(v_template)]
    
    return cmax, vw_align


def reduce_te_2d(te_data_2d, tcut=None):
    from copy import deepcopy
    if tcut is None:
        tcut = te_data_2d["tlag"][-1]
    assert tcut > te_data_2d["tlag"][0]
    
    tlag = te_data_2d["tlag"]
    te_data = deepcopy(te_data_2d)
    
    N = int((tcut - tlag[0])/(tlag[1]-tlag[0]))+1
    N = min([N, len(tlag)])
    
    te_data["tlag"] = te_data["tlag"][:N]
    te_data["te"] = np.zeros((te_data["info"]["ntrue"], 2, N))
    te_data["te_surr"] = np.zeros((te_data["info"]["nsurr"], 2, N))

    for ntp in range(2):
        te_data["te"][:, ntp] = te_data_2d["te"][:,ntp,:N,:N].mean(axis=2) # source, receiver
        te_data["te_surr"][:, ntp] = te_data_2d["te_surr"][:,ntp,:N,:N].mean(axis=2)
        
    if "info" in te_data.keys():
        te_data["info"]["nmax_delay"] = N

    return te_data