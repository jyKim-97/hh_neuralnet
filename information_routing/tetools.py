import numpy as np
from numba import njit
from frites.core.gcmi_nd import cmi_nd_ggg
from typing import Annotated, Tuple, List


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
                  nmin_delay: int=1, nmax_delay: int=40, nstep_delay: int=1):
    
    # change data shape
    data = np.transpose(v_sample, (1, 2, 0))[..., np.newaxis, :]
    
    # target points to compue
    ndelays = -np.arange(nmin_delay, nmax_delay+1, nstep_delay)
    npoint_pos = np.arange(nmax_delay, v_sample.shape[-1], nmove)
    num_delays = len(ndelays)

    pair = ((0, 1), (1, 0))
    te_pair_2d = np.zeros((2, num_delays, num_delays)) # \tau_{x-}, \tau_{y-}
    for ntp, (ks, kd) in enumerate(pair):
        x = data[ks] # source, (time, nvar, epochs)
        y = data[kd] # desinatio
        
        nte = np.zeros(num_delays)
        for n0 in npoint_pos:
            yt_curr = np.tile(y[n0,...], (num_delays, 1, 1))
            yt_prev = y[n0+ndelays,...]
            
            for i, n2 in enumerate(ndelays):
                
                xt_prev = np.tile(x[n0+n2,...], (num_delays, 1, 1))
                _x, _y, _z = clean_null_points(yt_curr, xt_prev, yt_prev)
                
                if _x.shape[-1] < 5: continue
                try:
                    te = cmi_nd_ggg(_x, _y, _z, mvaxis=-2, demeaned=False)
                except:
                    continue
                
                te_pair_2d[ntp,i,:] += te
                nte[i] += 1
        
        nte[nte == 0] = 1
        te_pair_2d[ntp] /= nte[:, None]

    nlag = -ndelays
        
    return te_pair_2d, nlag


def compute_te_full(v_sample: np.ndarray,
                    nmove: int=5,
                    nmin_delay: int=1, nmax_delay: int=40, nstep_delay: int=1):
    
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
                nd_set = -np.arange(1, -nd+1)
                xt_prev = np.hstack([x[[n0+d],...] for d in nd_set])
                yt_prev = np.hstack([y[[n0+d],...] for d in nd_set])
                _x, _y, _z = clean_null_points(yt_curr, xt_prev, yt_prev)
                
                try:
                    te = cmi_nd_ggg(_x, _y, _z, mvaxis=-2, demeaned=False)
                except:
                    continue
                
                if abs(te) > 1e5 or np.isnan(te):
                    continue
                
                te_pair_full[ntp, i] += te
                nte[i] += 1
                
        nte[nte == 0] = 1
        te_pair_full[ntp] /= nte

    nlag = -ndelays
        
    return te_pair_full, nlag


def clean_null_points(x, y, z):
    is_in = x[-1, 0, :] != 0
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
              nadd: int=0):
    """
    Sampling by chunking the signal.
    The first 'max_delay' number of signals will be oeverlapped
    """
    
    assert max_delay < nadd
    
    N = len(v_set)
    v_sample = np.zeros((nchunks, 2, chunk_size+max_delay))
    
    n = 0
    refresh = True
    while n < nchunks:
        
        if refresh:
            v_sel = f_sampling(v_set)

            nmax = v_sel.shape[1]
            n0, n1 = 0, chunk_size+max_delay
            refresh = False

        if n1 <= nmax:
            v_sample[n] = v_sel[:,n0:n1]
            n0 = n1 - max_delay
            n1 = n0 + chunk_size + max_delay
        elif n0 < nmax:
            v_sample[n,:,:(nmax-n0)] = v_sel[:,n0:]
            refresh = True
        else:
            refresh = True
        
        n += 1

    return v_sample


def sample_true(v_set: np.ndarray,
                nchunks: int=1000,
                chunk_size: int=100,
                nmax_delay: int=0,
                nadd: int=0):
    
    def f_sampling(v_set):
        idx = np.random.randint(0, v_set.shape[0])
        v_sel = v_set[idx, :, nadd-nmax_delay:]
        is_in = v_sel[0, :] != 0
        v_sel = v_sel[:, is_in]
        return v_sel

    return _sampling(f_sampling, v_set, nchunks, chunk_size, nmax_delay, nadd)


def sample_surrogate(v_set: np.ndarray,
                     nchunks: int=1000,
                     chunk_size: int=100,
                     nmax_delay: int=0,
                     nadd: int=0,
                     warp_range=(0.8, 1.2)):
    
    dr = 0.05
    ratio_set = np.arange(warp_range[0],
                          warp_range[1]+dr//2, dr)
    
    def f_sampling(v_set):
        nt, ns = np.random.choice(v_set.shape[0], 2, replace=True)
        if nt == ns:
            return v_set[nt,:,nadd-nmax_delay:]
        
        idt = v_set[nt,0,:] != 0
        ids = v_set[ns,0,:] != 0
        
        if np.sum(idt) > np.sum(ids)*warp_range[1]-dr:
            nt, ns = ns, nt
            idt, ids = ids, idt
        
        n0 = nadd-nmax_delay
        v_sel = v_set[nt][:,idt]
        vf = v_sel[0,n0:]
        vs = v_sel[1,n0:]
        
        vs_surr = warp_surrogate_set(vs, v_set[ns,1,ids], ratio_set)
        
        return np.array([vf, vs_surr])

    return _sampling(f_sampling, v_set, nchunks, chunk_size, nmax_delay, nadd)


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