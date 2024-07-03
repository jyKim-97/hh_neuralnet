import numpy as np
from numba import njit

@njit
def _count(xdigit, nmax):
    ndigit = np.zeros(nmax)
    for i in range(nmax):
        ndigit[i] = np.sum(xdigit == i)
    return ndigit

@njit
def compute_ent(prob):
    return -np.sum(prob * np.log2(prob + 1e-19))

@njit
def _norm(x):
    return x / (x.sum() + 1e-19)

@njit 
def compute_prev_hist(y, y_prv, x_prv, nbin):
    pyy = np.zeros((nbin, nbin))
    pyyx = np.zeros((nbin, nbin, nbin))
    
    for ny in range(nbin):
        id_y = y_prv == ny        
        pyy[:, ny] = _count(y[id_y], nbin)
        
        for nx in range(nbin):
            id_x = x_prv == nx
            pyyx[:, ny, nx] = _count(y[id_y & id_x], nbin)
            
    return pyy, pyyx


@njit
def compute_joint_hist(y, x, nbin):
    pyx = np.zeros((nbin, nbin))
    for nx in range(nbin):
        pyx[:, nx] = _count(y[x==nx], nbin)
    return pyx


@njit
def _compute_joint_h(y, y_prv, x_prv, nbin):
    pyy, pyyx = compute_prev_hist(y, y_prv, x_prv, nbin)
    return compute_ent(_norm(pyy)), compute_ent(_norm(pyyx))

@njit
def _compute_te(x, y, nbin, nlag_max):
    
    # compute prob for curent ~
    py = _count(y, nbin)
    hy = compute_ent(_norm(py))
    
    pyx = compute_joint_hist(y, x, nbin)
    hyx = compute_ent(_norm(pyx))
    
    # compute TE 
    te = np.zeros(nlag_max)
    for nl in range(1, nlag_max):
        hyy, hyyx = _compute_joint_h(y[nl:], y[:-nl], x[:-nl], nbin)
        te[nl] = hyy + hyx - hy - hyyx
        
    return te, hy


def compute_te(xd, yd, num_bin, nlag_max):
    # Compute transfer entropy from x to y: TE_{X->Y}
    # x and y need to be digitized
    
    # check input validity
    def _isint(x):
        # return "int" in x.dtype
        return isinstance(x[0], np.int64)
    
    if not _isint(xd) or not _isint(yd):
        raise ValueError("Check the input variable datatype")
    
    # compute probability density
    return _compute_te(xd, yd, num_bin, nlag_max)


# compute oscillatory motif 