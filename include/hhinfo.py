# uncompyle6 version 3.9.1
# Python bytecode version base 3.7.0 (3394)
# Decompiled from: Python 3.7.7 (default, Mar 13 2020, 21:39:43) 
# [GCC 9.2.1 20190827 (Red Hat 9.2.1-1)]
# Embedded file name: /home/jungyoung/Project/hh_neuralnet/include/hhinfo.py
# Compiled at: 2024-05-19 17:08:05
# Size of source mod 2**32: 9732 bytes
import numpy as np
from numba import njit
import ctypes

@njit
def count(xdigit, nmax):
    ndigit = np.zeros(nmax)
    for i in range(nmax):
        ndigit[i] = np.sum(xdigit == i)

    return ndigit


@njit
def compute_ent(prob):
    return -np.sum(prob * np.log2(prob + 1e-19))


@njit
def norm(x):
    return x / (x.sum() + 1e-19)


def concat_signal(sig_set):
    N = len(sig_set)
    x, y, class_id = [], [], []
    for n in range(N):
        x.extend(sig_set[n][0])
        y.extend(sig_set[n][1])
        class_id.extend(np.ones_like(sig_set[n][0]) * n)

    return (np.array(x), np.array(y), np.array(class_id).astype(int))


_clib = ctypes.cdll.LoadLibrary("/home/jungyoung/Project/hh_neuralnet/include/estimate_prob.so")
_estimate_prob_c = _clib.estimate_full_hist
_estimate_prob_c.argtypes = (
 ctypes.c_int,
 ctypes.POINTER(ctypes.c_int),
 ctypes.POINTER(ctypes.c_int),
 ctypes.POINTER(ctypes.c_int),
 ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_estimate_prob_c.rettypes = ctypes.c_int

def estimate_hist_prob_concated_clib(x, y, class_id, nbin=30, nlag_max=20):
    """
    estimate the joint probaility P(Y, Y-, X-) and P(X, X-, Y-)
    
    """
    N = len(x)
    c_x = (ctypes.c_int * N)(*x)
    c_y = (ctypes.c_int * N)(*y)
    c_cid = (ctypes.c_int * N)(*class_id)
    nlen = int(nbin ** 3 * nlag_max * 2)
    tmp = np.zeros(nlen, dtype=int)
    c_out = (ctypes.c_int * nlen)(*tmp)
    ret = _estimate_prob_c(N, c_x, c_y, c_cid, nbin, nlag_max, c_out)
    out = np.array(c_out).reshape(nbin, nbin, nbin, nlag_max, 2)
    return out

_estimate_prob_2d_c = _clib.estimate_full_hist_2d
_estimate_prob_2d_c.argtypes = (
 ctypes.c_int,
 ctypes.POINTER(ctypes.c_int),
 ctypes.POINTER(ctypes.c_int),
 ctypes.POINTER(ctypes.c_int),
 ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
_estimate_prob_2d_c.rettypes = ctypes.c_int

def estimate_hist_prob_2d_concated_clib(x, y, class_id, nbin=30, nlag_max=20):
    """
    estimate the joint probaility P(Y, Y-, X-) and P(X, X-, Y-)
    
    """
    N = len(x)
    c_x = (ctypes.c_int * N)(*x)
    c_y = (ctypes.c_int * N)(*y)
    c_cid = (ctypes.c_int * N)(*class_id)
    nlen = int(nbin ** 3 * nlag_max * 2)
    tmp = np.zeros(nlen, dtype=int)
    c_out = (ctypes.c_int * nlen)(*tmp)
    ret = _estimate_prob_2d_c(N, c_x, c_y, c_cid, nbin, nlag_max, c_out)
    out = np.array(c_out).reshape(nbin, nbin, nbin, nlag_max, nlag_max, 2)
    return out


def estimate_hist_prob_concated(x, y, class_id, nbin=30, nlag_max=20):
    """
    Estimate the joint probability P(Y, Y-, X-) and P(X, X-, Y-) from concatentated signals to compute transfer entropy
    
    args:
        X (np.ndarray): concatenated signal (especially, fast subpopulation activity)
        Y (np.ndarray): concatenated signal (~, slow)
        class_id (np.ndarray with int)
    
    Returns:
        np.ndarray (nbin, nbin, nbin, nlag_max, 2): Estimated joint probability
            The last index indicates P(Y, Y-, X-) (0) and P(X, X-, Y-) (1)
    """
    nbuffer = int(200000.0)
    N = len(class_id)
    if nbuffer > N:
        return _estimate_hist_prob_concated(x, y, class_id, nbin, nlag_max)
    prob3 = 0
    n_prv = 0
    n_use = 0
    for _ in range(len(x) // nbuffer + 1):
        if n_use == N:
            break
        n_use += nbuffer
        n_use = min(n_use, N)
        _id = class_id[n_use - 1]
        while _id == class_id[n_use - 1]:
            n_use += 1
            if n_use >= N:
                break

        prob3 += _estimate_hist_prob_concated(x[n_prv[:n_use]], y[n_prv[:n_use]], class_id[n_prv[:n_use]], nbin, nlag_max)
        n_prv = n_use

    return prob3


@njit
def _estimate_hist_prob_concated(x, y, class_id, nbin=30, nlag_max=20):
    type_id = np.zeros(len(class_id))
    ntype = nlag_max
    for n in range(len(class_id) - 1, -1, -1):
        if ntype == 0:
            if class_id[n + 1] != class_id[n]:
                ntype = nlag_max
        type_id[n] = ntype
        if ntype > 0:
            ntype -= 1

    pyyx = np.zeros((nbin, nbin, nbin, nlag_max, 2))
    is_type = type_id[:-nlag_max] == 0
    for nl in range(1, nlag_max + 1):
        if nl == nlag_max:
            x_cur, y_cur = x[nl][is_type], y[nl][is_type]
        else:
            x_cur = x[nl[:-nlag_max + nl]][is_type]
            y_cur = y[nl[:-nlag_max + nl]][is_type]
        x_prv = x[:-nlag_max][is_type]
        y_prv = y[:-nlag_max][is_type]
        for ny in range(nbin):
            id_y = y_prv == nye
            for nx in range(nbin):
                id_x = x_prv == nx
                cond = id_x & id_y
                pyyx[:, ny, nx, nl-1, 0] = count(y_cur[cond], nbin)
                pyyx[:, nx, ny, nl-1, 1] = count(x_cur[cond], nbin)

    return pyyx


@njit
def estimate_full_hist_prob(x, y, nbin=30, nlag_max=20):
    pyyx = np.zeros((nbin, nbin, nbin, nlag_max, 2))
    for ny in range(nbin):
        id_y = y[:-nlag_max] == ny
        for nx in range(nbin):
            id_x = x[:-nlag_max] == nx
            cond = id_x & id_y
            for nl in range(1, nlag_max + 1):
                if nl == nlag_max:
                    _x, _y = x[nl:], y[nl:]
                else:
                    _x, _y = x[nl[:-nlag_max + nl]], y[nl[:-nlag_max + nl]]
                pyyx[:, ny, nx, nl-1, 0] = count(_y[cond], nbin)
                pyyx[:, nx, ny, nl-1, 1] = count(_x[cond], nbin)

    return pyyx


@njit
def compute_hist_prob(y, y_prv, x_prv, nbin):
    pyyx = np.zeros((nbin, nbin, nbin))
    for ny in range(nbin):
        id_y = y_prv == ny
        for nx in range(nbin):
            id_x = x_prv == nx
            pyyx[:, ny, nx] = count(y[id_y & id_x], nbin)

    return pyyx


@njit
def compute_prob(y, nbin):
    return count(y, nbin)


@njit
def compute_joint_prob(x, y, nbin):
    pass


def _compute_te(xd, yd, nbin, nlag_max):
    cid = np.zeros_like(xd)
    pyyx = estimate_hist_prob_concated_clib(xd, yd, cid, nbin, nlag_max)
    pyyx = norm_prob3(pyyx)
    te_set, h_set = compute_te_from_prob(pyyx)
    return te_set, h_set, pyyx


@njit
def compute_te_from_prob(prob3):
    nlag_max = prob3.shape[3]
    te_set = np.zeros((2, nlag_max))
    h_set = np.zeros((2, nlag_max))
    for nl in range(nlag_max):
        for nd in range(2):
            _p3 = prob3[:, :, :, nl, nd]
            py = _p3.sum(axis=2).sum(axis=0)
            pyy = _p3.sum(axis=2)
            pyx = _p3.sum(axis=0)
            hy = compute_ent(py)
            hyy = compute_ent(pyy)
            hyx = compute_ent(pyx)
            hyyx = compute_ent(_p3)
            te = hyy + hyx - hy - hyyx
            te_set[nd, nl] = np.round(te, 10)
            h_set[nd, nl] = np.round(hy, 10)

    return te_set, h_set


@njit
def compute_dmi_from_prob(prob3):
    """
    Compute delayed mutual information of X and Y
    I(Y; X-), I(X, Y-), I(Y; Y-), I(X, X-)
    
    prob3: p(Y, Y-, X-) & P(X, X-, Y-)
    
    """
    nlag_max = prob3.shape[3]
    dmi_set = np.zeros((4, nlag_max))
    pxy = prob3.sum(axis=2).sum(axis=1)
    py, px = pxy[:, :, 0], pxy[:, :, 1]
    pxy_p = prob3.sum(axis=1).sum(axis=0)
    px_p, py_p = pxy_p[:, :, 0], pxy_p[:, :, 1]
    pjoint1 = prob3.sum(axis=1)
    pyx, pxy = pjoint1[:, :, :, 0], pjoint1[:, :, :, 1]
    pjoint2 = prob3.sum(axis=2)
    pyy, pxx = pjoint2[:, :, :, 0], pjoint1[:, :, :, 1]
    for nl in range(nlag_max):
        dmi_set[0, nl] = compute_ent(py[:, nl]) + compute_ent(px_p[:, nl]) - compute_ent(pyx[:, :, nl])
        dmi_set[1, nl] = compute_ent(px[:, nl]) + compute_ent(py_p[:, nl]) - compute_ent(pxy[:, :, nl])
        dmi_set[2, nl] = compute_ent(py[:, nl]) + compute_ent(py_p[:, nl]) - compute_ent(pyy[:, :, nl])
        dmi_set[3, nl] = compute_ent(px[:, nl]) + compute_ent(px_p[:, nl]) - compute_ent(pxx[:, :, nl])

    return dmi_set


# def norm_prob3(prob3):
#     prob3 = prob3.astype(float)
#     nlag_max = prob3.shape[3]
#     for nd in range(2):
#         for nl in range(nlag_max):
#             prob3[:, :, :, nl, nd] /= prob3[:, :, :, nl, nd].sum()

#     return prob3

def norm_prob3(prob3):
    psum = np.sum(prob3, axis=(0,1,2), keepdims=True)
    return prob3 / psum


def compute_te(xd, yd, num_bin, nlag_max):

    def _isint(x):
        return isinstance(x[0], np.int64)

    if not (_isint(xd) and _isint(yd)):
        raise ValueError("Check the input variable datatype")
    if xd.max() >= num_bin or yd.max() >= num_bin:
        raise ValueError("Unexpect bin number in xd : %d, yd: %d" % (xd.max(), yd.max()))
    te, h, prob3 = _compute_te(xd, yd, num_bin, nlag_max)
    if np.any(te < 0):
        case_id = report_error_case(xd, yd, num_bin, nlag_max, te)
    return (
     te, h, prob3)


def report_error_case(xd, yd, num_bin, nlag_max, te):
    import os, pickle as pkl
    fdir = "./error_case"
    case_id = len([f for f in os.listdir(fdir) if ".pkl" in f])
    with open(os.path.join(fdir, "#%04d.pkl" % case_id), "wb") as fp:
        pkl.dump(dict(xd=xd, yd=yd, num_bin=num_bin, nlag_max=nlag_max, te=te), fp)
    return case_id

# okay decompiling hhinfo.cpython-37.pyc
