import numpy as np


def perm_test(x, y, N=1000, verbose=False, seed=42):
    import matplotlib.pyplot as plt
    
    nx = len(x)
    dm = x.mean() - y.mean()
    
    dm_perm = np.zeros(N)
    xypool = np.concatenate((x, y))
    
    np.random.seed(seed)
    for n in range(N):
        np.random.shuffle(xypool)

        xp = xypool[:nx]
        yp = xypool[nx:]

        dm_perm[n] = xp.mean() - yp.mean()
    
    if verbose:
        plt.figure(figsize=(6, 3))
        plt.subplot(121)
        plt.hist(x, alpha=0.2, density=True)
        plt.hist(y, alpha=0.2, density=True)
        
        plt.subplot(122)
        plt.hist(dm_perm, density=True)
        yl = plt.ylim()
        plt.vlines(dm, yl[0], yl[1], color='k', linestyle='--')
        plt.show()
        
    if np.percentile(dm_perm, 1) > dm or np.percentile(dm_perm, 99) < dm:
        return True
    else:
        return False        
    
    
def conf_test(x, y, alpha=0.05, verbose=False):
    import matplotlib.pyplot as plt
    
    p = 100*alpha
    xbot = np.percentile(x, p)
    xtop = np.percentile(x, 100-p)
    
    ybot = np.percentile(y, p)
    ytop = np.percentile(y, 100-p)
    
    if verbose:
        plt.figure(figsize=(4,3))
        plt.hist(x, density=True, alpha=0.2)
        plt.hist(y, density=True, alpha=0.2)
        yl = plt.ylim()
        plt.vlines(xbot, yl[0], yl[1], color='C0', linestyle='--')
        plt.vlines(xtop, yl[0], yl[1], color='C0', linestyle='--')
        plt.vlines(ybot, yl[0], yl[1], color='C1', linestyle='--')
        plt.vlines(ytop, yl[0], yl[1], color='C1', linestyle='--')
    
    if (xbot > ytop) or (xtop < ybot):
        return True
    else:
        return False
    
    
def te_stat_test(te_data, method="conf", **kwstat):
    assert method in ("conf", "perm")
    
    if method == "conf":
        stat_test = conf_test
    else:
        stat_test = perm_test
    
    sig_arr = np.zeros(te_data["te"].shape[1:], dtype=bool)
    
    npair, nlen = 2, te_data["te"].shape[-1]
    for ntp in range(npair):
        for nd in range(nlen):
            sig_arr[ntp, nd] = stat_test(te_data["te"][:,ntp,nd],
                                         te_data["te_surr"][:,ntp,nd], 
                                         verbose=False,
                                         **kwstat)
    
    return sig_arr