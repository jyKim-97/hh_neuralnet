import numpy as np
import matplotlib.pyplot as plt
import sys
from numba import jit

sys.path.append("/home/jungyoung/Project/hh_neuralnet/include/")
import hhtools


def get_pth_percentile(vec, q):
    return np.sort(vec)[int(q/100*len(vec))]


def draw_binarize_psd(psd, pth, x=None, y=None, flim=None, ylabel=None, xlabel=None, axs=None):

    gen_subplot=False
    if axs is None:
        gen_subplot = True
        axs = []

    if gen_subplot is False:
        plt.axes(axs[0])
    else:
        axs.append(plt.subplot(121))

    hhtools.imshow_xy(psd, x=x, y=y, cmap="jet", interpolation="spline16")
    plt.ylim(flim)
    plt.ylabel(ylabel, fontsize=14)
    plt.xlabel(xlabel, fontsize=14)

    if gen_subplot is False:
        plt.axes(axs[1])
    else:
        axs.append(plt.subplot(121))

    hhtools.imshow_xy(psd>=pth, x=x, y=y, cmap="gray", interpolation="none")
    plt.ylim(flim)  
    plt.xlabel(xlabel, fontsize=14)

    return axs


def draw_burst_attrib(im_class, burst_f, burst_range, tpsd=None, fpsd=None, flim=None):

    flim = [min(fpsd), max(fpsd)] if flim is None else flim

    num_class = len(np.unique(im_class))-1
    cmap = plt.get_cmap("jet", num_class)

    im_show = im_class.copy()
    im_show[im_show == -1] = np.nan
    hhtools.imshow_xy(im_show, x=tpsd, y=fpsd, cmap=cmap, interpolation="None")
    for n in range(num_class):
        bd = burst_range[n, :].astype(int)
        plt.plot([tpsd[bd[0]]]*2, flim, '--', c=cmap(n), lw=0.5, zorder=-1)
        plt.plot([tpsd[bd[1]]]*2, flim, '--', c=cmap(n), lw=0.5, zorder=-1)
        t0 = (tpsd[bd[0]] + tpsd[bd[1]])/2
        plt.plot(t0, burst_f[n], 'wo', ms=3)

    plt.ylim(flim)


# @jit(nopython=True)
def extract_burst_attrib(psd, fpsd, im_class):

    num_class = len(np.unique(im_class)) - 1
    burst_f = np.zeros(num_class)
    burst_range = np.zeros([num_class, 2])
    for cid in range(num_class):
        id_row, id_col = np.where(im_class == cid)

        burst_range[cid, 0] = np.min(id_col)
        burst_range[cid, 1] = np.max(id_col)

        # weighted sum
        fc = 0
        sum_amp = 0
        for nr, nc in zip(id_row, id_col):
            fc += psd[nr, nc] * fpsd[nr]
            sum_amp += psd[nr, nc]
        burst_f[cid] = fc / sum_amp
        
    return burst_f, burst_range


@jit(nopython=True)
def find_blob(im_binary):
    im_class = np.zeros(im_binary.shape) - 1
    num_row, num_col = im_binary.shape
    cid = 0
    for nr in range(num_row):
        for nc in range(num_col):
            if im_binary[nr, nc] == 1:
                flag = search_blob(nr, nc, cid, im_class, im_binary)
                if flag == 0:
                    cid += 1
    return im_class

@jit(nopython=True)
def search_blob(nr0, nc0, cid, im_class, im_binary):
    if im_class[nr0, nc0] != -1:
        return -1

    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    def is_out(nr, nc):
        num_row, num_col = im_class.shape
        return ((nr < 0) or (nr >= num_row) or (nc < 0) or (nc >= num_col))
    
    points = [(nr0, nc0)]
    im_class[nr0, nc0] = cid

    while len(points) > 0:
        nr, nc = points.pop()
        for d in dirs:
            nr_new, nc_new = nr+d[0], nc+d[1]
            if is_out(nr_new, nc_new):
                continue

            if im_binary[nr_new, nc_new] == 0 or im_class[nr_new, nc_new] != -1:
                continue
            
            im_class[nr_new, nc_new] = cid
            points.append((nr_new, nc_new))
    
    return 0