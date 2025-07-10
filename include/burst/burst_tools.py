import numpy as np
import matplotlib.pyplot as plt
import sys
from numba import jit, typed
# from numba.typed import List
from typing import Dict, List
import pandas as pd

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
    plt.colorbar()

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
    if num_class == 0:
        return

    cmap = plt.get_cmap("turbo", num_class)

    im_show = im_class.copy()
    im_show[im_show == -1] = np.nan
    hhtools.imshow_xy(im_show, x=tpsd, y=fpsd, cmap=cmap, interpolation="None")
    for n in range(num_class):
        bd = burst_range[n, :].astype(int)
        plt.plot([tpsd[bd[0]]]*2, flim, '--', c=cmap(n), lw=0.5, zorder=-1)
        plt.plot([tpsd[bd[1]]]*2, flim, '--', c=cmap(n), lw=0.5, zorder=-1)
        t0 = (tpsd[bd[0]] + tpsd[bd[1]])/2
        plt.plot(t0, burst_f[n], 'wo', ms=1.5)

    plt.ylim(flim)


# @jit(nopython=True)
def extract_burst_attrib(psd, fpsd, burst_map):

    num_class = np.max(burst_map)
    burst_f = np.zeros(num_class)
    burst_range = np.zeros([num_class, 2])
    burst_amp = np.zeros(num_class)
    for cid in range(num_class):
        id_row, id_col = np.where(burst_map == cid+1)

        burst_range[cid, 0] = np.min(id_col)
        burst_range[cid, 1] = np.max(id_col)

        # weighted sum
        fc = 0
        sum_amp = 0
        for nr, nc in zip(id_row, id_col):
            fc += psd[nr, nc] * fpsd[nr]
            sum_amp += psd[nr, nc]
        burst_f[cid] = fc / sum_amp
        burst_amp[cid] = sum_amp / len(id_row)
        
    return burst_f, burst_range, burst_amp


@jit(nopython=True)
def find_blob(im_binary, im_class=None):
    if im_class is None:
        im_class = np.zeros(im_binary.shape)
    num_row, num_col = im_binary.shape
    cid = 1
    for nr in range(num_row):
        for nc in range(num_col):
            if im_binary[nr, nc] == 1:
                flag = search_blob(nr, nc, cid, im_class, im_binary)
                if flag == 0:
                    cid += 1
    return im_class


def find_blob_filtration(psd, psd_th_m, psd_th_s,
                             std_min=3.3, std_max=10, std_step=0.1,
                             nmin_width=3):
    
    std_ratio_set = np.arange(std_max, std_min-std_step/2, -std_step)

    burst_map = np.zeros(psd.shape, dtype=int)
    null_map = np.zeros(psd.shape)
    burst_start_pts = typed.List([(0, 0)])
    num_burst = 0

    for ns in range(len(std_ratio_set)):
        s = std_ratio_set[ns]
        pth = psd_th_m + s * psd_th_s

        im_binary = psd >= pth
        expand_null(im_binary, null_map) # 1st 

        if len(burst_start_pts) > 1:
            im_expand, overlapped = expand_exist_clusters(im_binary, burst_start_pts)

        for cid in range(1, num_burst+1):
            if overlapped[cid] == 0:
                burst_map[im_expand == cid] = cid
            else:
                null_map[im_expand == cid] = 1

        im_class_new = explore_new_clusters(im_binary, null_map, burst_map)
        
        cid_new = np.unique(im_class_new)
        for cid in cid_new:
            if cid == 0: continue
            is_cid = im_class_new == cid
            br, bc = np.where(is_cid)
            if np.max(bc) - np.min(bc) < nmin_width:
                continue

            burst_start_pts.append((br[0], bc[0]))
            burst_map[is_cid] = num_burst + 1
            num_burst += 1

    return burst_map.astype(int)


@jit(nopython=True)
def expand_null(im_binary, null_map):
    num_row, num_col = null_map.shape
    for nr in range(num_row):
        for nc in range(num_col):
            if im_binary[nr, nc] == 1 and null_map[nr, nc] == 1:
                flag = search_blob(nr, nc, 1, null_map, im_binary)

@jit(nopython=True)
def expand_exist_clusters(im_binary, burst_start_pts):
    im_class = np.zeros(im_binary.shape)

    overlapped = np.zeros(len(burst_start_pts)+1)
    for cid in range(1, len(burst_start_pts)):
        nr =  burst_start_pts[cid][0]
        nc =  burst_start_pts[cid][1]
        flag = search_blob(nr, nc, cid, im_class, im_binary)

        if flag != 0:
            overlapped[cid] = 1
            if flag > 0: overlapped[int(flag)] = 1

    return im_class, overlapped

@jit(nopython=True)
def explore_new_clusters(im_binary, null_map, burst_map):
    im_class = np.zeros(null_map.shape)
    num_row, num_col = im_class.shape

    cid = 1
    for nr in range(num_row):
        for nc in range(num_col):
            if im_binary[nr, nc] == 1 and null_map[nr, nc] == 0 and burst_map[nr, nc] == 0:
                flag = search_blob(nr, nc, cid, im_class, im_binary)
                if flag == 0: cid += 1

    return im_class


@jit(nopython=True)
def search_blob(nr0, nc0, cid, im_class, im_binary):
    if cid < 1:
        raise ValueError("blob id must be larger than 0")

    if im_class[nr0, nc0] != 0:
        return im_class[nr0, nc0]
    
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
    def is_out(nr, nc):
        num_row, num_col = im_class.shape
        return ((nr < 0) or (nr >= num_row) or (nc < 0) or (nc >= num_col))

    flag = 0    
    points = [(nr0, nc0)]
    im_class[nr0, nc0] = cid

    while len(points) > 0:
        nr, nc = points.pop()
        for d in dirs:
            nr_new, nc_new = nr+d[0], nc+d[1]

            if is_out(nr_new, nc_new) or im_binary[nr_new, nc_new] == 0:
                continue

            if im_class[nr_new, nc_new] != 0:
                if im_class[nr_new, nc_new] != cid:
                    flag = im_class[nr_new, nc_new]
                continue
            
            im_class[nr_new, nc_new] = cid
            points.append((nr_new, nc_new))
    
    return flag


def align_burst(burst_info):
    data = []
    col_names = ["burst_f", "burst_range", "burst_amp"]
    for n in range(len(burst_info["burst_f"])):
        for i in range(len(burst_info["burst_f"][n])):
            data_sub = [
                burst_info["burst_f"][n][i],
                burst_info["burst_amp"][n][i],
                burst_info["burst_range"][n][i][0],
                burst_info["burst_range"][n][i][1],
                # burst_info["burst_range"][n][i][1] - burst_info["burst_range"][n][i][0],
                burst_info["cluster_id"][n],
                burst_info["pop_type"][n]
            ]
            data.append(data_sub)
    df_burst = pd.DataFrame(data, columns=["burst_f", "burst_amp", "burst_t0", "burst_t1", "cluster_id", "pop_type"])
    df_burst["burst_duration"] = df_burst["burst_t1"] - df_burst["burst_t0"]
    if df_burst["cluster_id"].min() == 0:
        df_burst["cluster_id"] += 1
    
    return df_burst


# cleaning
@jit(nopython=True, cache=True)
def find_fid(f: int, amp_range: np.ndarray)->int:
    m = (amp_range[0][1] + amp_range[1][0]) / 2
    if f < m:
        return 0
    elif f > m:
        return 1
    
    # if f < amp_range[0][1]+2:
    #     return 0
    # elif f < amp_range[1][1]+2:
    #     return 1
    # else:
    #     return -1


def to_ndarray_bprops(bprops):
    for npop in range(2):
        bprops[npop]["burst_range"] = np.array(bprops[npop]["burst_range"], dtype=int)
        bprops[npop]["burst_f"] = np.array(bprops[npop]["burst_f"])
        bprops[npop]["burst_amp"] = np.array(bprops[npop]["burst_amp"])
        bprops[npop]["id_trial"] = np.array(bprops[npop]["id_trial"], dtype=int)
    return bprops


def remove_short_burst(bprops):
    bprops_new = []
    for npop in range(2):
        bprops_new.append({
            "burst_range": [],
            "burst_f": [],
            "burst_amp": [],
            "id_trial": [],
            "tpsd": bprops[npop]["tpsd"]
        })
        
        dt = bprops[npop]["tpsd"][1] - bprops[npop]["tpsd"][0]
        for n, (r, f) in enumerate(zip(bprops[npop]["burst_range"], bprops[npop]["burst_f"])):
            if r[1] - r[0] < 3/f/dt: # less then 3 cycles
                continue
            
            bprops_new[npop]["burst_range"].append(r)
            bprops_new[npop]["burst_f"].append(f)
            bprops_new[npop]["burst_amp"].append(bprops[npop]["burst_amp"][n])
            bprops_new[npop]["id_trial"].append(bprops[npop]["id_trial"][n])
            
    return to_ndarray_bprops(bprops_new)


@jit(nopython=True, cache=True)
def _concatentate_burst_single(burst_range: List, burst_f: List, burst_amp: List, id_trial: List, amp_range: np.ndarray):
    brange_new = []
    bf_new = []
    ba_new = []
    id_trial_new = []
    
    N = len(id_trial)
    is_collected = np.zeros(N)

    for i in range(N):
        if is_collected[i] == 1:
            continue
        id_i = id_trial[i]
        nr = [burst_range[i][0], burst_range[i][1]]
        id_concat = [i]
        id_repr = i
        is_collected[i] = 1
        nf = find_fid(burst_f[i], amp_range)
    
        is_changed = True
        while is_changed:
            is_changed = False
            for j in range(N):
                if is_collected[j] == 1:
                    continue
                id_j = id_trial[j]
                if id_i != id_j: 
                    continue
                if nf != find_fid(burst_f[j], amp_range):
                    continue
                
                nr1 = burst_range[j]
                if not (nr[1]+2 < nr1[0] or nr[0] > nr1[1]+2):
                    nr[0] = min(nr[0], nr1[0])
                    nr[1] = max(nr[1], nr1[1])
                    is_collected[j] = 1
                    id_concat.append(j)
                    is_changed = True
                    
                    if burst_amp[j] > burst_amp[id_repr]:
                        id_repr = j
                    
        
        brange_new.append(nr)
        bf_new.append(burst_f[id_repr])
        ba_new.append(burst_amp[id_repr])
        id_trial_new.append(id_trial[id_repr])
    
    return brange_new, bf_new, ba_new, id_trial_new
        

def concatenate_burst(bprops: List[Dict], amp_range): 
    # concatenatie burst if they are close enough
    bprops_new = []
    ramp_range = resize_amp_range(amp_range)
    for npop in range(2):
        
        # ar = amp_range["fpop" if npop == 0 else "spop"]
        # if len(ar[0]) == 0:
        #     ar[0] = [0, 0]
        # elif len(ar[1]) == 0:
        #     ar[1] = [0, 0]
        # ar = np.array(ar)
        ar = ramp_range[npop]

        brange_new, bf_new, ba_new, id_trial_new = _concatentate_burst_single(
            bprops[npop]["burst_range"],
            bprops[npop]["burst_f"],
            bprops[npop]["burst_amp"],
            bprops[npop]["id_trial"],
            ar
        )
        
        bprops_new.append({
            "burst_range": brange_new,
            "burst_f": bf_new,
            "burst_amp": ba_new,
            "id_trial": id_trial_new,
            "tpsd": bprops[npop]["tpsd"]
        })
        
            
    return to_ndarray_bprops(bprops_new)


def resize_amp_range(amp_range):
    ramp_range = []
    for npop in range(2):
        ar = amp_range["fpop" if npop == 0 else "spop"]
        if len(ar[0]) == 0:
            ar[0] = [0, 0]
        elif len(ar[1]) == 0:
            ar[1] = [0, 0]
        ramp_range.append(ar)
    return np.array(ramp_range)


def identify_burst_fid(bprop, ramp_range):
    bprop["burst_fid"] = np.zeros(len(bprop["burst_f"]), dtype=int)
    for n in range(len(bprop["burst_f"])):
        bprop["burst_fid"][n] = find_fid(bprop["burst_f"][n], ramp_range)
    
    
