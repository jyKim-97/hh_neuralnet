import numpy as np

import sys
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
import hhsignal


_teq = 0.5

def compute_stfft_all(data, frange=(5, 100), mbin_t=0.1, wbin_t=0.5, srate=2000):
    psd_set = []
    t = data["ts"]
    for v in data["vlfp"]:
        veq, teq = hhsignal.get_eq_dynamics(v, t, _teq)
        psd, fpsd, tpsd = hhsignal.get_stfft(veq, teq, srate, mbin_t=mbin_t, wbin_t=wbin_t, frange=frange)
        psd_set.append(psd)
    return psd_set, fpsd, tpsd


def dec2bin(w, digit):
    if digit == 0:
        return []
    
    dw = w - 2**(digit-1)
    if dw >= 0:
        return dec2bin(dw, digit-1) + [True]
    else:
        return dec2bin(w, digit-1) + [False]
    
    
# ----- extract oscillatory activity ----- #
def pick_osc(y, min_len=0, cat_th=3, q=90, reversed=False):
    yth = np.percentile(y, q)
    if reversed:
        idy = y < yth
    else:
        idy = y >= yth
    bd_idy = get_boundary(idy)
    bd_idy = bd_idy[(bd_idy[:, 1] - bd_idy[:, 0] >= min_len), :]
    bd_idy = cat_boundary(bd_idy, cat_th)
    
    return bd_idy, yth


def get_boundary(bool_idx):
    bd_idx = []
    flag = True
    for i in range(len(bool_idx)):
        if flag and bool_idx[i]:
            bd_idx.append([i])
            flag = False
        elif not flag and not bool_idx[i]:
            bd_idx[-1].append(i-1)
            flag = True
    
    if len(bd_idx[-1]) == 1:
        bd_idx[-1].append(len(bool_idx)-1)
    
    return np.array(bd_idx, dtype=int)


def cat_boundary(bd_idx, th=2):
    if len(bd_idx) == 0:
        return []
    
    bd_idx_cat = [bd_idx[0]]
    for n in range(1, len(bd_idx)):
        if bd_idx[n][0] - bd_idx_cat[-1][1] <= th:
            bd_idx_cat[-1][1] = bd_idx[n][1]
        else:
            bd_idx_cat.append(bd_idx[n])
    return bd_idx_cat


# ----- convert oscillatory activity to code words ----- #

def compute_osc_bit(psd_set, fpsd, tpsd, amp_range, q=80, min_len=2, cat_th=2):
    
    words = np.zeros(len(tpsd))
    for tp in range(2): # fast / slow subpop
        
        tp_label = "fpop" if tp == 0 else "spop"
        psd = psd_set[tp]
        
        for nf, ar in enumerate(amp_range[tp_label]):
            if len(ar) == 0: continue
        
            y = psd[(fpsd >= ar[0]) & (fpsd < ar[1]), :].mean(axis=0)
            bd_idy, _ = pick_osc(y, min_len=min_len, cat_th=cat_th, q=q)
            
            for bd in bd_idy:
                words[bd[0]:bd[1]] += 2**(2*tp+nf)
                
    return align_cobit(words, 4)    


def align_cobit(_words, digit=4):
    digit = 4
    pos, mixed = False, False
    words = _words.copy()
    for n in range(len(words)):
        if not pos and words[n] > 0:
            id0 = n
            w = words[n]
            pos = True
            
        if pos and words[n] != w:
            mixed = True
        
        if words[n] == 0:
            if mixed:
                wu = np.unique(words[id0:n])
                cond = np.zeros(digit, dtype=bool)
                for w in wu:
                    cond = cond | dec2bin(w, digit)

                wnew = 0
                for i in range(digit):
                    if cond[i]: wnew += 2**i
                
                words[id0:n] = wnew
            
            pos, mixed = False, False
            
    return words.astype(np.int16)


# 
def get_motif_boundary(words, tpsd):
    bd = get_boundary(words > 0)
    bd_motif = []
    for i in range(len(bd)):
        bd_motif.append(dict(
            id=words[bd[i][0]],
            range=tpsd[bd[i]]
        ))
    
    return bd_motif


# ---- count
def count_motif(words, digit=4):
    word_count = np.zeros(16)
    bd_words = get_boundary(words > 0)
    for i in range(len(bd_words)):
        w = words[bd_words[i][0]].astype(int)
        word_count[w] += 1
    return word_count


# ---- get labels
def get_motif_labels():
    lb = []
    lb_f, lb_s = ["_", "f"], ["_", "s"]
    
    for i in range(16):
        x2 = dec2bin(i, 4)
        
        # s = "F(%s%s)S(%s%s)"%(lb_s[x2[3]], lb_f[x2[2]], lb_s[x2[1]], lb_f[x2[0]])
        # s = "F(%s%s)S(%s%s)"%(lb_f[x2[2]], lb_s[x2[3]], lb_f[x2[0]], lb_s[x2[1]])
        lb.append(
            "F(%s%s)S(%s%s)"%(lb_f[x2[2]], lb_s[x2[3]], lb_f[x2[0]], lb_s[x2[1]])
        )
        # lb.append("%d%d%d%d"%(x2[3], x2[2], x2[1], x2[0]))
    return lb