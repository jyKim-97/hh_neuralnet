import numpy as np

import sys
sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
# import hhtools
import hhsignal


_teq = 0.5


def detect_osc_motif(detail_data: dict, amp_range: dict, mbin_t=0.01, wbin_t=0.5, srate=2000):
    # detail_data: from summary_obj.load_detail(., .)
    # amp_range: dict, contains 'fpop' and 'spop'
    psd, fpsd, tpsd = compute_stfft_all(detail_data, mbin_t=mbin_t, wbin_t=wbin_t, srate=srate)
    words = compute_osc_bit(psd[1:], fpsd, tpsd, amp_range, q=80, min_len=2, cat_th=2)
    
    return get_motif_boundary(words, tpsd)


def compute_stfft_all(data, frange=(5, 100), mbin_t=0.01, wbin_t=0.5, srate=2000):
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
    
    if len(bd_idx) > 0 and len(bd_idx[-1]) == 1:
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
def compute_osc_trit(psd_set, fpsd, tpsd, amp_range, q=80, min_len=2, cat_th=2):
    if len(psd_set) != 2:
        raise ValueError("Unexpected length of psd_set")
    
    # compute tenary bit (0/1/2): 
    words_p = np.zeros(len(tpsd))
    words_n = np.zeros(len(tpsd))
    
    for tp in range(2):
        tp_label = "fpop" if tp == 0 else "spop"
        psd = psd_set[tp]
        
        for nf, ar in enumerate(amp_range[tp_label]):
            if len(ar) == 0:
                words_n += 2**(2*tp+nf)
                continue
        
            y = psd[(fpsd >= ar[0]) & (fpsd < ar[1]), :].mean(axis=0)
            
            # compute boundary when the amplitude is increased
            bd_idy_p, _ = pick_osc(y, min_len=min_len, cat_th=cat_th, q=q)
            for bd in bd_idy_p:
                words_p[bd[0]:bd[1]] += 2**(2*tp+nf)
            
            # compute boundary when the amplitude is decreased
            bd_idy_n, _ = pick_osc(y, min_len=min_len, cat_th=cat_th, q=100-q, reversed=True)
            for bd in bd_idy_n:
                words_n[bd[0]:bd[1]] += 2**(2*tp+nf)
                
    words_p = align_cobit(words_p)
    words_n = align_cobit(words_n)
    
    bd_words = get_boundary(words_p + words_n >= 15)
    bd_words = bd_words[(bd_words[:, 1] - bd_words[:, 0] >= min_len), :]
    bd_words = cat_boundary(bd_words, cat_th)
    
    words = np.zeros_like(words_p) - 1
    for bd in bd_words:
        words[bd[0]:bd[1]] = words_p[bd[0]]
    words = align_cobit(words)
    
    return words# , words_n
            

def compute_osc_bit(psd_set, fpsd, tpsd, amp_range, q=80, min_len=2, cat_th=2, reversed=False):
    
    words = np.zeros(len(tpsd))
    for tp in range(2): # fast / slow subpop
        
        tp_label = "fpop" if tp == 0 else "spop"
        psd = psd_set[tp]
        
        for nf, ar in enumerate(amp_range[tp_label]):
            if len(ar) == 0: continue
        
            y = psd[(fpsd >= ar[0]) & (fpsd < ar[1]), :].mean(axis=0)
            bd_idy, _ = pick_osc(y, min_len=min_len, cat_th=cat_th, q=q, reversed=reversed)
            
            for bd in bd_idy:
                words[bd[0]:bd[1]] += 2**(2*tp+nf)
    
    # return words
    return align_cobit(words)
                

def align_cobit(_words, digit=4):
    
    digit = 4
    nstack_cut = 4
    words = _words.copy()
    min_words = words.min()
    
    nstack = 0
    nt_start = None
    w1 = None
    
    nt = 0
    while nt < len(words):
        if nt < 0:
            raise ValueError("nt becomes negative")
        
        if words[nt] > min_words:
            if w1 is None: # write w1 only at the beginning
                # update start index
                if nt_start is None:
                    nt_start = nt
                w1 = words[nt]
                wc = np.array(dec2bin(w1, digit))
                nstack = 0
                
            elif words[nt] != w1: # mixed case
                # nstack += 1
                w2 = words[nt]
                if np.any(dec2bin(w2, digit) & wc):
                    w1 = w2
                    nstack = 0 
                else:
                    nstack += 1 # count stack
                    
            else: # zero case
                nstack = 0
                
        else:
            nstack += 1
        
        if nstack == nstack_cut and w1 is not None:
            nt -= nstack_cut-1
            
            count_bit = np.zeros(digit)
            w, wbit = None, None
            for n in np.arange(nt_start, nt):
                if words[n] != w:
                    w = words[n]
                    wbit = dec2bin(int(words[n]), digit)
                
                count_bit += wbit
            
            count = np.zeros(2**digit)
            for w in range(1, 16):
                wbit = dec2bin(w, digit)
                count[w] = count_bit[wbit].min()
            
            wcand = np.where(count > (nt-nt_start)*0.3)[0]
            # print(wcand)
            cond = np.zeros(digit, dtype=bool)
            for w in wcand:
                cond = cond | dec2bin(w, digit)
                
            wnew = 0
            for i in range(digit):
                if cond[i]: wnew += 2**i

            # w_set = np.unique(words[nt_start:nt])
            # cond = np.ones(digit, dtype=bool)
            # for w in w_set:
            #     cond = cond | dec2bin(w, digit)
                
            # wnew = 0
            # for i in range(digit):
            #     if cond[i]: wnew += 2**i

            
            # if not reversed:
            #     # cond = np.zeros(digit, dtype=bool)
            #     cond = np.ones(digit, dtype=bool)
            #     for w in w_set:
            #         # cond = cond | dec2bin(w, digit)
            #         cond = cond & dec2bin(w, digit)
            # else:
            #     cond = np.ones(digit, dtype=bool)
            #     for w in w_set:
            #         cond = cond & dec2bin(w, digit)
            
            words[nt_start:nt] = wnew
            # words[nt_start:nt] = words[nt]
            
            nstack = 0
            nt_start = None
            w1 = None
            
        nt += 1
        
    # remove if the length of words lower than nstack_cut
    words_new = np.zeros_like(words)
    for bd in get_boundary(words > 0):
        if bd[1]-bd[0] >= nstack_cut:
            words_new[bd[0]:bd[1]] = words[bd[0]]
    
    return words_new.astype(np.int16)
    # return words.astype(np.int16)

# 
def get_motif_boundary(words, tpsd):
    bd_motif_tmp = []
    for w in range(1, 16):
        bd = get_boundary(words == w)
        for i in range(len(bd)):
            bd_motif_tmp.append(dict(id=w, range=tpsd[bd[i]]))
    
    # sort
    t0 = [motif["range"][0] for motif in bd_motif_tmp]
    id_sort = np.argsort(t0)
    bd_motif = [bd_motif_tmp[i] for i in id_sort]
    
    return bd_motif


# # ---- count
def count_motif(words, digit=4):
    word_counts = np.zeros(16)
    word_length = np.zeros(16)
    osc_motif = get_motif_boundary(words, np.arange(len(words)))
    
    for om in osc_motif:
        idw = int(om["id"])
        word_counts[idw] += 1
        word_length[idw] += om["range"][1]-om["range"][0]

    return word_counts, word_length


# ---- get labels
def get_motif_labels():
    lb = []
    lb_f, lb_s = ["_", "f"], ["_", "s"]
    
    for i in range(16):
        x2 = dec2bin(i, 4)
        
        # s = "F(%s%s)S(%s%s)"%(lb_s[x2[3]], lb_f[x2[2]], lb_s[x2[1]], lb_f[x2[0]])
        # s = "F(%s%s)S(%s%s)"%(lb_f[x2[2]], lb_s[x2[3]], lb_f[x2[0]], lb_s[x2[1]])
        lb.append(
            # "F(%s%s)S(%s%s)"%(lb_f[x2[2]], lb_s[x2[3]], lb_f[x2[0]], lb_s[x2[1]])
            "F(%s%s)S(%s%s)"%(lb_f[x2[1]], lb_s[x2[0]], lb_f[x2[3]], lb_s[x2[2]])
        )
        # lb.append("%d%d%d%d"%(x2[3], x2[2], x2[1], x2[0]))
    return lb


def argsort_motif_labels():
    # get index to sort motif labels
    # label will be sort as
    
    # F(__)S(__)
    # F(f_)S(_s)
    # F(__)S(_s)
    # F(_s)S(_s)
    # F(fs)S(_s)
    # F(f_)S(__)
    # F(f_)S(f_)
    # F(f_)S(fs)
    # F(fs)S(fs)
    # F(_s)S(f_)
    # F(__)S(f_)
    # F(__)S(fs)
    # F(fs)S(f_)
    # F(_s)S(__)
    # F(fs)S(__)
    # F(_s)S(fs)
    
    return [0, 6, # indep
            4, 5, 7, # interaction with slow
            2, 10, 14, # interaction with fast
            15, # interaction with both 
            9, 8, 12, 11, 1, 3, 13]