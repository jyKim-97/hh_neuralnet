import numpy as np
import matplotlib.pyplot as plt


def load_spk(fname):
    with open(fname, "rb") as fid:
        data = np.fromfile(fid, dtype=np.int32)
    N, nmax = data[0], data[1]
    num_spk = data[2:N+2]
    step_spks = []
    cum = N+2
    for n in range(N):
        step_spks.append(data[cum:cum+num_spk[n]])
        cum += num_spk[n]
    return step_spks, nmax
    

def load_dynamics(fname, N, fs=None):
    with open(fname, "rb") as fid:
        xs = np.fromfile(fid, dtype=np.float32).reshape([-1, N])
    ts = None
    if fs is not None:
        l = xs.shape[0]
        ts = np.arange(l)/fs
    return xs, ts


def load_vlfp(fname):
    with open(fname, "rb") as fid:
        data = np.fromfile(fid, dtype=np.float32)
    num_types = int(data[0])
    max_step  = int(data[1])
    
    n0 = 2
    l = (len(data)-2)//(num_types+1)
    vlfps = []
    for n in range(num_types+1):
        vlfps.append(data[n0:n0+l])
    return vlfps


def load_network(fname):
    ntk_in = []
    ntk_win = []
    with open(fname, "r") as fid:
        line = fid.readline()
        line = fid.readline()
        while line:
            tmp = line.split("<-")
            npost = int(tmp[0])
            npre = int(tmp[1].split(",")[0])
            if len(ntk_in) <= npost:
                ntk_in.append([])
                ntk_win.append([])
            ntk_in[npost].append(npre)
            ntk_win[npost].append(float(tmp[1].split(",")[1][:-1]))
            line = fid.readline()
    return ntk_in, ntk_win
        

def draw_spk(step_spk, dt=0.01, xl=None, color_ranges=None, colors=None, ms=1):
    if color_ranges is not None:
        if colors is None:
            print("Type the colors")
            return
        else:
            if len(color_ranges) != len(colors):
                print("The length of color and color range does not match")
                return
    
    cid = 0
    N = len(step_spk)
    for n in range(N):
        t_spk = np.array(step_spk[n]) * dt
        if xl is not None:
            t_spk = t_spk[(t_spk >= xl[0]) & (t_spk <= xl[1])]
        if color_ranges is not None:
            if n >= color_ranges[cid]:
                cid += 1
            c = colors[cid]
        else:
            c = 'k'

        plt.plot(t_spk, np.ones_like(t_spk)*n, '.', ms=ms, c=c)
    plt.xlim(xl)
