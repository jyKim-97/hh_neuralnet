# from re import S
import numpy as np
import os
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
    fs = float(data[1])
    
    n0 = 2
    l = (len(data)-2)//(num_types+1)
    vlfps = []
    for n in range(num_types+1):
        vlfps.append(data[n0:n0+l])
    return vlfps, fs


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
            while len(ntk_in) <= npost:
                ntk_in.append([])
                ntk_win.append([])
            ntk_in[npost].append(npre)
            ntk_win[npost].append(float(tmp[1].split(",")[1][:-1]))
            line = fid.readline()
    return ntk_in, ntk_win


def convert_in2outdeg(ntk_in, N=None):
    if N is None:
        N = len(ntk_in)
    ntk_out = [[] for _ in range(N)]
    for n in range(N):
        for nid in ntk_in[n]:
            ntk_out[nid].append(n)
    return ntk_out


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


# def get_autocorr(x, t, tlag_max):
#     dt = t[1] - t[0]
#     idt = t >= 1
    
#     num_lag = int(tlag_max/dt)
#     arr_app = np.zeros(num_lag)
    
#     x_cp = np.copy(x)[idt]
#     x_cp = x_cp - np.average(x_cp)
#     y = np.concatenate((arr_app, x_cp, arr_app))
#     xcorr = np.correlate(y, x_cp, mode="valid")
#     l = len(xcorr)//2
#     xcorr /= xcorr[l]
#     lags = np.arange(-num_lag, num_lag+1) * dt
    
#     return xcorr, lags


def get_autocorr(x, t, tlag_max):
    dt = t[1] - t[0]
    idt = t >= 1
    
    num_lag = int(tlag_max/dt)
    arr_app = np.zeros(num_lag)
    
    x_cp = np.copy(x)[idt]
    x_cp = x_cp - np.average(x_cp)
    y = np.concatenate((arr_app, x_cp, arr_app))
    xcorr = np.correlate(y, x_cp, mode="valid")
    
    l = len(xcorr)//2
    norm = np.concatenate((np.arange(0, num_lag+1), np.arange(num_lag,0,-1)))
    norm = len(x_cp) - norm
    # norm = np.concatenate((np.arange(num_lag+1), np.arange(l,0,-1)+1))
    xcorr = xcorr/norm
    lags = np.arange(-num_lag, num_lag+1) * dt
    
    return xcorr, lags


def get_fft(x, fs, nbin=None, nbin_t=None):
    if nbin is None and nbin_t is None:
        N = len(x)
    elif nbin_t is not None:
        N = int(nbin_t*fs)
    elif nbin is not None:
        N = nbin

    yf = np.fft.fft(x, axis=0, n=N)
    yf = 2/N * np.abs(yf[:N//2])
    freq = np.linspace(0, 1/2*fs, N//2)
    return yf, freq


def get_stfft(x, t, fs, mbin_t=0.1, wbin_t=1, f_range=None, buf_size=100):

    wbin = int(wbin_t * fs)
    mbin = int(mbin_t * fs)
    window = np.hanning(wbin)
    
    ind = np.arange(wbin//2, len(t)-wbin//2, mbin, dtype=int)
    psd = np.zeros([wbin//2, len(ind)])
    
    n_id = 0
    while n_id < len(ind):
        n_buf = min([buf_size, len(ind)-n_id])
        y = np.zeros([wbin, n_buf])

        for i in range(n_buf):
            n = i + n_id
            n0 = max([0, ind[n]-wbin//2])
            n1 = min([ind[n]+wbin//2, len(t)])
            y[n0-(ind[n]-wbin//2):wbin-(ind[n]+wbin//2)+n1, i] = x[n0:n1]
        y = y * window[:,np.newaxis]
        yf, fpsd = get_fft(y, fs)
        psd[:, n_id:n_id+n_buf] = yf

        n_id += n_buf
    
    if f_range is not None:
        idf = (fpsd >= f_range[0]) & (fpsd <= f_range[1])
        psd = psd[idf, :]
        fpsd = fpsd[idf]
    tpsd = ind / fs
    
    return psd, fpsd, tpsd


def get_network_frequency(vlfp, fs=2000):
    from scipy.signal import find_peaks

    yf, freq = get_fft(vlfp, fs)
    idf = (freq >= 2) & (freq < 200)
    yf = yf[idf]
    freq = freq[idf]

    inds = find_peaks(yf)[0]
    n = np.argmax(yf[inds])
    return freq[inds[n]]


# Source code for loadding summarys
class SummaryLoader:
    def __init__(self, fdir):
        # read control param infos
        self.fdir = fdir
        self._load_controls()
        self._read_data()
    
    def _load_controls(self):
        with open(os.path.join(self.fdir, "control_params.txt"), "r") as fid:
            line = fid.readline()
            self.num_controls = [int(n) for n in line.split(",")[:-1]]
            self.control_names = []
            self.controls = dict()
            line = fid.readline()
            while line:
                tmp = line.split(":")
                self.control_names.append(tmp[0])
                self.controls[tmp[0]] = [float(x) for x in tmp[1].split(",")[:-1]]
                line = fid.readline()

    def _read_data(self):
        fnames = [f for f in os.listdir(self.fdir) if "id" in f and "result" in f]
        nums = len(fnames)
        nums_expect = 1
        for n in self.num_controls:
            nums_expect *= n
        
        if nums != nums_expect:
            print("Expected number of # results and exact file number are different!: %d/%d"%(nums, nums_expect))
        
        self.summary = {}
        self.load_success = np.ones(nums_expect)
        var_names = ["chi", "cv", "frs_m", "frs_s", "spike_sync"]
        # iinit
        for k in var_names:
            self.summary[k] = []

        for n in range(nums_expect):
            fname = os.path.join(self.fdir, "id%06d_result.txt"%(n))
            summary_single = read_summary(fname)
            if summary_single == -1:
                self.load_success[n] = 0
                for k in var_names:
                    val_prev = self.summary[k][-1]
                    self.summary[k].append(np.zeros_like(val_prev) * np.nan)
            
            else:
                for k in var_names:
                    self.summary[k].append(summary_single[k])
        
        # reshape
        for k in var_names:
            shape = np.shape(self.summary[k])[1:]
            new_shape = list(self.num_controls)+list(shape)
            self.summary[k] = np.reshape(self.summary[k], new_shape)

    def get_id(self, *nid):
        return get_id(self.num_controls, *nid)
    
    def load_detail(self, *nid):
        n = get_id(self.num_controls, *nid)
        tag = os.path.join(self.fdir, "id%06d"%(n))
        data = {}
        data["step_spk"], _ = load_spk(tag+"_spk.dat")
        data["vlfp"], fs = load_vlfp(tag+"_lfp.dat")
        data["ts"] = np.arange(len(data["vlfp"][0])) / fs
        if os.path.exists(tag+"_info.txt"):
            with open(tag+"_info.txt", "r") as fid:
                data["info"] = fid.readlines()
        else:
            data["info"] = None
        return data

    def print_params(self, *nid):
        # check
        if len(nid) != len(self.control_names):
            print("The number of arguments does not match to expected #")
        for n in range(len(nid)):
            if nid[n] >= self.num_controls[n]:
                print("Wrong index number typed, size", self.num_controls[n], "typed", nid)
        # print
        for n in range(len(nid)):
            var = self.control_names[n]
            print("%s: %f"%(var, self.controls[var][nid[n]]))


def get_id(num_xs, *nid):
    if len(nid) != len(num_xs):
        print("The number of arguments does not match to expected #")
    num_tag = 0
    stack = 1
    for n in range(len(nid)-1, -1, -1):
        if nid[n] >= num_xs[n]:
            print("Wrong index number typed, size", num_xs, "typed", nid)
            return -1
        num_tag += stack * nid[n]
        stack *= num_xs[n]
    
    return num_tag


def read_summary(fname):
    # check does the file exist
    if not os.path.exists(fname):
        return -1

    summary = dict()
    with open(fname, "r") as fid:
        line = fid.readline()
        summary["num_types"] = int(line.split(":")[1])
        line = fid.readline()
        while line:
            tmp = line.split(":")
            var_name = tmp[0]
            if var_name == "spike_syn":
                break
            summary[var_name] = [float(x) for x in tmp[1].split(",")[:-1]]
            line = fid.readline()
            
        # read spike sync
        summary["spike_sync"] = []
        line = fid.readline()
        while line:
            summary["spike_sync"].append([float(x) for x in line.split(",")[:-1]])
            line = fid.readline()
    
    for k in summary.keys():
        summary[k] = np.array(summary[k])
    
    return summary


def imshow_xy(im, x=None, y=None, **kwargs):
    extent = []
    if x is not None:
        extent = [x[0], x[-1]]
    else:
        extent = [0, np.shape(im)[1]]
    if y is not None:
        extent.extend([y[0], y[-1]])
    else:
        extent.extend([0, np.shape(im)[0]])
    
    return plt.imshow(im, aspect="auto", extent=extent, origin="lower", **kwargs)


def plot_sub(x, y, xl=None, *args, **kwargs):
    if xl is not None:
        idx = (x >= xl[0]) & (x <= xl[1])
    y = np.array(y)

    if len(y.shape) == 2:
        p =  plt.plot(x[idx], y[idx,:], *args, **kwargs)
    elif len(y.shape) == 1:
        p =  plt.plot(x[idx], y[idx], *args, **kwargs)

    plt.xlim(xl)
    return p


def extract_value_on_line(im, x, y, xq=None, yq=None):
    # im: x-column, y-row
    if len(x) != im.shape[1] or len(y) != im.shape[0]:
        print("size is different")
        
    if xq is None:
        xq = x.copy()
    
    if xq[0] < x[0] or xq[-1] > x[-1]:
        print("Range exceed")
        return
    
    def is_in(r0, rlim):
        cond = True
        for i in range(2):
            cond = cond and (rlim[i][0] <= r0[i]) and (rlim[i][1] >= r0[i])
        return cond
    
    def get_dist(r0, r1):
        return np.sqrt((r0[1] - r0[0])**2 + (r1[1] - r1[0])**2)
    
    
    num = len(xq)
    xline, yline, zline = [], [], []
    xlim = [x[0], x[-1]]
    ylim = [y[0], y[-1]]
    for n in range(num):
        if not is_in([xq[n], yq[n]], [xlim, ylim]):
            continue
            
        x0, y0 = xq[n], yq[n]
        xline.append(x0)
        yline.append(y0)
        
        # find the nearest 4 points
        nx0 = np.where(x0 <= x)[0][0]
        nx1 = np.where(x0 >= x)[0][-1]
        ny0 = np.where(y0 <= y)[0][0]
        ny1 = np.where(y0 >= y)[0][-1]
        n_pts = [[nx0, ny0], [nx0, ny1], [nx1, ny0], [nx1, ny1]]
        
        # get distance
        zs, ds = [], []
        on_the_point = False
        for i in range(4):
            ds.append(get_dist([x0, y0], [x[n_pts[i][0]], y[n_pts[i][1]]]))
            zs.append(im[n_pts[i][1], n_pts[i][0]])
            
            if ds[-1] == 0:
                on_the_point = True
                z = zs[-1]
                break
        
        if not on_the_point:
            z = 0
            dsum = np.sum(ds)
            for i in range(4):
                z += ds[i]/dsum * zs[i]
        zline.append(z)
        
    return zline, xline, yline
        

def get_palette(cmap="jet"):
    from matplotlib.cm import get_cmap
    return get_cmap(cmap)