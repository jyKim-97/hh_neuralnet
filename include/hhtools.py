# from re import S
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pkl


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
        n0 += l
    return vlfps, fs


def load_mua(fname):
    if not os.path.isfile(fname):
        return
    
    with open(fname, "rb") as fid:
        data = np.fromfile(fid, dtype=np.float32)
    # dtaa[0]: dt
    # data[1]: st
    return np.reshape(data[2:], (2, -1))
    


def load_network(fname):
    """
    Read in-degree network from a file.

    Parameters:
    fname (str): The name of the file containing the network data.

    Returns:
    tuple: (ntk_in, ntk_win)
        ntk_in (list of lists): Incoming connections for each node.
        ntk_win (list of lists): Weights of incoming connections for each node.
    """
    
    with open(fname, "r") as fid:
        line = fid.readline()
        
        # read number of cells
        tmp = line.split("=")
        N = int(tmp[1][:-2])
        ntk_in = [[] for _ in range(N)]
        ntk_win = [[] for _ in range(N)]
        
        # read connectivity
        line = fid.readline()
        while line:
            tmp = line.split("<-")
            npost = int(tmp[0])
            npre = int(tmp[1].split(",")[0])
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


def draw_spk(step_spk, dt=0.01, sequence=None, xl=None, color_ranges=None, colors=None, ms=1):
    if color_ranges is not None:
        if colors is None:
            print("Type the colors")
            return
        else:
            if len(color_ranges) != len(colors):
                print("The length of color and color range does not match")
                return

    N = len(step_spk)
    if sequence is None:
        sequence = np.arange(N)
    else:
        if len(sequence) >= N:
            raise ValueError("Length of sequence (%d) exceeds N (%d)"%(len(sequence), N))

    cid = 0
    for n, nid in enumerate(sequence):
        t_spk = np.array(step_spk[nid]) * dt
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
    plt.ylim([0, len(sequence)])
    
    
def draw_spk_single(step_spk, dt=0.01, y0=0, **plot_opt):
    x, y = [], []
    for n, nstep in enumerate(step_spk):
        x.extend([nstep*dt, nstep*dt, np.nan])
        y.extend([y0, y0+1, np.nan])
    
    plt.plot(x, y, **plot_opt)

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
#     norm = np.concatenate((np.arange(0, num_lag+1), np.arange(num_lag,0,-1)))
#     norm = len(x_cp) - norm
#     # norm = np.concatenate((np.arange(num_lag+1), np.arange(l,0,-1)+1))
#     xcorr = xcorr/norm
#     lags = np.arange(-num_lag, num_lag+1) * dt
    
#     return xcorr, lags


# Source code for loadding summarys
class SummaryLoader:
    def __init__(self, fdir, load_only_control=False, load_cache=True):
        # read control param infos
        self.fdir = fdir
        self._load_controls()
        if not load_only_control:
            if load_cache:
                self._load_cache()
            else:
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
                
            self.num_total = 1
            for n in self.num_controls:
                self.num_total *= n
                
    def _load_cache(self):
        fcache = os.path.join(self.fdir, "summary.pkl")
        if os.path.exists(fcache):
            with open(fcache, "rb") as fp:
                self.summary = pkl.load(fp)
                return
        else:
            print("Cache file does not exist, read data")
            self._read_data()

    def _read_data(self):
        
        fnames = [f for f in os.listdir(self.fdir) if "id" in f and "result" in f and "monitor" not in f]
        self.num_overlap = self.check_overlap(fnames)

        nums = len(fnames)
        nums_expect = 1
        for n in self.num_controls:
            nums_expect *= n
        self.num_total = nums_expect
        
        if nums != nums_expect * self.num_overlap:
            print("Expected number of # results and exact file number are different!: %d/%d"%(nums, nums_expect*self.num_overlap))
        
        self.summary = {}
        self.load_success = np.ones(nums_expect)
        var_names = ["chi", "cv", "frs_m", "frs_s"]
        for k in var_names:
            self.summary[k] = []

        for n in range(nums_expect):

            for i in range(self.num_overlap):
                if self.num_overlap == 1:
                    fname = os.path.join(self.fdir, "id%06d_result.txt"%(n))    
                else:
                    fname = os.path.join(self.fdir, "id%06d_%02d_result.txt"%(n, i))
                summary_single = read_summary(fname)

                if summary_single == -1:
                    self.load_success[n] = 0
                    for k in var_names:
                        if len(self.summary[k]) > 0:
                            val_prev = self.summary[k][-1]
                            self.summary[k].append(np.zeros_like(val_prev) * np.nan)
                        else:
                            self.summary[k].append([])
                
                else:
                    for k in var_names:
                        self.summary[k].append(summary_single[k])
        
        # reshape
        for k in var_names:
            shape = np.shape(self.summary[k])[1:]
            num_tmp = self.num_controls.copy()
            if self.num_overlap != 1:
                num_tmp[-1] *= self.num_overlap # last index corresponds to # of samples
            new_shape = list(num_tmp)+list(shape)
            self.summary[k] = np.reshape(self.summary[k], new_shape)

        # save cache
        self._save_cache()
    
    def _save_cache(self):
        with open(os.path.join(self.fdir, "summary.pkl"), "wb") as fp:
            pkl.dump(self.summary, fp)

    def check_overlap(self, fnames):
        # check overlap based on the first element
        f0 = fnames[0]
        frags = f0.split("_")
        if len(frags) == 2: # no overlap
            return 1
        
        prefix = frags[0]
        fsub = [f for f in fnames if prefix in f]
        return len(fsub)

    def get_id(self, *nid):
        return get_id(self.num_controls, *nid)
    
    def load_detail(self, *nid, load_now=True):
        if len(nid) == 1:
            n = nid[0]
        else:
            n = get_id(self.num_controls, *nid)
        
        tag = os.path.join(self.fdir, "id%06d"%(n))
        obj = SampleLoader(tag, nid)
        
        if load_now:
            return obj.load()
        else:
            return obj

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

    def export_summary(self):
        import pickle as pkl

        fdir = self.fdir
        if fdir[-1] == '/':
            fdir = fdir[:-1]
        
        # need to error log
        f = fdir + ".pkl"
        print("Save summary to %s"%(f))
        with open(f, "wb") as fid:
            pkl.dump(self, fid)


class SampleLoader:
    def __init__(self, tag, data_id):
        self.tag = tag
        self.data_id = data_id[0] if len(data_id) == 1 else data_id
        
    def load(self):
        data = {}
        data["step_spk"], _ = load_spk(self.tag+"_spk.dat")
        data["vlfp"], fs = load_vlfp(self.tag+"_lfp.dat")
        data["mua"] = load_mua(self.tag+"_mua.dat")
        data["ts"] = np.arange(len(data["vlfp"][0])) / fs
        data["nid"] = self.data_id
        data["prefix"] = self.tag
        
        if os.path.exists(self.tag+"_info.txt"):
            with open(self.tag+"_info.txt", "r") as fid:
                data["info"] = fid.readlines()
        else:
            data["info"] = None
        
        return data


def read_info(info_string):
    import re

    def isfloat(s):
        try:
            _ = float(s)
            return True
        except:
            return False

    key = None
    key_p = None
    flag_type = 0
    info = dict()
    for l in info_string:
        l_split = re.split(r":|,", l[:-1])

        # remove if there is any "" or " "
        for n in range(len(l_split)-1, 0, -1):
            l_split[n] = l_split[n].strip()
            if l_split[n] == "":
                l_split.pop(n)

        num_split = len(l_split)
        # get key
        tmp_key = ''
        nl = 0
        flag_semi = False
        while (nl < num_split) and not isfloat(l_split[nl]):
            if flag_semi:
                s = "; " + l_split[nl]
            else:
                s = l_split[nl]
            tmp_key += s
            flag_semi = True
            nl += 1

        if tmp_key != "": # key is given
            key = tmp_key
            if nl < num_split:
                if flag_type != 1:
                    info[key] = []
                    flag_type = 0
                else:
                    info[key_p][key] = []
            else:
                flag_type = 1
                key_p = key
                info[key_p] = {}
                continue
        else: # 2D case
            if flag_type == 1:
                info[key] = [[]]
            else:
                info[key].append([])
            flag_type = 2

        for i in range(nl, num_split):
            if flag_type == 0:
                info[key].append(float(l_split[i]))
            elif flag_type == 1:
                info[key_p][key].append(float(l_split[i]))
            else:
                info[key][-1].append(float(l_split[i]))

        if flag_type == 1 and len(info[key_p][key]) == 1:
            info[key_p][key] = info[key_p][key][0]
        elif flag_type == 0 and len(info[key]) == 1:
            info[key] = info[key][0]
          
    return info


def get_id(num_xs, *nid):
    if len(nid) != len(num_xs):
        raise ValueError("The number of arguments does not match to expected #")
    num_tag = 0
    stack = 1
    for n in range(len(nid)-1, -1, -1):
        if nid[n] >= num_xs[n]:
            raise ValueError("Wrong index number typed, size", num_xs, "typed", nid)
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
        # summary["spike_sync"] = []
        # line = fid.readline()
        # while line:
        #     summary["spike_sync"].append([float(x) for x in line.split(",")[:-1]])
        #     line = fid.readline()
    
    for k in summary.keys():
        summary[k] = np.array(summary[k])
    
    return summary


def imshow_xy(im, x=None, y=None, **kwargs):
    extent = []
    if x is not None:
        if len(x) != im.shape[1]:
            raise ValueError("len(x)=%d and im.shape[1]=%d do not match"%(len(x), im.shape[1]))

        dx = (x[1] - x[0])/2
        extent = [x[0]-dx, x[-1]+dx]
    else:
        extent = [-0.5, np.shape(im)[1]-0.5]
    if y is not None:
        if len(y) != im.shape[0]:
            raise ValueError("len(y)=%d and im.shape[0]=%d do not match"%(len(y), im.shape[0]))

        dy = (y[1] - y[0])/2
        extent.extend([y[0]-dy, y[-1]+dy])
    else:
        extent.extend([-0.5, np.shape(im)[0]-0.5])
    
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


# include these functions to object
def export_summary(obj:SummaryLoader):
    import pickle as pkl

    fdir = obj.fdir
    if fdir[-1] == '/':
        fdir = fdir[:-1]
    
    # need to error log
    f = fdir + ".pkl"
    print("Save summary to %s"%(f))
    with open(f, "wb") as fid:
        pkl.dump(obj, fid)
    

# def load_summary(f):
#     import pickle as pkl

#     # need to error log
#     with open(f, "rb") as fid:
#         obj = pkl.load(fid)
    
#     return obj

