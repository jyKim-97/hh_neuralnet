import ctypes

c_hhnet = ctypes.CDLL("./simulation.so")

class simul_info_t(ctypes.Structure):
    _fields_ = [("p_out", (ctypes.c_double*2)*2),
               ("w", (ctypes.c_double*2)*2),
               ("t_lag", ctypes.c_double),
               ("nu_ext_mu", ctypes.c_double),
               ("w_ext_mu", ctypes.c_double)]
    
    
def set_parent_dir(fdir):
    f = ctypes.create_string_buffer(fdir.encode())
    c_hhnet.set_parent_dir(f)
    
    
def set_tmax(_tmax):
    tmax = ctypes.c_double(_tmax)
    c_hhnet.set_tmax(tmax)

def set_taue(tr, td):
    tr_c = ctypes.c_double(tr)
    td_c = ctypes.c_double(td)
    c_hhnet.set_taue(tr_c, td_c)

def set_taui(tr, td):
    tr_c = ctypes.c_double(tr)
    td_c = ctypes.c_double(td)
    c_hhnet.set_taui(tr_c, td_c)

class SimulParams:
    def __init__(self, job_id):
        self.job_id = job_id
        self.p_out = [[None, None], [None, None]]
        self.w = [[None, None], [None, None]]
        self.t_lag = None
        self.nu_ext_mu = None
        self.w_ext_mu = None
    
    def check_params(self):
        params = [self.p_out, self.w, self.t_lag, self.nu_ext_mu, self.w_ext_mu]
        param_names = ["p_out", "W", "t_lag", "nu_ext", "w_ext"]

        for p, pname in zip(params, param_names):
            if isinstance(p, list):
                for arr in p:
                    for x in arr:
                        if x is None:
                            print("%s is None"%(pname))
                    return

            else:
                if p is None:
                    print("%s is None"%(pname))
                    return


def run_simulation(job_id, env: SimulParams):
    c_hhnet.run(job_id, env)


if __name__ == "__main__":

    # set simulation parameter
    info = simul_info_t()
    info.p_out[0][0] = 0.5
    info.p_out[0][1] = 0.5
    info.p_out[1][0] = 0.5
    info.p_out[1][1] = 0.5

    info.w[0][0] = 0.1
    info.w[0][1] = 0.1
    info.w[1][0] = 0.2
    info.w[1][1] = 0.2

    info.t_lag = 0.5
    info.nu_ext_mu = 2000
    info.w_ext_mu = 0.002

    set_taue(0.3, 1)
    set_taui(0.5, 2)

    print("Simulation Start")
    set_parent_dir("./tmp")

    res = c_hhnet.run(1000, info)
