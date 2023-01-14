import ctypes


flib = ctypes.CDLL("./simulation.so")


class simul_info_t(ctypes.Structure):
    _fields_ = [("p_out", (ctypes.c_double*2)*2),
               ("w", (ctypes.c_double*2)*2),
               ("t_lag", ctypes.c_double),
               ("nu_ext_mu", ctypes.c_double),
               ("w_ext_mu", ctypes.c_double)]
    
    
def set_parent_dir(fdir):
    f = ctypes.create_string_buffer(fdir.encode())
    flib.set_parent_dir(f)
    
    
def set_tmax(_tmax):
    tmax = ctypes.c_double(_tmax)
    flib.set_tmax(tmax)


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
    flib.run(job_id, env)
