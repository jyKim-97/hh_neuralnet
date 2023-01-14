import numpy as np
import csimul
import evolve
from tqdm import tqdm
import os

import sys
sys.path.append("../include")
import hhtools

# Parameters
# pE, pI, gE, gI, tlag, nu, w

target_chi = 0.
target_firing_rate = 8

def fobj(args):
    data = args[0]
    info = csimul.simul_info_t()
    info.p_out[0][0] = data[0]
    info.p_out[0][1] = data[0]
    info.p_out[1][0] = data[1]
    info.p_out[1][1] = data[1]

    info.w[0][0] = data[2]
    info.w[0][1] = data[2]
    info.w[1][0] = data[3]
    info.w[1][1] = data[3]

    info.t_lag = data[4]
    info.nu_ext_mu = data[5]
    info.w_ext_mu = data[6]

    job_id = args[1]
    res = csimul.c_hhnet.run(job_id, info)

    # get fitness
    score = calculate_fitness(info, job_id)
    # score = np.random.randn()
    return score
    

def calculate_fitness(info, job_id):
    # To make asynchronous state, give high fitness to high g
    # target firing rate
    fname = os.path.join(fdir, "id%06d_result.txt"%(job_id))
    summary = hhtools.read_summary(fname)
    
    score = 0
    score += 2 *((0.5-info.w[0][0])**2 + (0.5-info.w[1][1])**2)
    score += 10*(summary["chi"][0] - target_chi)**2
    score += (summary["frs_m"][0] - target_firing_rate)**2
    return -score


def check_directory():
    if len(os.listdir(fdir)) > 0:
        raise AttributeError("Some file exist in target directory")


def clean_directory():
    for f in os.listdir(fdir):
        os.remove(os.path.join(fdir, f))


taur_i = 0.5
taud_i = 2
fdir = "./tmp"


if __name__ == "__main__":

    check_directory()

    # the number of parameters: 7
    num_params = 7
    
    np.random.seed(1000)
    pmin = [0.01, 0.01, 0.01, 0.01, 0,  700,     0]
    pmax = [ 0.2,  0.2,  0.5,  0.5, 1, 4000, 0.005]

    csimul.set_tmax(500)
    csimul.set_parent_dir(fdir)
    csimul.change_taui(taur_i, taud_i)
    csimul.change_teq(0)

    solver = evolve.EA(num_params, log_dir=fdir, mu=3, num_select=4, num_offspring=16, num_parent=40, use_multiprocess=True, num_process=4)
    solver.set_min_max(pmin, pmax)
    solver.set_object_func(fobj)

    solver.check_setting()
    solver.random_initialization()
    clean_directory()
    
    solver.reset_job_id()
    for n in tqdm(range(30)):
        solver.next_generation()
        solver.print_log(n)
