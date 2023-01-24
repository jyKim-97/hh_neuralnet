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
    info.p_out[0][1] = data[1]
    info.p_out[1][0] = data[2]
    info.p_out[1][1] = data[3]

    info.w[0][0] = data[4]
    info.w[0][1] = data[5]
    info.w[1][0] = data[6]
    info.w[1][1] = data[7]

    info.t_lag = data[8]
    info.nu_ext_mu = data[9]
    info.w_ext_mu = data[10]

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
    # score += 10*((0.5-info.w[0][0])**2 + (0.5-info.w[1][1])**2)
    # score += 200*(summary["chi"][0] - target_chi)**2
    score += 50 * (summary["cv"][0] - 1) ** 2
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
fdir = "./opt_fast_all_seperate"


if __name__ == "__main__":

    check_directory()

    # the number of parameters: 7
    num_params = 11

    np.random.seed(2000)
    pmin = [0.005, 0.005, 0.001, 0.001, 0.01, 0.01, 0.01, 0.01, 0,  700,     0]
    # pmax = [  0.2,   0.2,   0.2,   0.2,  0.5,  0.5,  0.5,  0.5, 1, 2500, 0.005]
    pmax = [  0.9,   0.9,   0.9,   0.9,  0.5,  0.5,  0.5,  0.5, 1, 20000, 0.005]

    csimul.set_tmax(2000)
    csimul.set_parent_dir(fdir)
    csimul.change_taui(taur_i, taud_i)
    csimul.change_teq(500)

    solver = evolve.EA(num_params, log_dir=fdir, mu=3, num_select=5, num_offspring=20, num_parent=100, use_multiprocess=True, num_process=20)
    solver.set_min_max(pmin, pmax)
    solver.set_object_func(fobj)

    solver.check_setting()
    solver.random_initialization()
    clean_directory()

    solver.reset_job_id()
    for n in tqdm(range(1000)):
        solver.next_generation()
        solver.print_log()
