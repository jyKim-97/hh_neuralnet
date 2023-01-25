import numpy as np
import evolve
from tqdm import tqdm
import os
import sys
import pickle as pkl

sys.path.append("../include")
import hhtools


# Parameters
# pE, pI, gE, gI, tlag, nu, w

target_chi = 0.
target_firing_rate = 8
num_offspring = 10
num_core = 16

# ========== Parameters ==========
# control parameter x: p_{E->E,I}

# For faster one

# 0: a1: w_{I->E} = a1 * wE
# 1: a2: w_{I->I} = a2 * wE
# 2: b1: p_{I->E} = b1 * x
# 3: b2: p_{I->I} = b2 * x

# For slower one

# 4: a'1: w_{I->E} = a'1 * wE
# 5: a'2: w_{I->I} = a'2 * wE
# 6: b'1: p_{I->E} = b'1 * x
# 7: b'2: p_{I->I} = b'2 * x

# 8: c: wE = c * sqrt(0.01)/sqrt(x)
# 9: d: nu_ext = d * sqrt(pE)
#10: tlag: time lag between neurons

# num params: 11

# ========== ========== ==========


taur_set = [0.5, 1]
taud_set = [2.5, 8]
fdir = "./data" # -> Need to be fixed to "data"

def fobj(args):
    data = args[0]
    job_id = args[1]
    div = num_core//2

    # generate parameters for slow & faster one
    # select the control parameter range
    b1 = data[2]
    b2 = data[3]
    bp1 = data[6]
    bp2 = data[7]
    bmax = max([b1, b2, bp1, bp2])
    xs_end = 0.9/bmax
    xs = np.linspace(0.01, xs_end, div)

    # read parameter and allocate simulation parameters
    offspring_id = job_id % num_offspring
    for n in range(num_core):
        inh_type = n // div
        x = xs[n%div]

        fname = os.path.join(fdir, "offspring%d/param%d.txt"%(offspring_id, n))
        generate_info(x, data, inh_type, fname)

    # run simulation with the code
    com = "mpirun -np %d --hostfile ./data/host%d ./run_mpi.out %d"%(num_core, offspring_id, offspring_id)
    res = os.system(com)

    # # calculate fitness
    fit_score, chis, cvs, frs = calculate_fitness(offspring_id)
    save_result(job_id, data, chis, cvs, frs)

    # clean
    fdir_off = os.path.join(fdir, "offspring%d"%(offspring_id))
    for f in os.listdir(fdir_off):
        os.remove(os.path.join(fdir_off, f))

    return fit_score


def calculate_fitness(offspring_id):
    # read summaries
    chis = [[], []]
    cvs = [[], []]
    frs = [[], []]
    for n in range(num_core):
        inh_type = n // (num_core//2)
        summary = hhtools.read_summary(os.path.join(fdir, "offspring%d/result%d.txt"%(offspring_id, n)))
        chis[inh_type].append(summary["chi"][0])
        cvs[inh_type].append(summary["cv"][0])
        frs[inh_type].append(summary["frs_m"][0])

    chis = np.array(chis)
    cvs  = np.array(cvs)
    frs  = np.array(frs)

    # calculate fitness
    # similarity chi
    score = 100 * np.average((chis[0] - chis[1])**2)
    score += 100 * np.average((chis[:, :2]**2 + (1-chis[:, -2:])**2))

    # firing rate
    score += np.average((frs - target_firing_rate)**2)

    # CV
    score += 30 * np.average((1-cvs)**2)

    return -score, chis, cvs, frs
    

def save_result(job_id, data, chis, cvs, frs):
    save_data = {'params': data, 'chis': chis, 'cvs': cvs, 'frs': frs}
    with open(os.path.join(fdir, "result%d.pkl"%(job_id)), "wb") as fid:
        pkl.dump(save_data, fid)


def generate_info(x, data, inh_type, fname):
    we = data[8] * np.sqrt(0.01)/np.sqrt(x)
    nu_ext = data[9] * np.sqrt(x)
    tlag = data[10]

    n0 = inh_type * 4
    w_ie = data[n0] * we
    w_ii = data[n0+1] * we
    p_ie = data[n0+2] * x
    p_ii = data[n0+3] * x

    with open(fname, "w") as fid:
        fid.write("%f,"%(we))
        fid.write("%f,"%(we))
        fid.write("%f,"%(w_ie))
        fid.write("%f,"%(w_ii))
        fid.write("%f,"%(x))
        fid.write("%f,"%(x))
        fid.write("%f,"%(p_ie))
        fid.write("%f,"%(p_ii))
        fid.write("%f,"%(tlag))
        fid.write("%f,"%(taur_set[inh_type]))
        fid.write("%f,"%(taud_set[inh_type]))
        fid.write("%f,"%(nu_ext))


def init_simulation():
    # generate sibling's directory
    for n in range(num_offspring):
        os.mkdir(os.path.join(fdir, "offspring%d"%(n)))
    
    # generate hostfiles
    nhost = 25
    for n in range(num_offspring):
        with open("./data/host%d"%(n), "w") as fid:
            for i in range(2):
                if nhost == 41: # dead host
                    nhost = nhost+1
                fid.write("node%d cpu=8\n"%(nhost))
                nhost += 1


if __name__ == "__main__":

    np.random.seed(2000)
    init_simulation()
    
    num_params = 11
    pmin = [ 1,  1, 1, 1,  1,  1, 1, 1, 0.01,  7000, 0]
    pmax = [10, 10, 5, 5, 10, 10, 5, 5,  0.2, 20000, 1]

    solver = evolve.EA(num_params, log_dir=fdir, mu=3, num_select=5, num_offspring=num_offspring, num_parent=60, use_multiprocess=False, num_process=num_offspring)
    # solver = evolve.EA(num_params, log_dir="./", mu=3, num_select=5, num_offspring=20, num_parent=60, use_multiprocess=True, num_process=2)
    solver.set_min_max(pmin, pmax)
    solver.set_object_func(fobj)

    solver.check_setting()
    solver.random_initialization()

    solver.reset_job_id()
    for n in tqdm(range(1000)):
        solver.next_generation()
        solver.print_log()
