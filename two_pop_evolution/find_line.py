import numpy as np
from tqdm import tqdm
import os
import sys
import subprocess
import pickle as pkl
import paramiko

sys.path.append("../include")
import hhtools

sys.path.append("/home/jungyoung/Project/genetic_algorithm/")
import genalg.evolve as evolve

# Parameters
# pE, pI, gE, gI, tlag, nu, w


# ========== Parameters ==========
# control parameter x: p_ee (p_{E->E})

#  0: w_ee = c * sqrt(0.01) / sqrt(x)
#  1: d: nu_ext = d * sqrt(p_ee)
#  2: tlag: time lag between neurons
#  3: a1: w_ei = a1 * w_ee (w_{E->I})
#  4: b1: p_ei = b1 * x

# For faster one
#  5: a2: w_ie = a2 * w_ee
#  6: a3: w_ii = a3 * w_ee
#  7: b2: p_ie = b2 * x
#  8: b3: p_ii = b3 * x

# For slower one
#  9: a'2: w_ie = a'2 * w_ee
# 10: a'3: w_ii = a'3 * w_ee
# 11: b'2: p_ie = b'2 * x
# 12: b'3: p_ii = b'3 * x

# num params: 13

# ========== ========== ==========

taur_set = [0.5, 1]
taud_set = [2.5, 9]
fdir = "./data" # -> Need to be fixed to "data"
max_wait_time = 600

avail_nodes = ["rode9", "rode10", "rode11", "rode13", "rode14", "rode15"]
avail_cores = 10
num_overlap = 4

num_point = 12
num_offspring = len(avail_nodes) * avail_cores * num_overlap // num_point
num_process = num_offspring // num_overlap

fdir_abs = "/home/jungyoung/Project/hh_neuralnet/two_pop_evolution"

def fobj(args):
    data = args[0]
    job_id = args[1]
    div = num_point//2

    # generate parameters for slow & faster one
    # select the control parameter range
    b1 = data[1]
    bp1 = data[3]
    bmax = max([b1, bp1])
    xs_end = 0.9/bmax
    xs = np.linspace(0.01, xs_end, div)

    # read parameter and allocate simulation parameters
    process_id = job_id % num_process

    with open("./data/host%d"%(process_id), "r") as fid:
        line = fid.readline()
        tmp = line.split(":")
        pid = int(tmp[0][4:])

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect("10.100.1.%d"%(60+pid))
    ssh.exec_command("module load mpi/mpich")
    # stdin, stdout, stderr = ssh.exec_command("cd %s"%(fdir_abs))
    # stdin, stdout, stderr = ssh.exec_command("pwd")

    for n in range(num_point):
        inh_type = n // div
        x = xs[n%div]

        fname = os.path.join(fdir, "process%d/param%d.txt"%(process_id, n))
        generate_info(x, data, inh_type, fname)

    
    # run simulation with the code
    # com = "/usr/lib64/mpich/bin/mpirun -np %d --hostfile ./data/host%d ./run_mpi.out %d"%(num_point, process_id, process_id)
    # d = "/home/jungyoung/Project/hh_neuralnet/two_pop_evolution"
    com = "/usr/lib64/mpich/bin/mpirun -np %d --hostfile %s/data/host%d %s/run_mpi.out %d"%(num_point, fdir_abs, process_id, fdir_abs, process_id)

    try:
        # res = subprocess.run(com, timeout=max_wait_time, shell=True)
        stdin, stdout, stderr = ssh.exec_command(com, timeout=max_wait_time)
        exit_status = stdout.channel.recv_exit_status()
        # # calculate fitness
        fit_score, chis, cvs, frs = calculate_fitness(process_id)
        save_result(job_id, data, chis, cvs, frs)
    except Exception as e:
        print("Process #%d generated error, job_id: %d"%(process_id, job_id))
        print(e)
        fit_score = np.nan

    # stdin, stdout, stderr = ssh.exec_command(com, timeout=max_wait_time)
    # exit_status = stdout.channel.recv_exit_status()
    # # # calculate fitness
    # fit_score, chis, cvs, frs = calculate_fitness(process_id)
    # save_result(job_id, data, chis, cvs, frs)

    # clean
    fdir_off = os.path.join(fdir, "process%d"%(process_id))
    for f in os.listdir(fdir_off):
        os.remove(os.path.join(fdir_off, f))

    ssh.close()

    return fit_score


def calculate_fitness(process_id):
    # read summaries
    chis = [[], []]
    cvs = [[], []]
    frs = [[], []]
    for n in range(num_point):
        inh_type = n // (num_point//2)
        summary = hhtools.read_summary(os.path.join(fdir, "process%d/result%d.txt"%(process_id, n)))
        chis[inh_type].append(summary["chi"][0])
        cvs[inh_type].append(summary["cv"][0])
        frs[inh_type].append(summary["frs_m"][0])

    chis = np.array(chis)
    cvs  = np.array(cvs)
    frs  = np.array(frs)

    # calculate fitness
    # similarity chi
    score = 100 * np.average((chis[0] - chis[1])**2)
    score += 50 * np.average(chis[:, :2]**2)
    score +=  5 * np.average((1-chis[:, -2:])**2)

    # firing rate
    score += 50 * np.average((frs[0] - frs[1])**2)
    if (frs > 10).any():
        score += 20
    else:
        score += 10 * np.average(np.cos(np.pi * frs) + 1)

    # CV
    score += 5 * np.average((1-cvs)**2)

    return -score, chis, cvs, frs
    

def save_result(job_id, data, chis, cvs, frs):
    save_data = {'params': data, 'chis': chis, 'cvs': cvs, 'frs': frs}
    with open(os.path.join(fdir, "result%d.pkl"%(job_id)), "wb") as fid:
        pkl.dump(save_data, fid)


def generate_info(x, data, inh_type, fname):
    w_ee = data[0] * np.sqrt(0.01)/np.sqrt(x)
    nu_ext = data[1] * np.sqrt(x)
    tlag = data[2]
    w_ei = data[3] * w_ee
    p_ei = data[4] * x

    n0 = inh_type*4 + 5
    w_ie = data[n0]   * w_ee
    w_ii = data[n0+1] * w_ee
    p_ie = data[n0+2] * x
    p_ii = data[n0+3] * x

    with open(fname, "w") as fid:
        fid.write("%f,"%(w_ee))
        fid.write("%f,"%(w_ei))
        fid.write("%f,"%(w_ie))
        fid.write("%f,"%(w_ii))
        fid.write("%f,"%(x))
        fid.write("%f,"%(p_ei))
        fid.write("%f,"%(p_ie))
        fid.write("%f,"%(p_ii))
        fid.write("%f,"%(tlag))
        fid.write("%f,"%(taur_set[inh_type]))
        fid.write("%f,"%(taud_set[inh_type]))
        fid.write("%f,"%(nu_ext))


def init_simulation():
    # generate sibling's directory
    for n in range(num_process):
        os.mkdir(os.path.join(fdir, "process%d"%(n)))
    
    # generate hostfiles
    num_spawn = num_offspring * num_point
    num_avail = len(avail_nodes) * avail_cores * num_overlap

    if num_spawn != num_avail:
        raise ValueError("The number of cores to spawn are not same with expected (%d, %d)"%(num_spawn, num_avail))

    # allocate the cores
    num = num_spawn // num_overlap

    nid = 0
    num_need = num_point

    ncore = 0
    num_avail_core = avail_cores
    
    pairs = [[[avail_nodes[ncore], 0]]]
    for n in range(num):

        if num_avail_core == 0:
            ncore += 1
            num_avail_core = avail_cores
            pairs[nid].append([avail_nodes[ncore], 0])
            
        if num_need == 0:
            nid += 1
            num_need = num_point
            pairs.append([[avail_nodes[ncore], 0]])

        pairs[nid][-1][1] += 1

        num_avail_core -= 1
        num_need -= 1

    for n in range(num_process):
        with open("./data/host%d"%(n), "w") as fid:
            for i in range(len(pairs[n])):
                fid.write("%s:%d\n"%(pairs[n][i][0], pairs[n][i][1]))
    


if __name__ == "__main__":

    np.random.seed(2000)
    init_simulation()
    
    num_params = 7
    pmin = [ 1, 1,  1, 1, 0.01,  7000, 0]
    pmax = [10, 5, 10, 5,  0.2, 20000, 1]

    solver = evolve.EA(num_params, log_dir=fdir, mu=3, num_select=5, num_offspring=num_offspring, num_parent=60, use_multiprocess=True, num_process=num_process, num_overlap=num_overlap)
    solver.set_min_max(pmin, pmax)
    solver.set_object_func(fobj)

    solver.check_setting()

    # solver.load_history("./data")
    # print("load previous result, job_id: %d"%(solver.job_id))
    solver.random_initialization()
    solver.reset_job_id()

    for n in tqdm(range(1000), ncols=100):
        solver.next_generation()
        solver.print_log()
