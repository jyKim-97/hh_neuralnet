import numpy as np
from tqdm import tqdm
import os
import sys
import subprocess
import pickle as pkl
import traceback
import time

sys.path.append("../include")
import hhtools

sys.path.append("/home/jungyoung/Project/genetic_algorithm/")
import genalg.evolve as evolve

# Parameters
# pE, pI, gE, gI, tlag, nu, w


# ========== Parameters ==========
# control parameter x: p_e (p_{E->?})
# 0: tlag: time lag between neurons

# For faster one
# 1:  d: nu_ext = d * sqrt(p_e) 
# 2:  c: w_e = c * sqrt(0.01) / sqrt(x)
# 3: a1: w_i = a1 * w_e
# 4: b1: p_i = b1 * x

# For slower one
# 5:  d': nu_ext = d' * sqrt(p_e) 
# 6:  c': w_e = c * sqrt(0.01) / sqrt(x)
# 7: a'1: w_i = a'1 * w_e
# 8: b'1: p_i = b'1 * x

# num params: 9

# ========== Check list ==========
# if you change the parameter
#   line  70- : maximal p_ee setting
#   line 190- : parameter writing
#   line 321- : num_params & pmin & pmax

# if you chage host / mpi system
#   line  71  : node names
#   line 107  : command line
#   line 253  : hostfile format
#   line  91  : hostfile pid reading part


# ========== ========== ==========

taur_set = [0.5,  1]
taud_set = [2.5, 10]
fdir = "./data" # -> Need to be fixed to "data"
max_wait_time = 600

CONNECT2SSH = False
USEMULTI = True
CHECKTIME = False
SSHTYPE = "mpich" # ['openmpi', 'mpich']
# check SSHTYPE uses same compiler with "mpifor"

if CONNECT2SSH:
    import paramiko

avail_nodes = ["node2", "node3", "node4", "node5",
            "node6", "node7", "node8", "node9",
            "node12", "node13", "node14", "node15", 
            "node16", "node17", "node18", "node19", "node20",
            "node21", "node22", "node23", "node24", "node25"]
num_cores = 84

avail_cores = 6
num_overlap = 2
num_point = 12 # number of point to calculate (= n * 2 (types))
if USEMULTI:
    num_offspring = len(avail_nodes) * avail_cores * num_overlap // num_point
else:
    num_offspring = num_cores * num_overlap // num_point

num_process = num_offspring // num_overlap

fdir_abs = "/home/jungyoung/Project/hh_neuralnet/two_pop_evolution"

def fobj(args):
    data = args[0]
    job_id = args[1]
    div = num_point//2

    # generate parameters for slow & faster one
    # select the control parameter range
    b = data[4]
    bp = data[8]
    bmax = max([b, bp])
    xs_end = 0.95/bmax
    xs = np.linspace(0.01, xs_end, div)

    # read parameter and allocate simulation parameters
    process_id = job_id % num_process

    if USEMULTI:
        with open("./data/host%d"%(process_id), "r") as fid:
            line = fid.readline()
            if SSHTYPE == "openmpi":
                tmp = line.split(" ")
            elif SSHTYPE == "mpich":
                tmp = line.split(":")
            pid = int(tmp[0][4:])

    for n in range(num_point):
        inh_type = n // div
        x = xs[n%div]

        fname = os.path.join(fdir, "process%d/param%d.txt"%(process_id, n))
        generate_info(x, data, inh_type, fname)

    if USEMULTI:
        com = "/usr/lib64/%s/bin/mpirun -np %d --hostfile %s/data/host%d %s/run_mpi.out %d"%(SSHTYPE, num_point, fdir_abs, process_id, fdir_abs, process_id)
    else:
        com = "/usr/lib64/%s/bin/mpirun -np %d %s/run_mpi.out %d"%(SSHTYPE, num_point, fdir_abs, process_id)
    # print(com)

    if CONNECT2SSH:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
        ssh.connect("10.100.1.%d"%(pid))

    try:
        if CHECKTIME:
            tic = time.time()
        if CONNECT2SSH:
            stdin, stdout, stderr = ssh.exec_command(com, timeout=max_wait_time)
            # exit_status = stdout.channel.recv_exit_status()
        else:
            res = subprocess.run(com, timeout=max_wait_time, shell=True)
        if CHECKTIME:
            print("Job %4d Done, elapsed=%.5f s"%(job_id, time.time()-tic))

        fit_score, chis, cvs, frs = calculate_fitness(process_id)
        save_result(job_id, data, chis, cvs, frs)

    except Exception as e:
        print("Process #%d generated error, job_id: %d"%(process_id, job_id))
        print(traceback.print_exc())
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

    if CONNECT2SSH:
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
    score =  10 * np.average((chis[0] - chis[1])**2)
    score += 5 * np.average(chis[:, :2]**2)
    score += 2.5 * np.average((1-chis[:, -2:])**2)

    # firing rate
    mf = np.average(frs)
    score += 5 * np.average(np.sqrt(((frs[0] - frs[1])/mf)**2))
    if (frs > 10).any():
        score += 50
    else:
        score += 10 * np.average(np.sqrt((frs/mf - 1)**2))
        score += 1 / mf
        # score +=  20 * np.cos(np.pi*mf/20)
        # score += 10 * np.average(np.cos(np.pi * frs) + 1)

    # CV
    score += 2 * np.average((1-cvs)**2)

    return 1/score, chis, cvs, frs
    

def save_result(job_id, data, chis, cvs, frs):
    save_data = {'params': data, 'chis': chis, 'cvs': cvs, 'frs': frs}
    with open(os.path.join(fdir, "result%d.pkl"%(job_id)), "wb") as fid:
        pkl.dump(save_data, fid)


def generate_info(x, data, inh_type, fname):
    tlag = data[0]
    n0 = inh_type * 4 + 1
    
    nu_ext = data[n0] * np.sqrt(x)
    w_e = data[n0+1] * np.sqrt(0.01) / np.sqrt(x)
    w_i = data[n0+2] * w_e
    p_i = data[n0+3] * x

    with open(fname, "w") as fid:
        fid.write("%f,"%(w_e))
        fid.write("%f,"%(w_e))
        fid.write("%f,"%(w_i))
        fid.write("%f,"%(w_i))
        fid.write("%f,"%(x))
        fid.write("%f,"%(x))
        fid.write("%f,"%(p_i))
        fid.write("%f,"%(p_i))
        fid.write("%f,"%(tlag))
        fid.write("%f,"%(taur_set[inh_type]))
        fid.write("%f,"%(taud_set[inh_type]))
        fid.write("%f,"%(nu_ext))


def init_simulation():
    # generate sibling's directory
    for n in range(num_process):
        os.mkdir(os.path.join(fdir, "process%d"%(n)))

    if not USEMULTI and CONNECT2SSH:
        print("If USEMULTI is false, you cannot select CONNECT2SSH True")
        exit(-1)

    if not USEMULTI:
        return
    
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
                if pairs[n][i][1] == 0:
                    break
                
                if SSHTYPE == "openmpi":
                    fid.write("%s cpu=%d\n"%(pairs[n][i][0], pairs[n][i][1]))
                elif SSHTYPE == "mpich":
                    fid.write("%s:%d\n"%(pairs[n][i][0], pairs[n][i][1]))
                else:
                    raise ValueError("You need to select SSH type within ['openmpi', 'mpich']")
                

def custom_initialization(solver):
    # Set Custom init distribution
    num_params = solver.num_params
    normal_param = [[0.0058, 0.0024],
                    [18926.5630, 692.6237],
                    [0.4613, 0.0812],
                    [0.9238, 0.5059],
                    [1.7784, 0.2631],
                    [15.2265, 0.7334],
                    [5.1271, 0.8421],
                    [3.5436, 0.2902],
                    [3.3752, 0.3796],
                    [5.2163, 1.1616],
                    [2.2909, 0.5225],
                    [3.0712, 0.4353],
                    [2.5381, 0.4166]]
    
    for n in range(solver.num_parent):
        for i in range(num_params):
            nstack = 0
            p = pmin[i]-100
            while (p < solver.pmin[i]) or (p > solver.pmax[i]):
                p = np.random.normal(loc=normal_param[i][0],
                                    scale=normal_param[i][1])
                nstack += 1
                
                if nstack > 100:
                    print(p)
                    raise ValueError("iteration for param %d exceed 100"%(i))
            
            solver.param_vec[i, n] = p

    solver.eval_initialization()


if __name__ == "__main__":

    np.random.seed(2000)
    init_simulation()
    
    num_params = 9

    pmin = [0,  5000, 0.1, 0.5, 0.5, 5000, 0.1, 0.5, 0.5]
    pmax = [1, 25000, 10, 20, 5, 25000, 10, 20, 5]

    solver = evolve.EA(num_params, log_dir=fdir, mu=3, num_select=5, num_offspring=num_offspring, num_parent=40, use_multiprocess=True, num_process=num_process, num_overlap=num_overlap)
    solver.set_min_max(pmin, pmax)
    solver.set_object_func(fobj)
    solver.check_setting()

    # solver.load_history("./data")
    # print("load previous result, job_id: %d"%(solver.job_id))
    solver.random_initialization()
    # custom_initialization(solver)

    for n in tqdm(range(1000), ncols=100):
        solver.next_generation()
        solver.print_log()
