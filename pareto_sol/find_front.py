import numpy as np
from tqdm import tqdm
import os
import sys
import subprocess
import pickle as pkl
from collections import defaultdict, OrderedDict
import traceback
import time
from pymoo.core.problem import Problem
import argparse

sys.path.append("../include")
import hhtools

_debug = True
num_parllel = 12 # Need to be controlled by arg

num_type = 2
num_obj = 2
num_constr = 6

fdir_param = "./params"
fdir_result = "./results"

# fname rule: "./data/result_%d_%d.txt"

# Generate key info
key_types = ("p", "g")
key_names = ("e", "ei", "i0", "i1", "i0e", "i1e")
key_params = ["tlag", "nu_ext"]
for k1 in key_types:
    for k2 in key_names:
        key_params.append("%s%s"%(k1, k2))

key_orders = ("chi", "cv", "frs_m")

taur_set = [0.5,  1]
taud_set = [2.5, 10]
max_wait_time = 1000

# introduce loose constraints
eps_fr = 1
eps_chi = 0.05
th_fr = 8
th_cv = 0.8

# ============= Parameter setting =============
key_names = ["ge", "gei", "gi0e", "gi1e", "gi0", "gi1", "pe", "pei", "pi0e", "pi1e", "pi0", "pi1", "tlag", "nu_ext"]
key_index = OrderedDict({k: n for k, n in zip(key_names, np.arange(len(key_names)))})
bound_param = [[1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 0, 1e3], # lower
               [   1,    1,    1,    1,    1,    1, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1, 2e4]] # upper
# ==============================================


def solve_problem(problem, pop_size=20, n_offspring=10, n_terminal=1, seed=1):
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination
    from pymoo.optimize import minimize


    algorithm = NSGA2(
        pop_size=pop_size,
        n_offspring=n_offspring,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=10),
        mutation=PM(eta=20),
        elimiate_duplicates=True
    )

    termination = get_termination("n_gen", n_terminal)
    res = minimize(problem, algorithm, termination, seed=seed, save_history=True, verbose=True)
    return res


class OptTarget(Problem):
    def __init__(self, bound_lower, bound_upper):
        n_var = len(key_params)

        super().__init__(n_var=n_var, n_obj=num_obj, n_ieq_constr=num_constr,
                         xl=np.array(bound_lower), xu=np.array(bound_upper))
        self.pid = 0 
    
    def _evaluate(self, params, out, *args, **kwargs):
        if len(params.shape) != 2:
            raise ValueError("Unexpected size of parameters, size: ", params.shape)

        # run given params
        pid_set = []
        for n in range(params.shape[0]):
            pid_set.append(self.pid)
            self.pid+=1
        # run_given_params(params, pid_set)

        # evalulate
        score_f = []
        score_g = []
        for pid in pid_set:
            score = calculate_fitness(pid)
            score_f.append(score["F"])
            score_g.append(score["G"])
        
        out["F"] = np.array(score_f)
        out["G"] = np.array(score_g)
        

def run_given_params(params, pid_set):
    if params.shape[0] != len(pid_set):
        raise ValueError("Size does not match")

    # Generate info
    for n, nid in enumerate(pid_set):
        generate_info(params[n], num_type * nid)

    # Run simulation
    if _debug:
        num_take = 1
    else:
        num_take = num_parllel

    start_id = pid_set[0] * num_type
    num_samples = len(pid_set) * num_type
    while num_samples > 0:
        if _debug:
            num_take = 1
            tic = time.time()
            # window
            cmd = "run_simul.out --start_id %d --len %d --tmax %d"%(start_id, num_take, 600)
        else:
            num_take = min([num_samples, num_take])
            cmd = "mpirun -np %d ./run_simul.out --start_id %d --len %d"%(num_take, start_id, num_take)

        res = subprocess.run(cmd, timeout=max_wait_time)

        if _debug:
            print("Job id %4d Done, elapsed=%.3f s"%(start_id, time.time()-tic))

        num_samples -= num_take
        start_id += num_take
    

# Note: Need to do normalization
def calculate_fitness(pid):
    # tag with job_id_0, job_id_1
    data = load_result(pid)

    # objective function (target to optimize, min f)
    f1 = 1-data["chi"][0]
    f2 = data["frs_m"][0]/10

    # constraints, g < 0
    g1 = abs(data["frs_m"][1] - data["frs_m"][0]) - eps_fr
    g2 = abs(data["chi"][1] - data["chi"][0]) - eps_chi
    g3 = th_fr - data["frs_m"][0]
    g4 = th_fr - data["frs_m"][1]
    g5 = th_cv - data["cv"][0]
    g6 = th_cv - data["cv"][1]

    fit = {"F": [f1, f2],
           "G": [g1, g2, g3, g4, g5, g6]}
    
    return fit


def generate_info(param, pid):
    if len(key_names) != len(param):
        raise ValueError("Size does not match betwen 'key_names' and 'param'")

    for n in range(num_type):
        fname = os.path.join(fdir_param, "param_%04d.txt"%(pid+n))
        if os.path.exists(fname):
            raise ValueError("%s exists"%(fname))

        with open(fname, "w") as fid:
            fid.write("%f,"%(param[key_index["ge"]]))
            fid.write("%f,"%(param[key_index["gei"]]))
            fid.write("%f,"%(param[key_index["gi%de"%(n)]]))
            fid.write("%f,"%(param[key_index["gi%d"%(n)]]))
            
            fid.write("%f,"%(param[key_index["pe"]]))
            fid.write("%f,"%(param[key_index["pei"]]))
            fid.write("%f,"%(param[key_index["pi%de"%(n)]]))
            fid.write("%f,"%(param[key_index["pi%d"%(n)]]))

            fid.write("%f,"%(param[key_index["tlag"]]))
            fid.write("%f,"%(taur_set[n]))
            fid.write("%f,"%(taud_set[n]))
            fid.write("%f,"%(param[key_index["nu_ext"]]))


def load_result(job_id):
    # Load data
    data = defaultdict(list)
    for n in range(2):
        fname = "summary_%04d.txt"%(job_id + n)
        summary = hhtools.read_summary(os.path.join(fdir_result, fname))
        print(summary)
        
        for key in ["chi", "cv", "frs_m"]:
            data[key].append(summary[key][0])
    
    for n in range(2):
        if np.isnan(data["chi"][n]):
            data["chi"][n] = 0

    return data


def save_moo(res, problem):
    import pickle as pkl

    n = 0
    fname = "./result_moo_%d.pkl"%(n)
    while os.path.exists(fname):
        n += 1
        fname = "./result_moo_%d.pkl"%(n)

    print("Save to %s"%(fname))
    simul = {"result": res,
             "problem": problem}

    with open(fname, "wb") as f:
        pkl.dump(simul, f)
    

if __name__ == "__main__":

    # Add parser
#     argParser = argparse.ArgumentParser()
# argParser.add_argument()

    problem = OptTarget(bound_param[0], bound_param[1])
    res = solve_problem(problem, pop_size=40, n_offspring=10, n_terminal=5)
    save_moo(res, problem)

