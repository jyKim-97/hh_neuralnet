import numpy as np
from tqdm import tqdm
import os
import sys
import paramiko
import subprocess
import pickle as pkl
from collections import defaultdict, OrderedDict
import traceback
import time
from pymoo.core.problem import Problem
import functools

from pymoo.util.display.column import Column
from pymoo.util.display.output import Output
from pymoo.util.display.single import MinimumConstraintViolation, AverageConstraintViolation
from pymoo.util.display.multi import MultiObjectiveOutput
import argparse

import signal

sys.path.append("../include")
import hhtools

_debug = False
_single_core = False
num_parllel = 20 # Need to be controlled by arg

num_type = 2
num_obj = 2
num_constr = 6

fdir = "/home/jungyoung/Project/hh_neuralnet/pareto_sol"
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
th_fr = 10
th_cv = 0.6

# ============= Parameter setting =============
key_names = ["ge", "gei", "gi0e", "gi1e", "gi0", "gi1", "pe", "pei", "pi0e", "pi1e", "pi0", "pi1", "tlag", "nu_ext"]
key_index = OrderedDict({k: n for k, n in zip(key_names, np.arange(len(key_names)))})
bound_param = [[1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 5e-2, 0, 1e3], # lower
               [ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 1, 2e4]] # upper
# ==============================================


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize


class OptTarget(Problem):
    def __init__(self, bound_lower, bound_upper):
        n_var = len(key_params)
        n_ieq_constr = num_constr if not _debug else 0
        super().__init__(n_var=n_var, n_obj=num_obj, n_ieq_constr=n_ieq_constr,
                         xl=np.array(bound_lower), xu=np.array(bound_upper))
        self.pid = 0 
    
    def _evaluate(self, params, out, *args, **kwargs):
        if len(params.shape) != 2:
            raise ValueError("Unexpected size of parameters, size: ", params.shape)

        # run given params
        pid_set = []
        for _ in range(params.shape[0]):
            pid_set.append(self.pid)
            self.pid += num_type
        
        run_given_params(params, pid_set)

        if _debug:
            pop_size = params.shape[0]
            out["F"] = np.random.uniform(size=(pop_size, 2)) * 5
        else:
            score_f = []
            score_g = []
            for pid in pid_set:
                score = calculate_fitness(pid)
                score_f.append(score["F"])
                score_g.append(score["G"])
            
            out["F"] = np.array(score_f)
            out["G"] = np.array(score_g)



def solver_handler(func):
    @functools.wraps(func)
    def wrapper(self, problem, *args, **kwargs):

        def error_handler(sig, frame):
            print("\nTermination...\n")
            save_moo(None, problem, self.algorithm)
            exit(1)
            
        signal.signal(signal.SIGINT, error_handler)

        try:
            return func(self, problem, *args, **kwargs)
        
        except Exception as e:
            print("Exception occcured")
            traceback.print_exception(e)
            error_handler(None, None)
    
    return wrapper


class Solver:
    def __init__(self, pop_size=40, n_offspring=10):

        self.algorithm = NSGA2(
            pop_size=pop_size,
            n_offspring=n_offspring,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.5, eta=20),
            mutation=PM(eta=20),
            elimiate_duplicates=True
        )
        self.problem = None

    @solver_handler
    def solve(self, problem, n_terminal=1, seed=100):
        # signal.signal(signal.SIGINT, self._handler)

        self.problem = problem
        termination = get_termination("n_gen", n_terminal)
        res = minimize(self.problem, self.algorithm, termination, seed=seed,
                    save_history=True, verbose=True, output=PrintLog(),
                    return_least_infeasible=True, copy_algorithm=False)
        save_moo(res, self.problem, self.algorithm)
        return res


class PrintLog(MultiObjectiveOutput):
    def __init__(self):
        super().__init__()
        # self.n_nds = Column("n_nds", width=10)
        self.chi0 = Column("mean_chi0", width=10)
        self.dchi = Column("delta_chi", width=10)
        self.fr0  = Column("mean_fr0", width=10)
        self.fr1  = Column("mean_fr1", width=10)
        self.cvs  = Column("mean_cv", width=10)
        self.columns += [self.chi0, self.dchi, self.fr0, self.fr1, self.cvs]
    
    def update(self, algorithm):
        super().update(algorithm)
        chi0 = np.average(1 - algorithm.pop.get("F")[:, 0])
        if not _debug:
            dchi = np.average(algorithm.pop.get("G")[:, 1] + eps_chi)
            fr0 = th_fr * (1 + np.average(algorithm.pop.get("G")[:, 2]))
            fr1 = th_fr * (1 + np.average(algorithm.pop.get("G")[:, 1]))
            cv_tmp = (np.average(algorithm.pop.get("G")[:, 4]) + np.average(algorithm.pop.get("G")[:, 5]))/2
            cv = th_cv * (1 - cv_tmp)

        self.chi0.set(chi0)
        if not _debug:
            self.dchi.set(dchi)
            self.fr0.set(fr0)
            self.fr1.set(fr1)
            self.cvs.set(cv)
        time.sleep(0.01)


def run_single(cmd):
    res = subprocess.run(cmd, timeout=max_wait_time, shell=True)
    return res


def run_with_mpi(cmd):
    if _debug:
        return

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy)
    ssh.connect("10.100.1.75")

    try:
        stdin, stdout, stderr = ssh.exec_command(cmd, timeout=max_wait_time)
        result = stderr.read().decode('ascii').strip("\n")
        
    except Exception as e:
        print("Process generated error")
        print(traceback.print_exc())

    ssh.close()
        

def run_given_params(params, pid_set):
    if params.shape[0] != len(pid_set):
        raise ValueError("Size does not match")

    # Generate info
    for n, pid in enumerate(pid_set):
        generate_info(params[n], pid)

    # Run simulation
    if _debug:
        num_take = 1
    else:
        num_take = num_parllel

    start_id = pid_set[0]
    num_samples = len(pid_set) * num_type
    while num_samples > 0:
        if _single_core:
            num_take = 1
            tic = time.time()
            cmd = "run_simul.out --start_id %d --len %d --tmax %d"%(start_id, num_take, 600) # Notion: run in Window system
            run_single(cmd)
            print("Job id %4d Done, elapsed=%.3f s"%(start_id, time.time()-tic))
        else:
            num_take = min([num_samples, num_take])
            cmd = "/usr/lib64/mpich/bin/mpirun -np %d --hostfile %s/host %s/run_simul.out --start_id %d --len %d"%(num_take+1, fdir, fdir, start_id, num_take)
            run_with_mpi(cmd)

        num_samples -= num_take
        start_id += num_take
    

def calculate_fitness(pid):
    # tag with job_id_0, job_id_1
    data = load_result(pid)

    # objective function (target to optimize, min f)
    f0 = data["chi"][0]
    f1 = th_fr - data["frs_m"][0]

    # constraints, g < 0
    g0 = abs(data["frs_m"][1] - data["frs_m"][0]) - eps_fr
    g1 = abs(data["chi"][1] - data["chi"][0]) - eps_chi
    g2 = (data["frs_m"][0] - th_fr)/th_fr
    g3 = (data["frs_m"][1] - th_fr)/th_fr
    g4 = (th_cv - data["cv"][0])/th_cv
    g5 = (th_cv - data["cv"][1])/th_cv

    fit = {"F": [f0, f1],
           "G": [g0, g1, g2, g3, g4, g5]}
    if _debug:
        fit = {"F": [f0, f1]}
    
    return fit


def generate_info(param, pid):
    if len(key_names) != len(param):
        raise ValueError("Size does not match betwen 'key_names' and 'param'")

    if _debug:
        return

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


def load_result(pid):
    key_names = ["chi", "cv", "frs_m"]
    # Load data
    data = defaultdict(list)
    for n in range(2):
        fname = "summary_%04d.txt"%(pid + n)
        summary = hhtools.read_summary(os.path.join(fdir_result, fname))

        if summary == -1:
            data = {key: [0, 0] for key in key_names}
            break
        
        for key in key_names:
            data[key].append(summary[key][0])
    
    for n in range(2):
        for key in key_names:
            if np.isnan(data[key][n]):
                data[key][n] = 0

    return data


def save_moo(res, problem, algorithm):
    import pickle as pkl

    n = 0
    fname = "./result_moo_%d.pkl"%(n)
    while os.path.exists(fname):
        n += 1
        fname = "./result_moo_%d.pkl"%(n)

    print("Save to %s"%(fname))
    simul = {"result": res,
             "problem": problem,
             "algorithm": algorithm}

    with open(fname, "wb") as f:
        pkl.dump(simul, f)


def main():
    global _debug, num_parllel

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_core", type=int, default=20)
    parser.add_argument("--num_parent", type=int, default=40)
    parser.add_argument("--num_offspring", type=int, default=10)
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--debug_mod", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    num_parllel = args.num_core
    num_parent = args.num_parent
    num_offspring = args.num_offspring
    num_terminal = args.num_epoch
    seed = args.seed
    _debug = args.debug_mod

    problem = OptTarget(bound_param[0], bound_param[1])
    solver = Solver(pop_size=num_parent, n_offspring=num_offspring)
    res = solver.solve(problem, n_terminal=num_terminal, seed=seed)
    print(res.F)


if __name__ == "__main__":
    main()

    # # Add parser
    # argParser = argparse.ArgumentParser()
    # argParser.add_argument()

    # problem = OptTarget(bound_param[0], bound_param[1])
    # solver = Solver()
    # res = solver.solve(problem, n_terminal=200, seed=100)
