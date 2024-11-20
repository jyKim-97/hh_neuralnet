import numpy as np
import os
# import subprocess
# import xarray as xa
from tqdm import tqdm
from collections import OrderedDict
import argparse

import pickle as pkl

fdir_template = "./data_template"
fdir_out = "./tmp"

nums = [800, 200, 200]

def main():
    ratio_set = [0.02, 0.1]
    template_params = read_template_samples()
    
    param_set = []
    for r in ratio_set:
        for p in template_params:
            # print(p)
            param_set.append(convert_template_params(p, r))
        
    _, num_itr, cid_set = read_controls()
    controls = OrderedDict(cluster_id=cid_set,
                           ratio_set=ratio_set)
    write_control_params(fdir_out, controls, num_itr)
    write_params(fdir_out, param_set)
    
    
    
def convert_template_params(params, ratio, poisson_fr=0.9):
    
    keys = ("EF", "IF", "ES", "IS", "RF", "RS")
    
    seed = params[0]
    wE = [params[1], params[3]]
    wI = [params[2], params[4]]
    pE = [params[5], params[7],  params[13], params[15]]
    pI = [params[9], params[11], params[17], params[19]]
    nu = params[21:23]
    
    prob = dict(
        EF=(pE[0], pE[0], pE[1], pE[1], pE[0], pE[1]),
        IF=(pI[0], pI[0], pI[1], pI[1], pI[0], pI[1]),
        ES=(pE[2], pE[2], pE[3], pE[3], pE[2], pE[3]),
        IS=(pI[2], pI[2], pI[3], pI[3], pI[2], pI[3]),
        RF=(    0,     0,     0,     0,     0,     0),
        RS=(    0,     0,     0,     0,     0,     0)
    )
    
    weight = dict(
        EF=(wE[0], wE[0], wE[0], wE[0], wE[0], wE[0]),
        IF=(wI[0], wI[0], wI[0], wI[0], wI[0], wI[0]),
        ES=(wE[1], wE[1], wE[1], wE[1], wE[1], wE[1]),
        IS=(wI[1], wI[1], wI[1], wI[1], wI[1], wI[1]),
        RF=(    0,     0,     0,     0,     0,     0),
        RS=(    0,     0,     0,     0,     0,     0)
    )
    
    weight_tr = [ratio, ratio]
    
    params = [seed] + nums[:]
    for k in keys:
        params.extend(prob[k])
    for k in keys:
        params.extend(weight[k])
    
    params.extend(weight_tr)
    params.append(poisson_fr)
    params.extend(nu)
    
    return params
    

def read_template_samples():
    
    params_set = []
    with open(os.path.join(fdir_template, "params_to_run.txt"), "r") as fp:
        line = fp.readline()
        line = fp.readline()
        while line:
            params_set.append([])
            for n, x in enumerate(line.split(",")[:-1]):
                if n == 0:
                    params_set[-1].append(int(x))
                else:
                    params_set[-1].append(float(x))

            line = fp.readline()
    
    return params_set        


def read_controls():
    with open(os.path.join(fdir_template, "control_params.txt"), "r") as fp:
        line = fp.readline()
        num, num_itr = [int(x) for x in line.split(",")[:-1]]
        
        line = fp.readline()
        cid_set = [int(float(x)) for x in line.split(":")[1].split(",")[:-1]]
        
    return num, num_itr, cid_set
        
        
def write_control_params(fdir_out, controls, num_itr):
    keys = list(controls.keys())
    
    fname = os.path.join(fdir_out, "control_params.txt")
    with open(fname, "w") as fp:
        for k in keys:
            fp.write("%d,"%(len(controls[k])))
        fp.write("%d,\n"%(num_itr))
        
        for k in keys:
            fp.write("%s:"%(k))
            for val in controls[k]:
                fp.write("%f,"%(val))
            fp.write("\n")
            
    
def write_params(fdir_out, params):
    fname_out = os.path.join(fdir_out, "params_to_run.txt")
    print("Write parametes to %s"%(fname_out))
    with open(fname_out, "w") as fp:
        fp.write("%d\n"%(len(params)))
        for pset in params:
            for n, p in enumerate(pset):
                if n < 4:
                    fp.write("%ld,"%(p))
                else:
                    fp.write("%f,"%(p))
            fp.write("\n")
            
            
if __name__=="__main__":
    main()