import numpy as np
import os
import sys
import xarray as xa

sys.path.append('/home/jungyoung/Project/hh_neuralnet/include/')
import hhtools
import hhsignal
import importlib
from tqdm import tqdm
# other options
np.set_printoptions(suppress=True)


fdirs = ("./data/pe_nu_fast", "./data/pe_nu_slow")
fout =  ("./postdata/pe_nu_fast.nc", "./postdata/pe_nu_slow.nc")

wbin = 1
srate = 2000
key_vars = ("chi", "cv", "fr", "fnet", "pnet")

def main():
    # construct summary
    np.random.seed(42)
    
    summary_obj = hhtools.SummaryLoader(fdirs[0])
    dataarray = export_summary(summary_obj)
    dataarray.to_netcdf(fout[0])
    print("Exported:", fout[0])
    
    summary_obj = hhtools.SummaryLoader(fdirs[1])
    dataarray = export_summary(summary_obj)
    dataarray.to_netcdf(fout[1])
    print("Exported:", fout[1])
    

def export_summary(summary_obj):
    data_dict = construct_summary(summary_obj)
    data = np.stack([data_dict[k] for k in key_vars], axis=2)
    
    dataarray = xa.DataArray(
        data=data,
        dims=["pe", "nu", "vars"],
        coords=dict(
            pe=summary_obj.controls["p_exc"],
            nu=summary_obj.controls["nu_ext"],
            vars=list(key_vars)
        ),
        attrs=dict(
            source=summary_obj.fdir,
            wbin=wbin
        )
    )
    
    return dataarray


def construct_summary(summary_obj):
    chi = summary_obj.summary['chi'][...,0].mean(axis=2)
    cv = summary_obj.summary['cv'][...,0].mean(axis=2)
    fr = summary_obj.summary['frs_m'][...,0].mean(axis=2)
    
    fnet = np.zeros_like(fr)
    pnet = np.zeros_like(fr)
    nitr = summary_obj.num_controls[2]
    # for i in tqdm(range(summary_obj.num_controls[0]), ncols=100):
    for i in tqdm(range(summary_obj.num_controls[0]), ncols=100):
        for j in range(summary_obj.num_controls[1]):
            for k in range(nitr):
                detail = summary_obj.load_detail(i,j,k)
                
                p, f = get_fpeak(detail, nitr=5)
                pnet[i,j] += p/nitr
                fnet[i,j] += f/nitr
    
    return dict(
        chi=chi,
        cv=cv,
        fr=fr,
        fnet=fnet,
        pnet=pnet
        )
                     
                
def get_fpeak(detail, nitr=10):
    vlfp = detail["vlfp"][0]
                
    nr0 = np.where(detail["ts"] >= 0.5)[0][0]
    nr1 = np.where(detail["ts"] >= detail["ts"][-1]-wbin)[0][0]
    psd = 0
    for n in range(nitr):
        n0 = int(np.random.rand() * (nr1 - nr0) + nr0)
        n1 = n0 + wbin*srate
        
        v = vlfp[n0:n1]
        _psd, f = hhsignal.get_fft(v, srate, frange=(10, 100))
        psd += _psd
    
    psd /= nitr
    
    nmax = np.argmax(psd)
    return psd[nmax], f[nmax]
                    
    

if __name__ == "__main__":
    main()