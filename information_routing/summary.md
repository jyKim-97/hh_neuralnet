# 

The meaning of digit for code word is:
    0: fast pop / fast osc
    1: fast pop / slow osc
    2: slow pop / fast osc
    3: slow pop / slow osc
==> Changed the label: F(fs)S(fs)

# Detect oscillation motif
[detect_oscillation_motif.ipynb](detect_oscillation_motif.ipynb)
Determine the threshold and frequency range of oscillation motif.

[export_oscmotif.py](export_oscmotif.py)
Detect oscillation motifs (or multi-frequency oscillation patterns, MFOP) based on the determined frequency range

* figures
- figs/freq_peaks: Show the process how I detected the oscillation motif
- figs/spec: Show the spectral properties

# Compute spectral characteristics 
[computeFT.py](./computeFT.py)
Compute averaged spectral characteristics

# Compute Transfer entropy between fast and slow subpopulation activity in each oscillation motif
[computeTE4.py](./computeTE4.py)
Compute TE in each oscillation motif, there are 5 options: naive, spo, mit, 2d, full

[te_result3.ipynb](./te_result3.ipynb)
Visualize the compute TE 
__TODO__: Need to arange the cells

<!-- - Validation of TE method is in [te_validation](te_validation.ipynb)
    - Use data "./data/te_validation_???.pkl" -->


# helper library
[oscdetector.py](oscdetector.py)
[hhinfo.py](../include/hhinfo.py)
[utils.py](utils.py)
[visu.py](visu.py)
[tetools.py](tetools.py)
