# 

The meaning of digit for code word is:
    0: fast pop / fast osc
    1: fast pop / slow osc
    2: slow pop / fast osc
    3: slow pop / slow osc
==> Changed the label: F(fs)S(fs)

# Detect oscillation motif
[detect_oscillation_motif.ipynb](detect_oscillation_motif.ipynb)

> TODO
> Need to handle the threshold (lower threshold quantile?)

# Compute Transfer entropy between fast and slow subpopulation activity in each oscillation motif


> TODO
> surrogate test


- Validation of TE method is in [te_validation](te_validation.ipynb)
    - Use data "./data/te_validation_???.pkl"


# helper library
[oscdetector.py](oscdetector.py)
[hhinfo.py](../include/hhinfo.py)
