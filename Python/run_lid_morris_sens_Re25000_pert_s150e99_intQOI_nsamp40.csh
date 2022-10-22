#!/bin/tcsh
#BSUB -W 48:00
#BSUB -n 12
#BSUB -R "select[hc] span[ptile=6]"
#BSUB -x
#BSUB -o outputs/out.%J
#BSUB -e outputs/err.%J

conda activate /rs1/researchers/r/rsmith/pkgs/env_aerofusion
mpirun -n 12 python3 lid_driven_sensitivity_analysis.py  run_morris Re 25000 t_forward 1 qoi_set intQOI n_samp_morris 40 mean_perturbed energy .99
mpirun -n 12 python3 lid_driven_sensitivity_analysis.py  run_morris Re 25000 t_forward 2 qoi_set intQOI n_samp_morris 40 mean_perturbed energy .99