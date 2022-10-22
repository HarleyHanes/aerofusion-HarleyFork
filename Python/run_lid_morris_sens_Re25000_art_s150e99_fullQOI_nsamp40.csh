#!/bin/tcsh
#BSUB -W 48:00
#BSUB -n 12
#BSUB -R "select[hc] span[ptile=6]"
#BSUB -x
#BSUB -o outputs/out.%J
#BSUB -e outputs/err.%J

conda activate /rs1/researchers/r/rsmith/pkgs/env_aerofusion
mpirun -n 12 python3 lid_driven_sensitivity_analysis.py run_morris t_forward 1 Re 25000 qoi_set fullQOI n_samp_morris 40 energy .99
mpirun -n 12 python3 lid_driven_sensitivity_analysis.py run_morris t_forward 2 Re 25000 qoi_set fullQOI n_samp_morris 40 energy .99