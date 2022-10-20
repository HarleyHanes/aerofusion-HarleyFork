#!/bin/tcsh
#BSUB -W 48:00
#BSUB -n 12
#BSUB -R "select[hc] span[ptile=6]"
#BSUB -x
#BSUB -o outputs/out.%J
#BSUB -e outputs/err.%J

conda activate /rs1/researchers/r/rsmith/pkgs/env_aerofusion
mpirun -n 12 python3 lid_driven_morris.py run_morris Re 17000 qoi_set fullQOI n_samp_morris 40