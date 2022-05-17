# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 10:58:11 2022

@author: USER
"""
import UQLibrary as uq
import numpy as np

algorithm = 'smith'
n_snapshot = 150
n_modes = 100
t_forward = 1
tolerance = 1e-7
reduction_nsamp = 200

data_folder = "../../lid_driven_data/"
name_qoi = np.array(["Kinetic Energy", "Vorticity", \
                      "Local Vorticity 0", "Local Vorticity 1", "Local Vorticity 2",\
                      "Local Vorticity 3", "Local Vorticity 4", "Local Vorticity 5", \
                      "Local Vorticity 6"])

    
save_location = data_folder + "sensitivity/"+str(algorithm).lower()+\
"_plots/reduction_s" + str(n_snapshot) + "m" + str(n_modes) + "_tForward"\
+ str(t_forward) + "_tol" +  str(int(np.log10(tolerance)*1000)/1000)\
+ "_nsamp" + str(reduction_nsamp) +  "_"

#load data
data = np.load(save_location + "data.npz")
qoi_samp_full = data['qoi_samp_full']
qoi_samp_reduced = data['qoi_samp_reduced']

uq.lsa.test_model_reduction_precomputed(name_qoi, qoi_samp_full, qoi_samp_reduced,\
                                        reduction_nsamp, save_location, tolerance,
                                        fontsize = 16, linewidth = 3)
    