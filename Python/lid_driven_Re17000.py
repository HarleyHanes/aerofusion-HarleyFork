# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:10:51 2021

@author: cjedw
"""

import scipy.io as mio
from scipy import array
from scipy.linalg import svd,svdvals
import numpy as np
import copy
import sys
import io
import os
from pathlib import Path
import numpy as np
import logging
import argparse
import libconf
import findiff
import matplotlib.pyplot as plt
try:
  from tqdm import tqdm
except:
  # tqdm not present; avoid crashing and simply not report progress bar
  def tqdm(input_arg):
    return input_arg

# aerofusion modules
#from aerofusion.io import hdf5_cell_data_to_numpy_array
from aerofusion.pod import pod_modes
#from aerofusion.dmd import dmd_modes
#from aerofusion.rom import incompressible_navier_stokes_rom as incrom
#from aerofusion.data import array_conversion as arr_conv
#from aerofusion.plot.plot_2D import plot_contour
#from aerofusion.numerics import derivatives_curvilinear_grid as curvder
#from aerofusion.numerics import curl_calc as curl_calc
import mat73

data_folder = "../../lid_driven_data"
velocity_file = "re20000_lr.mat"
weights_file = "w_LowRes.mat"
u0_type = "mean"
if u0_type.lower() == "artificial":
    u0_file = "vel_artificial"
modes = 4
pod_file = data_folder + "pod_Re20000lr_s500m" + str(modes)
t_iter = 20

mat=mat73.loadmat(data_folder + velocity_file)
velocity_1D_compact=mat['X'][:]
mat2=mio.loadmat(data_folder + weights_file)
weights=np.ndarray.flatten(mat2['w'])



simulation_time = np.zeros((t_iter,))
for i in range(t_iter):
    simulation_time[i]=i*0.1


num_dim  = 2
num_xi   = 130
num_eta  = 130
num_zeta = 1
num_cell = num_xi*num_eta*num_zeta
num_snapshots = velocity_1D_compact.shape[1]
num_dof = int(num_cell * num_dim)


#---------------------------------------Get Mean Velocity
velocity_mean = np.zeros([num_dof])
if u0_type.lower() == "artifial":
    velocity_mean = np.load(data_folder + u0_file)
elif u0_type.lower() == "mean":
    velocity_mean = pod_modes.Find_Mean(velocity_1D_compact)
else:
    raise(Exception("u0 type: " + str(u0_type) + "not recognized"))
    
#--------------------------------------
mean_reduced_velocity = np.zeros(velocity_1D_compact.shape)
for i_snap in range(num_snapshots):
  mean_reduced_velocity[:,i_snap] = \
    velocity_1D_compact[:,i_snap] - velocity_mean[:]

    
(phi, modal_coeff, pod_lambda) = pod_modes.Find_Modes(\
    mean_reduced_velocity,
    weights,
    modes)


np.savez(pod_file,
         simulation_time = simulation_time,
         phi = phi,
         pod_lambda = pod_lambda,
         modal_coeff = modal_coeff,
         velocity_mean = velocity_mean)

