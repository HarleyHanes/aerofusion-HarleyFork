# -*- coding: utf-8 -*-
"""
Created on Thu May 19 18:26:11 2022

@author: USER
"""

import numpy as np
import mat73
import scipy.io as mio

from aerofusion.io import hdf5_cell_data_to_numpy_array
from aerofusion.pod import pod_modes
from aerofusion.rom import incompressible_navier_stokes_rom as incrom
from aerofusion.data import array_conversion as arr_conv
from aerofusion.plot.plot_2D import plot_contour
from aerofusion.numerics import derivatives_curvilinear_grid as curvder

data_folder = "../../lid_driven_data/"
weights_file = "weights_hr.mat"
velocity_file = "re17000_hr.mat"

num_dim  = 2
num_xi   = 258
num_eta  = 258
num_zeta = 1
n_snapshot = 150


mat = mat73.loadmat(data_folder + velocity_file)
velocity_1D_compact=mat['X'][:,0:n_snapshot]
del mat

mat2 = mio.loadmat(data_folder + weights_file)
weights = np.ndarray.flatten(mat2['W'])
weights_ND = np.repeat(weights.reshape(weights.size,1), 2, axis=1).transpose().flatten()

mat2=mio.loadmat(data_folder + "Xi_hr.mat")
Xi=np.ndarray.flatten(mat2['Xi'])

mat2=mio.loadmat(data_folder + "Eta_hr.mat")
Eta=np.ndarray.flatten(mat2['Eta'])

del mat2

centroid_file=np.load(data_folder + "cell_center_high_res.npz")
cell_centroid=np.zeros((num_xi,num_eta,num_zeta,num_dim))
cell_centroid[:,:,0,0] = centroid_file['cell_center_x']
cell_centroid[:,:,0,1] = centroid_file['cell_center_y']
num_cell = num_xi*num_eta*num_zeta


num_dof = int(num_cell * num_dim)
# mean calculation of velocity
velocity_mean = np.zeros([num_dof])
velocity_mean = pod_modes.Find_Mean(velocity_1D_compact)

mean_reduced_velocity = np.zeros(velocity_1D_compact.shape)
for i_mode in range(n_snapshot):
  mean_reduced_velocity[:,i_mode] = \
    velocity_1D_compact[:,i_mode] - velocity_mean[:]



(phi, modal_coeff, pod_lambda) = pod_modes.Find_Modes(\
    mean_reduced_velocity,
    weights_ND,
    n_snapshot)
    
prop_energy = modal_coeff/ np.sum(np.abs(modal_coeff))

print(prop_energy)
print("100 snapshots: " + str(prop_energy[99]))
print("120 snapshots: " + str(prop_energy[119]))
