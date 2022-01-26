# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 12:21:45 2021

@author: cjedw
"""

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
from aerofusion.io import hdf5_cell_data_to_numpy_array
from aerofusion.pod import pod_modes
from aerofusion.rom import incompressible_navier_stokes_rom as incrom
from aerofusion.data import array_conversion as arr_conv
from aerofusion.plot.plot_2D import plot_contour
from aerofusion.numerics import derivatives_curvilinear_grid as curvder
from aerofusion.numerics import curl_calc as curl_calc
import mat73
import scipy.io as mio
#from aerofusion.numerics.interpolation import interpolate_vectorial_field_in_2D

# -----------------------------------------------------------------------------
def main(argv=None):


    rom_data = np.load('Rom_coeff.npz')
    # Assign data to convenience variables
    velocity_mean   = rom_data['velocity_mean']
    phi             = rom_data['phi']
    modal_coeff     = rom_data['aT']
    reconstructed=np.dot(phi,modal_coeff)
    
    mat2=mio.loadmat("Xi_hr.mat")
    Xi=np.ndarray.flatten(mat2['Xi'])
    mat2=mio.loadmat("Eta_hr.mat")
    Eta=np.ndarray.flatten(mat2['Eta'])
    mat2=mio.loadmat("C_x_hr.mat")
    cell_center_x=mat2['C']
    mat2=mio.loadmat("C_y_hr.mat")
    cell_center_y=mat2['C2']
    
    cell_centroid=np.zeros((258,258,1,2))
    cell_centroid[:,:,0,0]=cell_center_x
    cell_centroid[:,:,0,1]=cell_center_y
    

   
    num_snapshots=reconstructed.shape[1]
    num_dim  = 2
    num_xi   = 258
    num_eta  = 258
    num_zeta = 1
    num_cell = 66564
    zeta=np.zeros((Xi.shape[0],),dtype='int')
    

    for i_mode in range(num_snapshots):
      reconstructed[:,i_mode] = \
        reconstructed[:,i_mode] + velocity_mean[:]
        
    A1D=np.zeros([num_cell,num_dim,num_snapshots])
    for i in range(0,num_snapshots):
        A1D[:,:,i]=np.reshape(reconstructed[:,i],(num_dim,num_cell)).transpose()
        
    A3D=np.zeros([num_xi,num_eta,num_zeta,num_dim,num_snapshots])
    for i in range(num_dim):
        for i_mode in range(0,num_snapshots):
            A3D[:,:,:,i,i_mode]=arr_conv.array_1D_to_3D(Xi, Eta,zeta, num_cell, A1D[:,i,i_mode])

    vorticity = \
      curl_calc.curl_2d(-cell_centroid[:,0,0,1], -cell_centroid[0,:,0,0],
        A3D[:, :,0, 0, 1999], A3D[:, :, 0,1, 1999])
      
    fig, axs = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=0.3)
    
    cset1 = axs.pcolor(cell_centroid[:,:,0,0],cell_centroid[:,:,0,1],vorticity
                         , cmap='bwr', vmin=-3,vmax=3)
      #cset1 = axs.contourf(X, Y, Z, levels = n_levels, cmap='bwr')
      #axs.set_xlim([0.4, 1])
    fig.colorbar(cset1)
    fig.tight_layout()
    plt.savefig("ROM_Vorticity_t=200.pdf")


#------------------------------------------------------------------------------
if __name__ == "__main__":
  sys.exit(main())
