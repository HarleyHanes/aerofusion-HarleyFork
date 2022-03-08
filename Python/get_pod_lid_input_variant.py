# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 16:10:51 2021

@author: cjedw
"""

import scipy.io as mio
from scipy import array
from scipy.linalg import svd,svdvals
import numpy as np
#import copy
#import sys
#import io
#import os
#from pathlib import Path
#import logging
#import argparse
#import libconf
#import findiff
#import matplotlib.pyplot as plt
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

def main(velocity_1D_compact, u0_type, artificial_mean_reduced_velocity, weights_ND, modes):
    #print("velocity shape: " + str(velocity_1D_compact.shape))
    num_snapshots = velocity_1D_compact.shape[1]
    
    
    #---------------------------------------Get Mean Velocity
    velocity_mean = np.zeros(velocity_1D_compact.shape[0])
    if u0_type.lower() == "artificial":
        velocity_mean = artificial_mean_reduced_velocity
    elif u0_type.lower() == "mean":
        velocity_mean = pod_modes.Find_Mean(velocity_1D_compact)
    else:
        raise(Exception("u0 type: " + str(u0_type) + " not recognized"))
        
    #--------------------------------------
    mean_reduced_velocity = np.zeros(velocity_1D_compact.shape)
    for i_snap in range(num_snapshots):
      mean_reduced_velocity[:,i_snap] = \
        velocity_1D_compact[:,i_snap] - velocity_mean[:]
    
        
    (phi, modal_coeff, pod_lambda) = pod_modes.Find_Modes(\
        mean_reduced_velocity,
        weights_ND,
        modes)
    
    
    return (phi, modal_coeff, pod_lambda)

if __name__ == '__main__':
    main()


