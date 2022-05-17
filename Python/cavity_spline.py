# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:00:07 2022

@author: USER
"""
import numpy as np
import sys


def main():
    



def cavity_linear_spline(pos_vec, velocity_2D, cell_centroid):
    n_dim = 2
    #Get x and y center vectors
    x_center = cell_centroid[:,0,0,0]
    y_center = cell_centroid[0,:,0,1]
    if x_center[0] == x_center[1]:
        raise Exception("no stepping in x_centers")
    if y_center[0] == y_center[1]:
        raise Exception("no stepping in y_centers")
    # Check pos_vec dimensions and intialize velocity data 
    if pos_vec.ndim == 2:
        n_samp = pos_vec.shape[0]
        if pos_vec.shape[1] !=2:
            raise Exception(str(pos_vec.shape[1]) + " velocity dimensions detected. Only two implemented.")
    elif pos_vec.ndim == 1:
        n_samp = 1
        if pos_vec.size !=2:
            raise Exception(str(pos_vec.size) + " velocity dimensions detected. Only two implemented.")
    else:
        raise Exception("pos_vec is more than 2D")
    spline_vel = np.empty(n_samp, n_dim)
    # Loop Through pos_vec sample points
    for i_samp in range(n_samp):
        (x_pos, y_pos) = pos_vec[i_samp,:]
        #Identify v00, v01, v10, v11
        x_lower_index = np.searchsorted(x_pos, x_center, side='left', sorter=None)
        y_lower_index = np.searchsorted(y_pos, y_center, side='left', sorter=None)
        x_lower = x_center[x_lower_index]
        y_lower = y_center[y_lower_index]
        x_upper = x_center[x_lower_index + 1]
        y_upper = y_center[y_lower_index + 1]
        v11 = velocity_2D[x_lower_index + 1, y_lower_index + 1,:]
        v00 = velocity_2D[x_lower_index, y_lower_index,:]
        v01 = velocity_2D[x_lower_index, y_lower_index + 1,:]
        v10 = velocity_2D[x_lower_index + 1, y_lower_index,:]
        #Compute a and b
        x_prop = (x_pos-x_lower)/(x_upper-x_lower)
        y_prop = (y_pos-y_lower)/(y_upper-y_lower)
        
        #compute velocity
        spline_vel[i_samp, :] = x_prop*y_prop*v11+x_prop*(1-y_prop)*v10 + \
                              (1-x_prop)*y_prop*v01+(1-x_prop)*(1-y_prop)*v00
    return spline_vel

    
if __name__ == "__main__":
    sys.exit(main())
