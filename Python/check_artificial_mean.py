# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 11:55:35 2022

@author: USER
"""

import numpy as np
import get_artificial_mean
import scipy.io as mio

def main():
    basis_vort_vec = np.array([1, 2])
    basis_orient_vec = np.array([np.pi/2, np.pi/4])
    basis_x_loc_vec = np.array([-.75, .75])
    basis_y_loc_vec = np.array([-.75, .75])
    basis_extent_vec = np.array([2, 3])
    
    
    data_folder = "../../lid_driven_data/"
    mat2=mio.loadmat(data_folder + "weights_hr.mat")
    weights=np.ndarray.flatten(mat2['W'])
    mat2=mio.loadmat(data_folder + "Xi_hr.mat")
    Xi=np.ndarray.flatten(mat2['Xi'])
    mat2=mio.loadmat(data_folder + "Eta_hr.mat")
    Eta=np.ndarray.flatten(mat2['Eta'])
    centroid_file=np.load(data_folder + "cell_center_high_res.npz")
    
   
    num_dim  = 2
    num_xi   = 258
    num_eta  = 258
    num_zeta = 1
    num_cell = 66564
    
    cell_centroid=np.zeros((num_xi,num_eta,num_zeta,num_dim))
    cell_centroid[:,:,0,0] = centroid_file['cell_center_x']
    cell_centroid[:,:,0,1] = centroid_file['cell_center_y']
    #Zeta=np.zeros((Xi.shape[0],),dtype='int')
    
    
    Xi_mesh=Xi.reshape((num_eta, num_xi))
    Eta_mesh=Eta.reshape((num_eta, num_xi))
    Xi_mesh = (Xi_mesh- (num_xi-1)/2)/(num_xi/2)
    Eta_mesh = (Eta_mesh- (num_eta-1)/2)/(num_eta/2)
    
    (v1,v2,v3) = get_artificial_mean.main(basis_vort_vec, basis_orient_vec, basis_x_loc_vec, basis_y_loc_vec, \
             basis_extent_vec, Xi, Eta, Xi_mesh, Eta_mesh, cell_centroid, plot = True)
    

if __name__ == '__main__':
    main()

