# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 00:47:11 2022

@author: USER
"""

import numpy as np
from aerofusion.plot.plot_2D import plot_pcolormesh
import os
import scipy.io as mio
import aerofusion.data.array_conversion as arr_conv

def main():
    
    use_energy = False
    mean_type = 'art'
    Re = 25000
    poi_set = 'reduced'
    
    
    #tForward = '1.0'
    snapshots=150
    num_poi = 37
    modes=100
    energy =0.99
    delta = 1/40
    nSamp = 40
    qoi_type = "fullQOI"
    
    data_folder = "../../lid_driven_data/"
    data_set_prefix = "Re" + str(Re) +"_" + mean_type + "_s" + str(snapshots)
    #data_set_prefix = "s" + str(snapshots)
    if use_energy:
        data_set_prefix += "e" + str(energy)
    else : 
        data_set_prefix += "m" + str(100)
    data_set_prefix += "_l" + str(int(1/delta))
    data_set_suffix = "_nSamp" + str(nSamp) + "_" + qoi_type + "_morris_indices.npz"
    #data_set_suffix = "_nSamp" + str(nSamp) +  "_morris_indices.npz"
    
    data_set_first_name = "sensitivity/" + data_set_prefix+ "_tForward1.0" + data_set_suffix    
    data_set_second_name = "sensitivity/" + data_set_prefix+ "_tForward2.0" + data_set_suffix
    
    plot_folder = "../../Figures/LidDriven/morris/" + data_set_prefix + "_nSamp" +str(nSamp) + "/"
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)
    plot_folder += qoi_type + "_"
        
    filetype = '.pdf'
    
    
    
    num_dim  = 2
    num_xi   = 258
    num_eta  = 258
    num_zeta = 1
    num_cell = num_xi * num_eta
    
    data_set_first = np.load(data_folder + data_set_first_name)
    
    morris_mean_abs_1D_compact = data_set_first["morris_mean_abs"]
    morris_mean_1D_compact = data_set_first["morris_mean"]
    morris_std_1D_compact = data_set_first["morris_std"]
    morris_mean_abs_1D = arr_conv.array_compact_to_1D(morris_mean_abs_1D_compact.transpose(),
                                                 num_dim = num_dim,
                                                 num_cell = num_cell)
    morris_mean_1D = arr_conv.array_compact_to_1D(morris_mean_1D_compact.transpose(),
                                                 num_dim = num_dim,
                                                 num_cell = num_cell)
    morris_std_1D = arr_conv.array_compact_to_1D(morris_std_1D_compact.transpose(),
                                                 num_dim = num_dim,
                                                 num_cell = num_cell)
    
    
    
    centroid_file=np.load(data_folder + "cell_center_high_res.npz")
    cell_centroid=np.zeros((num_xi,num_eta,num_zeta,num_dim))
    cell_centroid[:,:,0,0] = centroid_file['cell_center_x']
    cell_centroid[:,:,0,1] = centroid_file['cell_center_y']
    num_cell = num_xi*num_eta*num_zeta
    mat2=mio.loadmat(data_folder + "Xi_hr.mat")
    Xi=np.ndarray.flatten(mat2['Xi'])
    
    mat2=mio.loadmat(data_folder + "Eta_hr.mat")
    Eta=np.ndarray.flatten(mat2['Eta'])
    
    morris_mean_abs_2D = np.empty((num_poi, num_dim, num_xi, num_eta))
    morris_mean_2D = np.empty((num_poi, num_dim,  num_xi, num_eta))
    morris_std_2D = np.empty((num_poi, num_dim,  num_xi, num_eta))
    
    
    for i in range(num_poi):
        for j in range(num_dim): 
            morris_mean_abs_2D[i,j,:,:] = arr_conv.array_1D_to_2D(\
                           Xi, Eta, num_xi, num_eta, morris_mean_abs_1D[:,j,i])
            
            morris_mean_2D[i,j,:,:] = arr_conv.array_1D_to_2D(\
                           Xi, Eta, num_xi, num_eta, morris_mean_1D[:,j,i])
            
            morris_std_2D[i,j,:,:] = arr_conv.array_1D_to_2D(\
                           Xi, Eta, num_xi, num_eta, morris_std_1D[:,j,i])
            
    
    centroid_file=np.load(data_folder + "cell_center_high_res.npz")
    cell_centroid=np.zeros((num_xi,num_eta,num_zeta,num_dim))
    cell_centroid[:,:,0,0] = centroid_file['cell_center_x']
    cell_centroid[:,:,0,1] = centroid_file['cell_center_y']
    num_cell = num_xi*num_eta*num_zeta
    mat2=mio.loadmat(data_folder + "Xi_hr.mat")
    Xi=np.ndarray.flatten(mat2['Xi'])
    
    mat2=mio.loadmat(data_folder + "Eta_hr.mat")
    Eta=np.ndarray.flatten(mat2['Eta'])
    
    #Convert sensitivity from 1D to 2D

    
    plot_pcolormesh(\
      cell_centroid[:,:,0,0],
      cell_centroid[:,:,0,1],
      morris_mean_abs_2D[1,0],
      plot_folder + "mean_abs_u_alpha",
      vmin = "auto", #-np.max(np.abs(data_2D)),
      vmax = "auto", #np.max(np.abs(data_2D)),
      colorbar_label= r'$\mu^*$ of $u$ with respect to $\log_{10}(\alpha)$',
      font_size = 30)
        
    plot_pcolormesh(\
      cell_centroid[:,:,0,0],
      cell_centroid[:,:,0,1],
      morris_mean_abs_2D[1,1].transpose(),
      plot_folder + "mean_abs_v_alpha",
      vmin = "auto", #-np.max(np.abs(data_2D)),
      vmax = "auto", #np.max(np.abs(data_2D)),
      colorbar_label= r'$\mu^*$ of $v$ with respect to $\log_{10}(\alpha)$',
      font_size = 30)
        
    plot_pcolormesh(\
      cell_centroid[:,:,0,0],
      cell_centroid[:,:,0,1],
      morris_mean_abs_2D[2,0],
      plot_folder + "mean_abs_u_tau",
      vmin = "auto", #-np.max(np.abs(data_2D)),
      vmax = "auto", #np.max(np.abs(data_2D)),
      colorbar_label= r'$\mu^*$ of $u$ with respect to $\log_{10}(\tau)$',
      font_size = 30)
        
    plot_pcolormesh(\
      cell_centroid[:,:,0,0],
      cell_centroid[:,:,0,1],
      morris_mean_abs_2D[2,1].transpose(),
      plot_folder + "mean_abs_v_tau",
      vmin = "auto", #-np.max(np.abs(data_2D)),
      vmax = "auto", #np.max(np.abs(data_2D)),
      colorbar_label= r'$\mu^*$ of $v$ with respect to $\log_{10}(\tau)$',
      font_size = 30)
        
    plot_pcolormesh(\
      cell_centroid[:,:,0,0],
      cell_centroid[:,:,0,1],
      morris_mean_abs_2D[3,0],
      plot_folder + "mean_abs_u_v1",
      vmin = "auto", #-np.max(np.abs(data_2D)),
      vmax = "auto", #np.max(np.abs(data_2D)),
      colorbar_label= r'$\mu^*$ of $u$ with respect to $\bar{v}_1$',
      font_size = 30)
        
    plot_pcolormesh(\
      cell_centroid[:,:,0,0],
      cell_centroid[:,:,0,1],
      morris_mean_abs_2D[3,1].transpose(),
      plot_folder + "mean_abs_v_v1",
      vmin = "auto", #-np.max(np.abs(data_2D)),
      vmax = "auto", #np.max(np.abs(data_2D)),
      colorbar_label= r'$\mu^*$ of $v$ with respect to $\bar{v}_1$',
      font_size = 30)
   

if __name__ == '__main__':
    main()