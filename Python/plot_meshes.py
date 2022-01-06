import sys
import io
import os
import UQtoolbox as uq
import aerofusion.data.array_conversion as arr_conv
from aerofusion.rom import incompressible_navier_stokes_rom as incrom
import numpy as np
import logging
import argparse
import libconf
from aerofusion.plot.plot_2D import plot_contour
from aerofusion.plot.plot_2D import plot_pcolormesh
import matplotlib.pyplot as plt
import scipy.io as mio

def main(argv=None):
    #Variables to load
    filename = "../../lid_driven_snapshots/full data/results_lsa.npz"
    num_points = [0, 129, -1]
    #Loop through modes
    results = np.load(filename)
    jac = results["sensitivities"]
    
    #Load mesh data
    #rom_matrices_filename="../../lid_driven_penalty/rom_matrices_50.npz"
    #penalty=10.0**4 penalty defined explicitly in function
    #QOI_type = "full data"
    tmax=50
    penalty_exp=4
    

    data_folder = "../../lid_driven_snapshots/"
    plot_folder = "../../lid_driven_snapshots/full data/"
    pod_data = np.load(data_folder + 'pod_lid_driven_50.npz')
    # Assign data to convenience variables
    #vel_0  = pod_data['velocity_mean']
    #simulation_time = pod_data['simulation_time']
    phi             = pod_data['phi']
    modal_coeff     = pod_data['modal_coeff']
    
    #Get solution
    sol = np.matmul(phi, modal_coeff)
    integration_index = 499
    
    #Specify when integration takes place 
    #integration_times = np.arange(.1,tmax,.1)
    #integration_indices = np.arange(1,len(integration_times)+1)
    #integration_indices = np.arange(1,500)
    #integration_times = simulation_time[integration_indices]
    #num_time = len(integration_times)
    
    #mat2=mio.loadmat(data_folder + "weights_hr.mat")
    #weights=np.ndarray.flatten(mat2['W'])
    mat2=mio.loadmat(data_folder + "Xi_hr.mat")
    Xi=np.ndarray.flatten(mat2['Xi'])
    mat2=mio.loadmat(data_folder + "Eta_hr.mat")
    Eta=np.ndarray.flatten(mat2['Eta'])
    mat2=mio.loadmat(data_folder + "C_x_hr.mat")
    cell_center_x=mat2['C']
    mat2=mio.loadmat(data_folder + "C_y_hr.mat")
    cell_center_y=mat2['C2']
    
    cell_centroid=np.zeros((258,258,1,2))
    cell_centroid[:,:,0,0]=cell_center_x
    cell_centroid[:,:,0,1]=cell_center_y

   
    num_dim  = 2
    num_xi   = 258
    num_eta  = 258
    #num_zeta = 1
    num_cell = 66564
    #Zeta=np.zeros((Xi.shape[0],),dtype='int')
    
    
    Xi_mesh=Xi.reshape((num_eta, num_xi))
    Eta_mesh=Eta.reshape((num_eta, num_xi))
    Xi_mesh = (Xi_mesh- (num_xi-1)/2)/(num_xi/2)
    Eta_mesh = (Eta_mesh- (num_eta-1)/2)/(num_eta/2)
    #Convert to 2D
    #Seperate Dimensions
    jac_1D = np.empty((num_cell, num_dim, jac.shape[1]))
    sol_1D = np.empty((num_cell, num_dim))
    for iTime in range(jac.shape[1]):
        jac_1D[:,:,iTime] = np.reshape(jac[:, iTime], (num_cell, num_dim), order = 'F')
    sol_1D[:,:] = np.reshape(sol[:, integration_index], (num_cell, num_dim), order= 'F')
    
    
    
    jac_2D=np.empty((num_xi, num_eta, num_dim, jac.shape[1]))
    sol_2D=np.empty((num_xi, num_eta, num_dim))
    for i_dim in range(num_dim):
        sol_2D[:,:,i_dim] = arr_conv.array_1D_to_2D(\
          Xi, Eta, num_xi, num_eta, sol_1D[:,i_dim])
        for i_point in range(jac.shape[1]):
          jac_2D[:,:,i_dim, i_point] = arr_conv.array_1D_to_2D(\
            Xi, Eta, num_xi, num_eta, jac_1D[:,i_dim,i_point])
  
    plot_pcolormesh(\
      Xi_mesh,
      Eta_mesh,
      sol_2D[:,:,0],
      plot_folder + 'u_reduced_pen=10^' + str(penalty_exp) +'_t= ' + str(tmax) + '.png',
      "auto",
      "auto",
      "auto",
      colorbar_label= 'u reduced',
      font_size = 30)
    plot_pcolormesh(\
      Xi_mesh,
      Eta_mesh,
      sol_2D[:,:,1],
      plot_folder + 'v_reduced_pen=10^' + str(penalty_exp) +'_t= ' + str(tmax) + '.png',
      "auto",
      "auto",
      "auto",
      colorbar_label= 'v reduced',
      font_size = 30)
    for i_point in range(len(num_points)):
        point = num_points[i_point]
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          jac_2D[:,:,0, point],
          plot_folder + 'u_sens_pen=10^' + str(penalty_exp) +'_x='+ str(Xi_mesh[0][point]) + '_t= ' + str(tmax) + '.png',
          "auto",
          "auto",
          "auto",
          colorbar_label= 'u reduced sensitivity',
          font_size = 30)
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          jac_2D[:,:,1, point],
          plot_folder + 'v_sens_pen=10^' + str(penalty_exp) +'_x='+ str(Xi_mesh[0][point]) + '_t= ' + str(tmax) + '.png',
          "auto",
          "auto",
          "auto",
          colorbar_label= 'v reduced sensitivity',
          font_size = 30)

if __name__ == "__main__":
    sys.exit(main())
