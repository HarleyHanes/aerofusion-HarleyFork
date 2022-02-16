import sys
import io
import os
import UQtoolbox as uq
import aerofusion.data.array_conversion as arr_conv
from aerofusion.rom import incompressible_navier_stokes_rom as incrom
from aerofusion.numerics import curl_calc as curl_calc
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
    method = "art"
    POI_type = "alpha"
    QOI_type = "vorticity"
    modes = 100
    tmax = 100
    penalty_exp=2
    
    #filename = "../../lid_driven_snapshots/full data/results_lsa.npz"
    if POI_type.lower() == "point":
        num_points = [20, 129, -2]
    elif POI_type.lower() == "alpha":
        alpha_value = .01
        
    data_folder = "../../lid_driven_data/"
    pod_filename="pod_Re17000hr_"+method+"_s500m100.npz"
    filename = "../../lid_driven_data/" + method + "_" + QOI_type + "_a" + str(alpha_value) + "_s500m" + str(modes) + "_results_lsa.npz"
    plot_folder = "../../lid_driven_snapshots/" + method + "_u0/sens/"
    #plot_folder = "../../lid_driven_snapshots/" + method + "_u0/"
    plot_name = "sens_abs_" + QOI_type + "_a" + str(alpha_value) + "_s500m" + str(modes) + "_p" + str(penalty_exp)+"_t="+ str(tmax) + ".png"
    #plot_name = QOI_type + "_s500m" + str(modes) + ".png" 
    
    
    #Load mesh data
    #rom_matrices_filename="../../lid_driven_penalty/rom_matrices_50.npz"
    #penalty=10.0**4 penalty defined explicitly in function
    #QOI_type = "full data"
    

    pod_data = np.load(data_folder + pod_filename)
    # Assign data to convenience variables
    vel_0  = pod_data['velocity_mean']
    #simulation_time = pod_data['simulation_time']
    phi             = pod_data['phi']
    modal_coeff     = pod_data['modal_coeff']
    
    #Get solution
    sol = np.matmul(phi, modal_coeff)
    integration_index = tmax*10-1
    
    #Specify when integration takes place 
    #integration_times = np.arange(.1,tmax,.1)
    #integration_indices = np.arange(1,len(integration_times)+1)
    #integration_indices = np.arange(1,500)
    #integration_times = simulation_time[integration_indices]
    #num_time = len(integration_times)
    
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
    #-------------------------------Get 2D data------------------------------
    weights_ND = np.zeros([num_cell*num_dim])
    for i_dim in range(num_dim):
        weights_ND[i_dim*num_cell : (i_dim+1)*num_cell] = weights
    #Parse data
    if QOI_type.lower() == "vorticity":
        #Loop through modes
        results = np.load(filename)
        jac = results["sensitivities"]
        data_1D = jac[:,0]
        #Convert Data to 2D
        data_2D = arr_conv.array_1D_to_2D(Xi, Eta, num_xi, num_eta, data_1D)
    elif QOI_type.lower() == "u0 vorticity":
        vel_1D_compact = vel_0
        #convert to 1D
        vel_1D = np.reshape(vel_1D_compact, (num_cell, 2), order = "F")
        #convert to 2D
        vel_2D = np.zeros((num_xi, num_eta, 2))
        for i_dim in range(2):
              vel_2D[:,:,i_dim] = arr_conv.array_1D_to_2D(\
                Xi, Eta, num_xi, num_eta, vel_1D[:,i_dim])
        #calculate vorticity
        data_2D = curl_calc.curl_2d(-cell_centroid[:,0,0,1], -cell_centroid[0,:,0,0],
          vel_2D[:, :, 0], vel_2D[:, :,1])
    else:
        raise(Exception("Plotting currently deprecated except for vorticity, see commented sections"))
    
    #-----------------------------Plot---------------------------------------
    plot_pcolormesh(\
      Xi_mesh,
      Eta_mesh,
      np.abs(data_2D),
      plot_folder + plot_name,
      vmin = "auto", #-np.max(np.abs(data_2D)),
      vmax = "auto", #np.max(np.abs(data_2D)),
      colorbar_label= QOI_type,
      font_size = 30)
''' 
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
              
    #Compute QOIs
    Sens_2D=np.empty((num_xi, num_eta, num_dim, len(num_points)))
    for iPoint in range(len(num_points)):
        # Get index of point
        uIndex = num_points[iPoint]*num_eta
        vIndex = uIndex+num_cell
        # Extract Phi_l and weightL
        phiU = phi[[uIndex],:].transpose()
        phiV = phi[[vIndex],:].transpose()
        weightU = weights_ND[uIndex]
        weightV = weights_ND[vIndex]
        # Compute Sensitivity for each
        SensU= tmax*10**(penalty_exp)*np.matmul(phi[0:num_cell], phiU*weightU)
        SensV= tmax*10**(penalty_exp)*np.matmul(phi[num_cell:], phiV*weightV)
        #Recombine sensitivities into usual structure
        Sens = np.concatenate((SensU, SensV))
        #transform to mesh form
        Sens_1D = np.reshape(Sens, (num_cell, num_dim), order = 'F')
        for i_dim in range(num_dim):
            Sens_2D[:,:,i_dim, iPoint] = arr_conv.array_1D_to_2D(\
             Xi, Eta, num_xi, num_eta, Sens_1D[:,i_dim])
          
  
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
    print (jac_2D.shape)
    for i_point in range(len(num_points)):
        point = num_points[i_point]
        #----------------------------Plot numerical sensitivities-------------
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          jac_2D[:,:,0, point],
          plot_folder + 'u_sens_numerical_pen=10^' + str(penalty_exp) +'_x='+ str(Xi_mesh[0][point]) + '_t= ' + str(tmax) + '.png',
          "auto",
          "auto",
          "auto",
          colorbar_label= 'u reduced sensitivity',
          font_size = 30)
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          jac_2D[:,:,1, point],
          plot_folder + 'v_sens_numerical_pen=10^' + str(penalty_exp) +'_x='+ str(Xi_mesh[0][point]) + '_t= ' + str(tmax) + '.png',
          "auto",
          "auto",
          "auto",
          colorbar_label= 'v reduced sensitivity',
          font_size = 30)
        #------------------------Plot analytical sensitivities
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          Sens_2D[:,:,0, i_point],
          plot_folder + 'u_sens_analytical_pen=10^' + str(penalty_exp) +'_x='+ str(Xi_mesh[0][point]) + '_t= ' + str(tmax) + '.png',
          "auto",
          "auto",
          "auto",
          colorbar_label= 'u reduced sensitivity',
          font_size = 30)
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          Sens_2D[:,:,1, i_point],
          plot_folder + 'v_sens_analytical_pen=10^' + str(penalty_exp) +'_x='+ str(Xi_mesh[0][point]) + '_t= ' + str(tmax) + '.png',
          "auto",
          "auto",
          "auto",
          colorbar_label= 'v reduced sensitivity',
          font_size = 30)
'''

if __name__ == "__main__":
    sys.exit(main())
