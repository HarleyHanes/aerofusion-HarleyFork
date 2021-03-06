# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 11:37:50 2021

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
from aerofusion.plot.plot_2D import plot_pcolormesh
from aerofusion.numerics import derivatives_curvilinear_grid as curvder
from aerofusion.numerics import curl_calc as curl_calc
import mat73
import scipy.io as mio
#from aerofusion.numerics.interpolation import interpolate_vectorial_field_in_2D

# -----------------------------------------------------------------------------
def main(argv=None):

    
    rom_matrices_filename="../../lid_driven_penalty/rom_matrices_s500_m1.npz"
    data_folder = "../../lid_driven_snapshots/"
    verify = True
    penalty=10.0**4
    modes = 1        #number of modes to use (starting with 1)
    
    #Define Boundary Function 
    boundaryFcn= lambda x: (np.abs(x-1))**2*(np.abs(x+1))**2
    
    pod_data = np.load(data_folder + 'pod_lid_driven_50.npz')
    # Assign data to convenience variables
    vel_0  = pod_data['velocity_mean']
    simulation_time = pod_data['simulation_time']
    phi             = pod_data['phi']
    modal_coeff     = pod_data['modal_coeff']
    
    #Reduce phi size
    phi=phi[:, 0:modes]
    modal_coeff=modal_coeff[0:modes]
    
    #Specify when integration takes place 
    integration_times = np.arange(.1,50+.1,.1)
    integration_indices = np.arange(1,len(integration_times)+1)
    #integration_indices = np.arange(1,500)
    #integration_times = simulation_time[integration_indices]
    num_time = len(integration_times)
    
    mat2=mio.loadmat(data_folder + "weights_hr.mat")
    weights=np.ndarray.flatten(mat2['W'])
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
    num_zeta = 1
    num_cell = 66564
    zeta=np.zeros((Xi.shape[0],),dtype='int')
    
    Xi_mesh=Xi.reshape((num_eta, num_xi))
    Eta_mesh=Eta.reshape((num_eta, num_xi))
    Xi_mesh = (Xi_mesh- (num_xi-1)/2)/(num_xi/2)
    Eta_mesh = (Eta_mesh- (num_eta-1)/2)/(num_eta/2)
    
    #Compute Boundary 
    boundary_vec=boundaryFcn(Xi_mesh[0,:])
    
    #Load True Deta
    mat2=mat73.loadmat(data_folder + "re25000_hr.mat")
    vel_true = mat2['X']
    vel_true = vel_true[:,integration_indices]
    
    #Seperate Dimensions
    vel_true_1D = np.empty((num_cell, num_dim, num_time))
    for i_time in range(num_time):
        vel_true_1D[:,:, i_time]=np.reshape(vel_true[:, i_time], (num_cell, num_dim), order = 'F')
        
    #Make 2D
    vel_true_2D=np.empty((num_xi, num_eta, num_dim, num_time))
    for i_dim in range(num_dim):
        for i_time in range(num_time):
          vel_true_2D[:,:,i_dim, i_time] = arr_conv.array_1D_to_2D(\
            Xi, Eta, num_xi, num_eta, vel_true_1D[:,i_dim,i_time])

    weights_ND = np.zeros([num_cell*num_dim])
    for i_dim in range(num_dim):
      weights_ND[i_dim*num_cell : (i_dim+1)*num_cell] = weights
        
    vel_0_1D = np.reshape(vel_0, (num_cell, num_dim), order= 'F')
    vel_0_2D = np.zeros([num_xi, num_eta, num_dim])
    vel_0_3D = np.zeros([num_xi, num_eta, num_zeta,num_dim])
    for i_dim in range(num_dim):
      vel_0_2D[:,:,i_dim] = arr_conv.array_1D_to_2D(\
        Xi, Eta, num_xi, num_eta, vel_0_1D[:,i_dim])
      vel_0_3D[:,:,:,i_dim] = arr_conv.array_1D_to_3D(\
        Xi, Eta, zeta, num_xi, num_eta, num_zeta, vel_0_1D[:,i_dim])

  
    print(' - Calculation of Jacobian')
    jacobian = curvder.jacobian_of_grid_2d(\
        Xi,
        Eta,
        zeta,
        cell_centroid,
        6)
        #options.rom.jacobian.order_derivatives_x,
        #options.rom.jacobian.order_derivatives_y,
        #options.rom.jacobian.order_derivatives_z)
    print(' - Calculation of ROM matrices')
    (L0_calc, LRe_calc, C0_calc, CRe_calc, Q_calc) = \
        incrom.pod_rom_matrices_2d(\
          Xi,
          Eta,
          zeta,
          cell_centroid,
          num_cell,
          phi,
          weights_ND,
          vel_0_3D,
          jacobian,
          6)
    #       #options.rom.jacobian.order_derivatives_x,
    #       #options.rom.jacobian.order_derivatives_y, 
    #       #options.rom.jacobian.order_derivatives_z)
    (B_calc, B0_calc) = incrom.pod_rom_boundary_matrices_2d(\
      Xi,
      Eta,
      zeta,
      cell_centroid,
      num_cell,
      phi,
      weights_ND,
      vel_0_2D, 
      boundary_vec) 
    #LRe_calc=LRe_calc*(1/17000)
    #CRe_calc=CRe_calc*(1/17000)
    #print(' - Saving matrices to file', rom_matrices_filename)
    #Saving Matrices
    np.savez(rom_matrices_filename,
              L0_calc  = L0_calc,
              LRe_calc = LRe_calc,
              C0_calc  = C0_calc,
              CRe_calc = CRe_calc,
              Q_calc   = Q_calc,
              B_calc = B_calc,
              B0_calc = B0_calc)
    

    #print('Reading matrices from file', rom_matrices_filename)
    matrices = np.load(rom_matrices_filename)
    L0_calc  = matrices['L0_calc']
    LRe_calc = matrices['LRe_calc']
    C0_calc  = matrices['C0_calc']
    CRe_calc = matrices['CRe_calc']
    Q_calc   = matrices['Q_calc']
    B_calc   = matrices['B_calc']
    B0_calc  = matrices['B0_calc']
    
    #Re-calculate boundary matrices
    print('Calculating Boundary Matrices')
    (B_calc, B0_calc) = incrom.pod_rom_boundary_matrices_2d(\
      Xi,
      Eta,
      zeta,
      cell_centroid,
      num_cell,
      phi,
      weights_ND,
      vel_0_2D, 
      boundary_vec) 

    # print('ROM RK45 integration over times', integration_times)
    char_L = 1

        
    #----------------------------------Verification Plots ---------------------------------
    if verify:
     
        plt.imshow(vel_0_2D[:,:,0])
        plt.colorbar(location='right', anchor=(0, 0.3), shrink=0.7)
        plt.title('Mean Velocity')
        plt.show()
        
        # #Check Defined boundary velocities
        # plt.plot(boundary_2D[0,:,0])
        # plt.plot(boundary_2D[-1,:,0])
        # plt.plot(boundary_2D[:,0,0])
        # plt.plot(boundary_2D[:,-1,0])
        # plt.title("Boundary Values in V_Gamma")
        # plt.ylabel("u")
        # plt.legend(['Xi=-1', 'Xi=1', 'Eta=-1', 'Eta=1'])
        # plt.show()
        # plt.plot(boundary_2D[0,:,1])
        # plt.plot(boundary_2D[-1,:,1])
        # plt.plot(boundary_2D[:,0,1])
        # plt.plot(boundary_2D[:,-1,1])
        # plt.title("Boundary Values in V_Gamma")
        # plt.ylabel("v")
        # plt.legend(['Xi=-1', 'Xi=1', 'Eta=-1', 'Eta=1'])
        # plt.show()
        
        # #Check Mean boundary velocities
        # plt.plot(vel_0_boundary_2D[0,:,0])
        # plt.plot(vel_0_boundary_2D[-1,:,0])
        # plt.plot(vel_0_boundary_2D[:,0,0])
        # plt.plot(vel_0_boundary_2D[:,-1,0])
        # plt.title("Boundary Values in V0")
        # plt.ylabel("u")
        # plt.legend(['Xi=-1', 'Xi=1', 'Eta=-1', 'Eta=1'])
        # plt.show()
        # plt.plot(vel_0_boundary_2D[0,:,1])
        # plt.plot(vel_0_boundary_2D[-1,:,1])
        # plt.plot(vel_0_boundary_2D[:,0,1])
        # plt.plot(vel_0_boundary_2D[:,-1,1])
        # plt.title("Boundary Values in V0")
        # plt.ylabel("v")
        # plt.legend(['Xi=-1', 'Xi=1', 'Eta=-1', 'Eta=1'])
        # plt.show()
        
    
    # #Code to run for multiple penalty values
    
    if type(penalty)==np.ndarray:
        #Intialize aT
        aT=np.empty((phi.shape[-1], num_time, len(penalty))) #n_modes x n_times x n_penalties
        RomEnergy=np.empty((num_time,len(penalty)))
        ErrEnergy=np.empty((num_time,len(penalty)))
        for iPenalty in range(len(penalty)):
            # Solve for coeffecients
            print('Solving Coeffecients')
            aT[:,:,iPenalty] = incrom.rom_calc_rk45_boundary(\
                    25000,
                    char_L,
                    L0_calc,
                    LRe_calc,
                    C0_calc,
                    CRe_calc,
                    Q_calc,
                    B_calc,
                    B0_calc,
                    modal_coeff,
                    integration_times,
                    penalty[iPenalty])
        #Compute Reconstructions
        print('Computing Velocities')
        vel_rom = np.tensordot(phi, aT, axes=(1,0))     #num_cell x n_times x n_penalties
        vel_pod = np.matmul(phi, modal_coeff)  #num_cell x n_times
        
        #Bump up true sol and mean dimension
        vel_0 = vel_0.reshape((num_cell*num_dim,1, 1))
        weights_ND = weights_ND.reshape((weights_ND.shape[0], 1, 1))
        vel_true = vel_true.reshape((num_cell*num_dim, num_time, 1))
        #Compute Errors
        print('Computing Errors')
        RomEnergy = np.linalg.norm(vel_rom*weights_ND, axis= 0)
        ErrEnergy = np.linalg.norm((vel_true - (vel_0+vel_rom))*weights_ND, axis= 0)
        TrueEnergy = np.linalg.norm(vel_true*weights_ND, axis=0)
        
        print('Plotting Results')
        #Plot Total Rom Energy
        for iPenalty in range(len(penalty)):
            plt.plot(integration_times, RomEnergy[:,iPenalty], \
                      label=r'$\tau=10^{' + str(int(np.log10(penalty[iPenalty]))) + '}$')
        plt.legend(loc='upper left')
        plt.ylabel('ROM Kinetic Energy')
        plt.savefig(data_folder + 'ROMenergy_m' +str(modes), bbox_inches='tight')
        plt.show()
        
        # for iPenalty in range(len(penalty_exp)-1):        
        #     plt.plot(integration_times, (RomEnergy[:,iPenalty+1]-RomEnergy[:,0])/RomEnergy[:,0], \
        #              label=r'$\tau=10^{' + str(penalty_exp[iPenalty+1]) + '}$')
        # plt.ylabel('Energy Increase Relative to no Penalty')
        # plt.legend(loc='upper left')
        # plt.savefig(data_folder + 'ROMenergyRelative')
        # plt.show()
        
        
        #Plot Rom Error
        for iPenalty in range(len(penalty)):
            plt.plot(integration_times, ErrEnergy[:,iPenalty], \
                      label=r'$\tau=10^{' + str(int(np.log10(penalty[iPenalty]))) + '}$')
        plt.legend(loc='upper left')
        plt.ylabel('ROM Error')
        plt.savefig(data_folder + 'ROMerror_m' +str(modes), bbox_inches='tight')
        plt.show()
        
        #Plot Rom Error
        for iPenalty in range(len(penalty)):
            plt.plot(integration_times, ErrEnergy[:,iPenalty]/TrueEnergy, \
                      label=r'$\tau=10^{' + str(int(np.log10(penalty[iPenalty]))) + '}$')
        plt.legend(loc='upper left')
        plt.ylabel('Relative ROM Error')
        plt.savefig(data_folder + 'ROMerrorRelative_m' +str(modes), bbox_inches='tight')
        plt.show()
        
        
        
        
    
    
    #Code to run for single penalty value
    if type(penalty)==int or type(penalty)== float:
        # Solve for coeffecients
        aT = incrom.rom_calc_rk45_boundary(\
                25000,
                char_L,
                L0_calc,
                LRe_calc,
                C0_calc,
                CRe_calc,
                Q_calc,
                B_calc,
                B0_calc,
                modal_coeff,
                integration_times,
                penalty)
        #Compute Reconstructions
        print('Computing Velocities')
        vel_rom = np.tensordot(phi, aT, axes=(1,0))     #num_cell x n_times x n_penalties
        vel_pod = np.matmul(phi, modal_coeff)  #num_cell x n_times
        
        #Seperate Dimensions
        vel_rom_1D = np.empty((num_cell, num_dim, num_time))
        vel_pod_1D = np.empty((num_cell, num_dim, num_time))
        print(vel_pod_1D.shape)
        print(vel_pod.shape)
        for i_time in range(num_time):
            vel_rom_1D[:,:, i_time]=np.reshape(vel_rom[:, i_time], (num_cell, num_dim), order = 'F')
            vel_pod_1D[:,:, i_time]=np.reshape(vel_pod[:, i_time], (num_cell, num_dim), order = 'F')
    
        #Make 2D
        vel_rom_2D=np.empty((num_xi, num_eta, num_dim, num_time))
        vel_pod_2D=np.empty((num_xi, num_eta, num_dim, num_time))
        for i_dim in range(num_dim):
            for i_time in range(num_time):
              vel_rom_2D[:,:,i_dim, i_time] = arr_conv.array_1D_to_2D(\
                Xi, Eta, num_xi, num_eta, vel_rom_1D[:,i_dim,i_time])
              vel_pod_2D[:,:,i_dim,i_time] = arr_conv.array_1D_to_2D(\
                Xi, Eta, num_xi, num_eta, vel_pod_1D[:,i_dim,i_time])
            
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,0,1]-(vel_0_2D[:,:,0]+vel_rom_2D[:,:,0,0]),
          data_folder + 'u_vel_rom_error_p' + str(int(np.log10(penalty))) + '_t1.png',
          "auto",
          "auto",
          "auto")
    
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,0,100]-(vel_0_2D[:,:,0]+vel_rom_2D[:,:,0,99]),
          data_folder + 'u_vel_rom_error_p' + str(int(np.log10(penalty))) + '_t100_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")
            
            
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,0,-1]-(vel_0_2D[:,:,0]+vel_rom_2D[:,:,0,-1]),
          data_folder + 'u_vel_rom_error_p' + str(int(np.log10(penalty))) + '_t499_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")
    
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,0,1]-(vel_0_2D[:,:,0]+vel_pod_2D[:,:,0,1]),
          data_folder + 'u_vel_pod_error_t1_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")
    
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,0,100]-(vel_0_2D[:,:,0]+vel_pod_2D[:,:,0,100]),
          data_folder + 'u_vel_pod_error_t100_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")
            
            
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,0,-1]-(vel_0_2D[:,:,0]+vel_pod_2D[:,:,0,-1]),
          data_folder + 'u_vel_pod_error_t499_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")
    
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,1,1]-(vel_0_2D[:,:,1]+vel_rom_2D[:,:,1,0]),
          data_folder + 'v_vel_rom_error_p' + str(int(np.log10(penalty))) + '_t1_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")
    
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,1,100]-(vel_0_2D[:,:,1]+vel_rom_2D[:,:,1,99]),
          data_folder + 'v_vel_rom_error_p' + str(int(np.log10(penalty))) + '_t100_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")
            
            
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,1,-1]-(vel_0_2D[:,:,1]+vel_rom_2D[:,:,1,-1]),
          data_folder + 'v_vel_rom_error_p' + str(int(np.log10(penalty))) + '_t498_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")
    
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,1,1]-(vel_0_2D[:,:,1]+vel_pod_2D[:,:,1,1]),
          data_folder + 'v_vel_pod_error_t1_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")
    
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,1,100]-(vel_0_2D[:,:,1]+vel_pod_2D[:,:,1,100]),
          data_folder + 'v_vel_pod_error_t100_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")
            
            
        plot_pcolormesh(\
          Xi_mesh,
          Eta_mesh,
          vel_true_2D[:,:,1,498]-(vel_0_2D[:,:,1]+vel_pod_2D[:,:,1,498]),
          data_folder + 'v_vel_pod_error_t5499_m' + str(modes) + '.png',
          "auto",
          "auto",
          "auto")            
    
    ##-----------------------Compute Energy Error---------------------------------
    #Get Snapshots
    
    #Format for velocity data
    
    #Compute Error Metric
    
    #Plot Results
    


#------------------------------------------------------------------------------
if __name__ == "__main__":
  sys.exit(main())
