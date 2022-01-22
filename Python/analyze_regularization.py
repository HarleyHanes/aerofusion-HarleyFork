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
from compute_vorticity import Vorticity_2D

def main(argv=None):
    # Run Settings
    penalty = 10.0 ** np.array([-15, -2, 4, 6])
    alpha = np.array([1, .5, .1, .01])
    tmax= 150
    snapshots=500
    modes = 50
    fig_size=(20,16)
    plot_data = "vorticity"
    plot_style =  "heat"   #stream heat
    
    #Define file locations
    data_folder = "../../lid_driven_snapshots/"
    pod_filename = "pod_lid_driven_50.npz"
    plot_folder = "../../lid_driven_snapshots/extended_analysis/"
    rom_filename="../../lid_driven_penalty/rom_matrices_s500_m50.npz"
    
    # Define Integration Times
    integration_times = np.arange(.1,tmax+.1,.1,)
    
    #--------------------------------------- Load Data
    pod_data = np.load(data_folder + pod_filename)
    vel_0           = pod_data['velocity_mean']
    phi             = pod_data['phi']
    modal_coeff     = pod_data['modal_coeff']
    
    
    matrices = np.load(rom_filename)
    L0_calc  = matrices['L0_calc']
    LRe_calc = matrices['LRe_calc']
    C0_calc  = matrices['C0_calc']
    CRe_calc = matrices['CRe_calc']
    Q_calc   = matrices['Q_calc']
    B_calc   = matrices['B_calc']
    B0_calc  = matrices['B0_calc']
    
    
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
    num_cell = num_xi*num_eta
    base_vec = np.linspace(-1,1,num = num_xi)
    zeta=np.zeros((Xi.shape[0],),dtype='int')
    
    #--------------------------------------Reformat Data
    #Location indices
    Xi_mesh=Xi.reshape((num_eta, num_xi))
    Eta_mesh=Eta.reshape((num_eta, num_xi))
    Xi_mesh = (Xi_mesh- (num_xi-1)/2)/(num_xi/2)
    Eta_mesh = (Eta_mesh- (num_eta-1)/2)/(num_eta/2)
    
    #Weights
    #Upsample for 2D
    weights_ND = np.zeros([num_cell*num_dim])
    
    #Seperate velocity components
    vel_0_1D = np.reshape(vel_0, (num_cell, num_dim), order= 'F')
    
    vel_0_2D = np.zeros([num_xi, num_eta, num_dim])
    for i_dim in range(num_dim):
      weights_ND[i_dim*num_cell : (i_dim+1)*num_cell] = weights
      vel_0_2D[:,:,i_dim] = arr_conv.array_1D_to_2D(\
          Xi, Eta, num_xi, num_eta, vel_0_1D[:,i_dim])
    
    # Loop Through Penalty Cases
    for iPenalty in range(len(penalty)):
        # Make Figure
        fig = plt.figure(figsize = fig_size)
        # Loop Through Boundary Cases
        for iAlpha in range(len(alpha)):
            #Define boundary
            boundary_vec=(np.abs(1-base_vec)**(2*alpha[iAlpha]))*(np.abs(1+base_vec)**(2*alpha[iAlpha]))
            # Compute Penalty Rom Matrices
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
            # Solve for modal coeffecients
            aT = incrom.rom_calc_rk45_boundary(\
                    25000,
                    1,
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
            # Construct Velocity
            vel_rom = np.matmul(phi, aT)     #num_cell x n_times
            #Reformat to 2D
            vel_rom_1D = np.reshape(vel_rom[:,-1], (num_cell, num_dim), order= 'F')
            vel_rom_2D = np.zeros([num_xi, num_eta, num_dim])
            for i_dim in range(num_dim):
              vel_rom_2D[:,:,i_dim] = arr_conv.array_1D_to_2D(\
                  Xi, Eta, num_xi, num_eta, vel_rom_1D[:,i_dim])
            #------------------Identifty plotted data--------------------------
            if plot_data.lower() == "vorticity reduced":
                data = Vorticity_2D(vel_rom_2D)
            if plot_data.lower() == "u reduced":
                data = vel_rom_2D[:,:,0]
            if plot_data.lower() == "v reduced":
                data = vel_rom_2D[:,:,1]
            if plot_data.lower() == "vorticity":
                data = Vorticity_2D(vel_0_2D + vel_rom_2D)
            if plot_data.lower() == "u":
                data = vel_0_2D[:,:,0] + vel_rom_2D[:,:,0]
            if plot_data.lower() == "v":
                data = vel_0_2D[:,:,1] + vel_rom_2D[:,:,1]
            #------------------------Plot Data ----------------------------
            # Make Subplot
            if iPenalty == 0:
                ax = fig.add_subplot(int(200+np.ceil(len(alpha)/2)*10+iAlpha+1))
            else: 
                ax = fig.add_subplot(int(200+np.ceil(len(alpha)/2)*10+iAlpha+1),\
                                    sharex=ax, sharey= ax)
            if plot_style.lower() == "heat":
                im = ax.pcolormesh(\
                             Xi_mesh,
                             Eta_mesh,
                             data,
                             cmap = "jet",
                             vmin= -np.max(np.abs(data)),
                             vmax= np.max(np.abs(data)))
                fig.colorbar(im, label = plot_data)
            elif plot_style.lower() == "stream":
                ax.streamplot(Xi_mesh, Eta_mesh, vel_rom_2D[:,:,0], vel_rom_2D[:,:,1],\
                              density = [.5, 1])
            if iAlpha%2 == 0:
                ax.set_ylabel("y")
            if (iAlpha == len(penalty)-1) + (iAlpha == len(penalty)-2):
                ax.set_xlabel("x")
            ax.set_title("a=" + str(alpha[iAlpha]))
        #save figure
        plt.savefig(plot_folder + "/extended_boundary_" + plot_style.lower() + "_s" 
                    + str(snapshots) + "m" + str(modes) + "t"  + str(tmax) + \
                    "_" + str(plot_data) + "_penalty=" + \
                    str(int(np.log10(penalty[iPenalty]))) +".png")
    
    
    
    
    
    
if __name__ == "__main__":
    sys.exit(main())
