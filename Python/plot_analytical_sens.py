
import sys
import io
import os
import UQtoolbox as uq
import aerofusion.data.array_conversion as arr_conv
#from aerofusion.rom import incompressible_navier_stokes_rom as incrom
from aerofusion.rom import incompressible_navier_stokes_rom as incrom
import numpy as np
import logging
import argparse
import libconf
from aerofusion.plot.plot_2D import plot_contour
from aerofusion.plot.plot_2D import plot_pcolormesh
import sys
import matplotlib.pyplot as plt
import scipy.io as mio

def main(argv=None):
    #Run settings 
    alpha = 10**(np.linspace(-3, 0, 30))
    penalty = [10 **(-12), 10**(-2), 10**(4)]
    QOI_type = "kinetic energy"
    tmax=50
    #File Paths
    rom_matrices_filename="../../lid_driven_penalty/rom_matrices_50.npz"
    data_folder = "../../lid_driven_snapshots/"
    plot_folder = data_folder + 'extended_analysis/'
    pod_data = np.load(data_folder + 'pod_lid_driven_50.npz')

    
    # Assign data to convenience variables
    vel_0  = pod_data['velocity_mean']
    #simulation_time = pod_data['simulation_time']
    phi             = pod_data['phi']
    modal_coeff     = pod_data['modal_coeff']
    
    #Specify when integration takes place 
    integration_times = np.arange(.1,tmax,.1)
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
    Zeta=np.zeros((Xi.shape[0],),dtype='int')
    
    
    #Reformat Velocity and weights
    vel_0_1D = np.reshape(vel_0, (num_cell, num_dim), order= 'F')
    vel_0_2D = np.zeros([num_xi, num_eta, num_dim])
    vel_0_3D = np.zeros([num_xi, num_eta, num_zeta,num_dim])
    for i_dim in range(num_dim):
      vel_0_2D[:,:,i_dim] = arr_conv.array_1D_to_2D(\
        Xi, Eta, num_xi, num_eta, vel_0_1D[:,i_dim])
      vel_0_3D[:,:,:,i_dim] = arr_conv.array_1D_to_3D(\
        Xi, Eta, Zeta, num_xi, num_eta, num_zeta, vel_0_1D[:,i_dim])
    
    
    weights_ND = np.zeros([num_cell*num_dim])
    for i_dim in range(num_dim):
      weights_ND[i_dim*num_cell : (i_dim+1)*num_cell] = weights
    
    

    for iPoint in range(len(points)):
        # Get index of point
        lindex = 
        # Extract Phi_l and weightL
        phiL = phi[lindex,:]
        weightL = weight[lindex,:]
        # Compute Sensitivity
        Sens= t*np.matmul(phi, phiL*weightL)
        #transform to mesh form
        Sens_1D = np.emtpy((num_cell, num_dim)
        Sens_2D = np.empty((num_xi, num_eta, num_dim))
        Sens_1D = np.reshape(phi, (num_cell, num_dim), order = 'F')
        for i_dim in range(num_dim):
            Sens_2D[:,:,i_dim] = arr_conv.array_1D_to_3D(\
             Xi, Eta, Zeta, num_xi, num_eta, num_zeta, Sens_1D[:,i_dim])
        #Plot Mesh
        
    #Run SA
    print("Running Sensitivity Analysis")
    
    #Compute QOIs
    for iPenalty in range(len(penalty)):
        for iAlpha in range(len(alpha)):
            #Compute Boundary
            boundary_vec = (abs(x_boundary-1)**(2*alpha[iAlpha]))*(abs(x_boundary+1)**(2*alpha[iAlpha]))
            #Get QOIs
            QOI = evalFcn(boundary_vec, penalty[iPenalty])
            if iAlpha == 0:
                if isinstance(QOI, np.ndarray):
                    Results = np.empty((len(penalty),)+(len(alpha),)+QOI[0].shape)
                else:
                    Results = np.empty((len(penalty),)+(len(alpha),))
            Results[iPenalty][iAlpha] = QOI
            
    #Plot Energy
    plt.figure(figsize=(10,8), dpi =150)
    for iPenalty in range(len(penalty)):
        plt.semilogx(alpha,Results[iPenalty], label=r'$\tau=10^{' + str(int(np.log10(penalty[iPenalty]))) + '}$')
    plt.legend()
    plt.xlabel('a')
    plt.ylabel('Kinetic Energy at t=50')
    #plt.show()
    plt.savefig(plot_folder + "KineticEnergy.png", bbox_inches='tight') 
    
'''
    #Reshape Sensitivities
    if QOI_type.lower()=='full data':
        muStar3D=arr_conv.array_1D_to_3D(xi, eta, zeta, num_cell, results.gsa.muStar.squeeze())
        sigma3D=np.sqrt(arr_conv.array_1D_to_3D(xi, eta, zeta, num_cell, results.gsa.sigma2.squeeze()))
        base3D=arr_conv.array_1D_to_3D(xi, eta, zeta, num_cell, model.baseQOIs)
    if QOI_type.lower()=='modal coeff':
        muStar3D=results.gsa.muStar.reshape((options.pod["num_modes"], len(integration_times)))
        sigma3D=np.sqrt(results.gsa.sigma2.reshape((options.pod["num_modes"], len(integration_times))))
        base3D=model.baseQOIs.reshape((options.pod["num_modes"], len(integration_times)))
    print('Raw Shape:' + str(results.gsa.muStar.shape))
    print('Shifted Shape:' + str(muStar3D.shape))
    print('Max Raw Results:' + str(np.max(results.gsa.muStar)))
    #print('Local Sensitivities:' + str(np.max(results.lsa.jac)))
    #print('Assessment Times:' + str(integration_times))
    print('Max Sensitivitiy:' + str(np.max(muStar3D)))
    print('Base Results: ' + str(np.max(base3D)) +', ' + str(np.mean(base3D)))
    print('Base Results Raw: ' + str(np.max(model.baseQOIs)) +', ' + str(np.mean(model.baseQOIs)))
    #Save sensitivities
    #Save sensitivities
    if hasattr(options.uq, 'save'):
        if hasattr(options.uq.save, 'morris_mean_filename'):
            np.save(options.uq.save.morris_mean_filename, muStar3D)
        if hasattr(options.uq.save, 'morris_sigma_filename'):
            np.save(options.uq.save.morris_sigma_filename, sigma3D)
        if hasattr(options.rom,'save_filename'):
            np.save(options.rom.save_filename, base3D)

    #Plot Morris Indices
    if (hasattr(options.uq,'plot'))&(QOI_type.lower()=='full data'):
        plot_pcolormesh(
            cell_centroid[:,:,0,0],
            cell_centroid[:,:,0,1],
            muStar3D[:, :, 0],
            options.uq.plot.plot_prefix + 'reconst_mu' + str(num_modes)+'.eps',
            sigma3D[:, :, 0],            
            fig_size=(24,7),
            font_size=23,
            vmin="auto",
            vmax="auto",
            cmap="jet",
            colorbar_label="$\mu^*_{Re}$",
            xlabel="$x/L$",
            ylabel="$y/L$",
            title="Streamline Velocity Mean Sensitivity to Reynolds Number")
        plot_pcolormesh(
            cell_centroid[:, :, 0, 0],
            cell_centroid[:, :, 0, 1],
            sigma3D[:, :, 0],
            options.uq.plot.plot_prefix + 'reconst_sigma' + str(num_modes)+'.eps',
            fig_size=(24, 7),
            font_size=23,
            vmin="auto",
            vmax="auto",
            cmap="jet",
            colorbar_label="$\sigma_{Re}$",
            xlabel="$x/L$",
            ylabel="$y/L$",
            title="Streamline Velocity Sensitivity Standard Deviation for Reynolds Number")
        plot_pcolormesh(
            cell_centroid[:, :, 0, 0],
            cell_centroid[:, :, 0, 1],
            base3D[:, :, 0],
            options.uq.plot.plot_prefix + 'mean_reduced' + str(num_modes)+'.eps',
            fig_size=(24, 7),
            font_size=23,
            vmin="auto",
            vmax="auto",
            cmap="jet",
            colorbar_label="$\sigma_{Re}$",
            xlabel="$x/L$",
            ylabel="$y/L$",
            title="Streamline Velocity Sensitivity Standard Deviation for Reynolds Number")
     
    if (hasattr(options.uq,'plot'))&(QOI_type.lower()=='modal coeff'):
        #plotted_modes=[0,1,4,9,18,24,32,48,49]
        plotted_modes=[0,1,8,9,18,19,23,24]
        for i in range(len(plotted_modes)):
            mode=plotted_modes[i]
            plt.figure(figsize=(3,2), dpi =150)
            plt.plot(integration_times, muStar3D[mode,:])
            plt.title("Mean Sensitivity")
            plt.ylabel("Mode " + str(mode+1))
            plt.xlabel("Time")
            plt.savefig(options.uq.plot.plot_prefix + "Mode_" + str(mode+1) + "sens.eps", format="eps", bbox_inches='tight')
            plt.savefig(options.uq.plot.plot_prefix + "Mode_" + str(mode+1) + "sens.pdf", bbox_inches='tight')
            plt.savefig(options.uq.plot.plot_prefix + "Mode_" + str(mode+1) + "sens.png", bbox_inches='tight')
            
            plt.figure(figsize=(3,2), dpi =150)
            plt.plot(integration_times, base3D[mode,:])
            plt.title("Integrated Mode Value")
            plt.ylabel("Mode " + str(mode+1))
            plt.xlabel("Time")
            plt.savefig(options.uq.plot.plot_prefix + "Mode_" + str(mode+1) + "base.eps", format="eps", bbox_inches='tight')
            plt.savefig(options.uq.plot.plot_prefix + "Mode_" + str(mode+1) + "base.pdf", bbox_inches='tight')
            plt.savefig(options.uq.plot.plot_prefix + "Mode_" + str(mode+1) + "base.png", bbox_inches='tight')
             
            plt.figure(figsize=(3,2), dpi =150)
            plt.plot(integration_times, sigma3D[mode,:])
            plt.title("Sensitivity Std")
            plt.ylabel("Mode " + str(mode+1))
            plt.xlabel("Time")
            plt.savefig(options.uq.plot.plot_prefix + "Mode_" + str(mode+1) + "sigma.eps", format="eps", bbox_inches='tight')
            plt.savefig(options.uq.plot.plot_prefix + "Mode_" + str(mode+1) + "sigma.pdf", bbox_inches='tight')
            plt.savefig(options.uq.plot.plot_prefix + "Mode_" + str(mode+1) + "sigma.png", bbox_inches='tight')
'''



def RunROM(POIs, QOIselector,rom_matrices_filename, integration_times, \
           discretization, pod, penalty):
    print("Running ROM")
    #Load ROM data
    #print('Reading matrices from file', rom_matrices_filename)
    
    #Unpack discretization
    Xi = discretization["Xi"]
    Eta = discretization["Eta"]
    Zeta = discretization["Zeta"]
    cell_centroid = discretization["cell_centroid"]
    num_cell = discretization["num_cell"]
    weights_ND = discretization["weights_ND"]
    
    #Unpack pod
    phi = pod["phi"]
    modal_coeff = pod["modal_coeff"]
    vel_0_2D = pod["vel_0_2D"]
    reynolds_number = pod["reynolds_number"]
    
    #Unpack POIs
    boundary_vec = POIs
    
    #Load ROM matrices
    matrices = np.load(rom_matrices_filename)
    L0_calc  = matrices['L0_calc']
    LRe_calc = matrices['LRe_calc']
    C0_calc  = matrices['C0_calc']
    CRe_calc = matrices['CRe_calc']
    Q_calc   = matrices['Q_calc']


    char_L = 1
    #print(reynolds_number)
    #Calculate ROM solution
    #Initialize matrices
    if boundary_vec.ndim == 2: 
        for iSamp in range(boundary_vec.shape[0]):
        
            #Compute Boundary ROM matrices
            (B_calc, B0_calc) = incrom.pod_rom_boundary_matrices_2d(\
              Xi,
              Eta,
              Zeta,
              cell_centroid,
              num_cell,
              phi,
              weights_ND,
              vel_0_2D, 
              boundary_vec[iSamp])
            #print('Reynolds Number before rom_calc_rk45:' + str(reynolds_number[iSamp]))
            aT = incrom.rom_calc_rk45_boundary(\
                    reynolds_number,
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
            
           # print('Modal Coefficients at t=0'+str(aT[:,0]))
           # print('Modal Coeffecients at t=tmax'+str(aT[:,-1]))
           # print('Max mean_reduced_velocity:'+str(np.max(mean_reduced_velocity_rom)))
           # print('Mean mean_reduced_velocity:'+ str(np.mean(mean_reduced_velocity_rom)))
           # print()
    
            #Extract QOIs
            if QOIselector.lower()=="full data":
                #print(str(np.max(mean_reduced_velocity_rom)) + ', ' + str(np.mean(mean_reduced_velocity_rom)))       
               # QOIs[iSamp,:]=mean_reduced_velocity_rom[:,-1]
                mean_reduced_velocity_rom = np.matmul(phi, aT)
                QOIs.append(mean_reduced_velocity_rom[:,-1])
                del mean_reduced_velocity_rom
            if QOIselector.lower()=="modal coeff":
                QOIs.append(aT.flatten())
            if QOIselector.lower() =="kinetic energy":
                energy = np.sum(aT, axis = 0)
                energy = energy[-1]
                QOIs.append(energy)
    elif boundary_vec.ndim == 1:
        #Compute Boundary ROM matrices
        (B_calc, B0_calc) = incrom.pod_rom_boundary_matrices_2d(\
          Xi,
          Eta,
          Zeta,
          cell_centroid,
          num_cell,
          phi,
          weights_ND,
          vel_0_2D, 
          boundary_vec)
        #print('Reynolds Number before rom_calc_rk45:' + str(reynolds_number[iSamp]))
        aT = incrom.rom_calc_rk45_boundary(\
                reynolds_number,
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
        
       # print('Modal Coefficients at t=0'+str(aT[:,0]))
       # print('Modal Coeffecients at t=tmax'+str(aT[:,-1]))
       # print('Max mean_reduced_velocity:'+str(np.max(mean_reduced_velocity_rom)))
       # print('Mean mean_reduced_velocity:'+ str(np.mean(mean_reduced_velocity_rom)))
       # print()

        #Extract QOIs
        if QOIselector.lower()=="full data":
            #print(str(np.max(mean_reduced_velocity_rom)) + ', ' + str(np.mean(mean_reduced_velocity_rom)))       
           # QOIs[iSamp,:]=mean_reduced_velocity_rom[:,-1]
            mean_reduced_velocity_rom = np.matmul(phi, aT)
            QOIs= mean_reduced_velocity_rom[:,-1].flatten()
            del mean_reduced_velocity_rom
        if QOIselector.lower()=="modal coeff":
            QOIs = aT.flatten()
            #QOIs=aT[:,-1]
        if QOIselector.lower() =="kinetic energy":
            energy = np.sum(aT, axis = 0)
            energy = energy[[-1]]
            QOIs = energy
    else:
        raise Exception("More than 2 dimensions detected for boundary vector")
    return QOIs.flatten()

if __name__ == "__main__":
    sys.exit(main())

