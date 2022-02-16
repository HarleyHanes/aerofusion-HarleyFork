
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
from boundary_rom import BoundaryPenaltyROM as RunROM
def main(argv=None):
    QOI_type = "vorticity"
    POI_type= "alpha"   #unreduced
    alpha_value = np.array([.01])
    modes = 100
    res = "high"
    method = "art"
    #penalty=10.0**4 penalty defined explicitly in function
    tmax=100
    penalty_exp = 2
    
    #rom_matrices_filename="../../lid_driven_penalty/rom_matrices_s500_m" + str(modes) + ".npz"
    save_path = "../../lid_driven_data/" + method +"_" + QOI_type + "_a" + str(alpha_value[0]) + "_s500m" + str(modes) + "_"
    #data_folder = "../../lid_driven_snapshots/"
    #pod_filename = 'pod_lid_driven_50.npz'
    data_folder = "../../lid_driven_data/"
    pod_filename = "pod_Re17000hr_mean_s500m100.npz"
    
    pod_data = np.load(data_folder + pod_filename)
    # Assign data to convenience variables
    vel_0  = pod_data['velocity_mean']
    simulation_time = pod_data['simulation_time']
    phi             = pod_data['phi']
    modal_coeff     = pod_data['modal_coeff']
    
    #curtail phi and modal_coeff
    phi=phi[:,0:modes]
    modal_coeff=modal_coeff[0:modes]
    
    #Specify when integration takes place 
    integration_times = np.arange(.1,tmax,.1)
    integration_indices = np.arange(1,len(integration_times)+1)
    #integration_indices = np.arange(1,500)
    #integration_times = simulation_time[integration_indices]
    num_time = len(integration_times)
    if res.lower() == "low":
        mat2=mio.loadmat(data_folder + "w_LowRes.mat")
        weights=np.ndarray.flatten(mat2['w'])
        mat2=mio.loadmat(data_folder + "Xi_lr.mat")
        Xi=np.ndarray.flatten(mat2['Xi'])
        mat2=mio.loadmat(data_folder + "Eta_lr.mat")
        Eta=np.ndarray.flatten(mat2['Eta'])
        centroid_file=np.load(data_folder + "cell_center_low_res.npz")
    
        num_dim  = 2
        num_xi   = 130
        num_eta  = 130
        num_zeta = 1
    elif res.lower() == "high":
        rom_matrices_filename="../../lid_driven_data/rom_Re17000hr_" + method + "_s500m" + str(modes) + ".npz"
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
    
   
    Xi=Xi[0:(num_xi*num_eta)]
    Eta=Eta[0:(num_xi*num_eta)]
    weights = weights[0:(num_xi*num_eta)]
    cell_centroid=np.zeros((num_xi,num_eta,num_zeta,num_dim))
    cell_centroid[:,:,0,0] = centroid_file['cell_center_x']
    cell_centroid[:,:,0,1] = centroid_file['cell_center_y']
    num_cell = num_xi*num_eta*num_zeta

   
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
    
    
    #Save ROM requirements into dictionaries
    discretization = {\
                      "Xi": Xi,
                      "Eta": Eta,
                      "Zeta": Zeta,
                      "cell_centroid": cell_centroid,
                      "num_cell":num_cell,
                      "weights_ND": weights_ND
        }
    pod = {\
           "phi": phi,
           "modal_coeff": modal_coeff,
           "vel_0_2D": vel_0_2D,
           "reynolds_number": 17000}
    
    x_boundary = np.linspace(-1,1, num_xi)
    x_boundary_reduced = np.array([0, 50, 128, 150])#np.arange(0,num_xi,50)
    boundary_vec_base = (abs(x_boundary-1)**(2*alpha_value))*(abs(x_boundary+1)**(2*alpha_value))
    boundary_vec_reduced = boundary_vec_base[x_boundary_reduced]
    
    
    
    
    
    #Define evalFunction
    evalFcn=lambda POI: RunROM(POI , QOI_type, rom_matrices_filename, integration_times, \
                               discretization, pod)
    evalFcnReduced = lambda POI: RunROMreduced(POI, x_boundary_reduced, QOI_type,\
                                               rom_matrices_filename, integration_times,\
                                               discretization, pod)
    evalFcnExtended = lambda POI:RunROMextended(POI, x_boundary, QOI_type,\
                                                rom_matrices_filename, integration_times,\
                                                discretization, pod,10**penalty_exp)
    #Define model structure
    if POI_type.lower() == 'reduced':
        model=uq.model(evalFcn=evalFcnReduced,
                        basePOIs=boundary_vec_reduced,
                        POInames=x_boundary_reduced.astype(str)
                        )
    elif POI_type.lower () == 'alpha':
        model = uq.model(evalFcn = evalFcnExtended,
                         basePOIs = alpha_value,
                         POInames = np.array(["alpha"]))
                
        
    else:
        model=uq.model(evalFcn=evalFcn,
                        basePOIs=boundary_vec_base,
                        POInames=x_boundary.astype(str)
                        )
    #Set options
    uqOptions = uq.uqOptions()
    uqOptions.lsa.run=True
    uqOptions.lsa.runActiveSubspace = False
    uqOptions.gsa.run=False
    uqOptions.lsa.method='finite'
    uqOptions.lsa.xDelta=10**(-8)
    uqOptions.save = True
    uqOptions.path = save_path
    uqOptions.display=False 
    uqOptions.print= False

    #Run SA
    print("Running Sensitivity Analysis")
    results=uq.RunUQ(model, uqOptions)
    
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


def RunROMreduced(POIs, POI_position, QOIselector,rom_matrices_filename, integration_times, \
           discretization, pod):
    #Rebuild full boundary vec
    xi_boundary = np.linspace(-1,1, discretization["cell_centroid"].shape[0])
    if POIs.ndim == 1:
        xi_boundary = np.linspace(-1,1, discretization["cell_centroid"].shape[0])
        boundary_vec = (abs(1+xi_boundary)**2)*(abs(1-xi_boundary))**2
        boundary_vec[POI_position]=POIs
    else :
        boundary_vec= np.empty((POIs.shape[0],len(xi_boundary)))
        for i in range(POIs.shape[0]):
            boundary_vec[i]=xi_boundary
            boundary_vec[i][POI_position]=POIs
    
    return RunROM(boundary_vec, QOIselector,rom_matrices_filename, integration_times, \
               discretization, pod)
    
def RunROMextended(POI, x_boundary, QOI_type,rom_matrices_filename, \
                   integration_times, discretization, pod, penalty_strength):
    boundary_vec = (np.abs(1-x_boundary)**(2*POI))*(np.abs(1+x_boundary)**(2*POI))
    return RunROM(boundary_vec, QOI_type, rom_matrices_filename, integration_times, \
               discretization, pod, penalty_strength)


if __name__ == "__main__":
    sys.exit(main())

