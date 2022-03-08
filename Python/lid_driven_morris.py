
import sys
import io
import os
import UQtoolbox as uq
import aerofusion.data.array_conversion as arr_conv
#from aerofusion.rom import incompressible_navier_stokes_rom as incrom
from aerofusion.rom import incompressible_navier_stokes_rom as incrom
import numpy as np
#import logging
import argparse
import libconf
from aerofusion.plot.plot_2D import plot_contour
from aerofusion.plot.plot_2D import plot_pcolormesh
import sys
import matplotlib.pyplot as plt
import scipy.io as mio
from lid_driven_pod_rom import main as lid_driven_pod_rom
import mat73
from lid_driven_pod_rom import normalize_pois
import mpi4py.MPI as MPI
import gc

def main(argv=None):
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    print("Hello from thread " + str(mpi_rank) + ".")
    
    #===============================Run Options================================
    n_modes = 100
    n_samp_morris = 40
    n_snapshot = 150
    t_forward = 2
    l_morris = 1/40
    logging = 2
    
    #rom_matrices_filename="../../lid_driven_penalty/rom_matrices_s500_m" + str(modes) + ".npz"
    save_path = "../../lid_driven_data/morris_screening_s"+str(n_snapshot) + \
        "m" + str(n_modes) + "_"
    data_folder = "../../lid_driven_data/"
    weights_file = "weights_hr.mat"
    velocity_file = "re17000_hr.mat"
    #============================Set POIs and QOIs=============================
    poi_names = np.array(["Re", "boundary exponent mult", "penalty strength", \
                          "basis vort (TL)", "basis orient (TL)", "basis x-location (TL)",
                          "basis y-location (TL)", "basis extent (TL)"])
    #poi_base = np.array([16000, 0, 1, 1, 0, -.75, .75, 1])
    poi_base = np.array([17000, -1.5, 1e-2, 1, 0, -.75, .75, 1])
    #poi_base = np.array([17000, -1.5, 1e-2, 1, -.75, .75])
    poi_ranges = np.array([[16500, 17500], \
                           [-2, -1],\
                           [0, 1e-2],\
                           [.9, 1.1],\
                           [0, np.pi/2],\
                           [-.1, .1]+poi_base[5],\
                           [-.1, .1]+poi_base[6],\
                           [1, 1.5]
                           ])
    # poi_ranges = np.array([[12000, 18000], \
    #                        [-2, 0],\
    #                        [0, 1e2],\
    #                        [.8, 1.2],\
    #                        [0, np.pi],\
    #                        [-.2, .2]+poi_base[5],\
    #                        [-.2, .2]+poi_base[6],\
    #                        [1,2]])
    poi_normalized = normalize_pois(poi_base, poi_ranges)
    poi_normalized_ranges = np.array([[0,1],\
                                      [0,1],\
                                      [0,1],\
                                      [0,1],\
                                      [0,1],\
                                      [0,1],\
                                      [0,1],\
                                      [0,1]]).transpose()
    qoi_names = np.array(["energy", "max vorticity", "min vorticity"])
    #======================Load velocity and discretization data=====================
    if logging:
        print("Loading data")
    integration_times=np.arange(0,.1*n_snapshot*t_forward, .1)
    
    num_dim  = 2
    num_xi   = 258
    num_eta  = 258
    num_zeta = 1
    
    mat = mat73.loadmat(data_folder + velocity_file)
    velocity_unreduced_1D_compact=mat['X'][:,0:n_snapshot]
    del mat
    
    mat2 = mio.loadmat(data_folder + weights_file)
    weights = np.ndarray.flatten(mat2['W'])
    weights_ND = np.repeat(weights.reshape(weights.size,1), 2, axis=1).transpose().flatten()
    
    mat2=mio.loadmat(data_folder + "Xi_hr.mat")
    Xi=np.ndarray.flatten(mat2['Xi'])
    
    mat2=mio.loadmat(data_folder + "Eta_hr.mat")
    Eta=np.ndarray.flatten(mat2['Eta'])
    
    del mat2
    
    centroid_file=np.load(data_folder + "cell_center_high_res.npz")
    cell_centroid=np.zeros((num_xi,num_eta,num_zeta,num_dim))
    cell_centroid[:,:,0,0] = centroid_file['cell_center_x']
    cell_centroid[:,:,0,1] = centroid_file['cell_center_y']
    num_cell = num_xi*num_eta*num_zeta

   
    Zeta=np.zeros((Xi.shape[0],),dtype='int')    
    Xi_mesh=Xi.reshape((num_eta, num_xi))
    Eta_mesh=Eta.reshape((num_eta, num_xi))
    Xi_mesh = (Xi_mesh- (num_xi-1)/2)/(num_xi/2)
    Eta_mesh = (Eta_mesh- (num_eta-1)/2)/(num_eta/2)

    #Convert to meshes for Xi and Eta
    
    #Formulate discretization into dictionary for use in executing function
    discretization = {"Xi":  Xi, 
                      "Eta": Eta, 
                      "Zeta": Zeta, 
                      "Xi_mesh": cell_centroid[:,:,0,0],
                      "Eta_mesh": cell_centroid[:,:,0,1],
                      "cell_centroid": cell_centroid,
                      "num_cell": num_cell,
                      "weights_ND": weights_ND
                  }
    gc.collect()
    #========================Setup Executing function==========================
    eval_fcn = lambda poi_normalized : lid_driven_pod_rom(\
                    poi_normalized, poi_names, qoi_names, poi_ranges, n_modes,\
                    discretization, velocity_unreduced_1D_compact, integration_times)
    
    if logging:
        print("Setting up model")
    model = uq.Model(eval_fcn = eval_fcn,
                      base_poi = poi_normalized,
                      dist_type = "uniform",
                      dist_param = poi_normalized_ranges,
                      name_poi = poi_names,
                      name_qoi = qoi_names
                      )

    #Set options
    uqOptions = uq.Options()
    uqOptions.lsa.run=False
    uqOptions.lsa.run_param_subset = False
    uqOptions.gsa.run=True
    uqOptions.gsa.run_morris = True
    uqOptions.gsa.run_sobol = False
    uqOptions.gsa.n_samp_morris = n_samp_morris
    uqOptions.gsa.l_morris = l_morris
    uqOptions.save = True
    uqOptions.path = save_path
    uqOptions.display = True
    uqOptions.print= True
    uqOptions.path = data_folder + "sensitivity/s" + str(n_snapshot) + "m" + str(n_modes) + "_l" + str(int(1/l_morris)) + "_"

    #Run SA
    print("Running Sensitivity Analysis")
    results=uq.run_uq(model, uqOptions, logging= logging)
    
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


if __name__ == "__main__":
    sys.exit(main())

