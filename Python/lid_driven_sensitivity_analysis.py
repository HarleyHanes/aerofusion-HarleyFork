
import sys
import io
import os
import UQLibrary as uq
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
    #Fixed arguments
    n_samp_morris = 40
    
    l_morris = 1/40
    logging = 1
    
    decomp_method = 'svd'
    #Parse Arguments 
    if "x_delta" in sys.argv:
        index = sys.argv.index("x_delta")
        x_delta = float(sys.argv[index+1])
        print("x_delta: " + str(x_delta))
    else:
        x_delta = 1e-6
        
    if "deriv_method" in sys.argv:
        index = sys.argv.index("deriv_method")
        deriv_method= str(sys.argv[index+1])
        print("deriv_method: " + str(deriv_method))
    else:
        deriv_method = 'finite'
        
    if "algorithm" in sys.argv:
        index = sys.argv.index("algorithm")
        algorithm = str(sys.argv[index+1])
        print("algorithm: " + str(algorithm))
        
    else:
        algorithm =  'rrqr'     
        
    if "rel_tol" in sys.argv:
        index = sys.argv.index("rel_tol")
        tolerance = float(sys.argv[index+1])
        print("rel_tol: "+ str(tolerance))
    else:
        tolerance = 1e-7
        
    if "n_modes" in sys.argv:
        index = sys.argv.index("n_modes")
        n_modes = int(sys.argv[index+1])
        print("n_modes: " + str(n_modes))
    else:
        n_modes = 100
        
    if "n_snapshot" in sys.argv:
        index = sys.argv.index("n_snapshot")
        n_snapshot = int(sys.argv[index+1])
        print("n_snapshot: " + str(n_snapshot))
    else:
        n_snapshot= 150
        
    if "t_forward" in sys.argv:
        index = sys.argv.index("t_forward")
        t_forward = float(sys.argv[index+1])
        print("t_forward: " + str(t_forward))
    else:
        t_forward = 1
        
    if "run_morris" in sys.argv:
        run_morris = True
    else:
        run_morris = False
        
    if "run_pss" in sys.argv:
        run_pss = True
    else:
        run_pss = False
    
    if "test_reduction" in sys.argv:
        test_reduction = True
    else:
        test_reduction = False
    
    if "reduction_nsamp" in sys.argv:
        index = sys.argv.index("reduction_nsamp")
        reduction_nsamp = int(sys.argv[index+1])
        print("reduction_nsamp: " + str(reduction_nsamp))
    else:
        reduction_nsamp = 200  #integrated measures
        
    if "inactive_pois" in sys.argv:
        index = sys.argv.index("inactive_pois")
        inactive_pois = np.array(sys.argv[index+1:])
        print("inactive_pois: " + str(inactive_pois))
    else:
        inactive_pois = "null" 
        
        
    if "qoi_set" in sys.argv:
        index = sys.argv.index("qoi_set")
        qoi_set = str(sys.argv[index+1])
        print("qoi_set: " + str(qoi_set))
    else:
        qoi_set = "full velocity"  #integrated measures
        
    
    
    
    #rom_matrices_filename="../../lid_driven_penalty/rom_matrices_s500_m" + str(modes) + ".npz"
    save_path = "../../lid_driven_data/morris_screening_s"+str(n_snapshot) + \
        "m" + str(n_modes) + "_"
    data_folder = "../../lid_driven_data/"
    weights_file = "weights_hr.mat"
    velocity_file = "re17000_hr.mat"
    #============================Set POIs and QOIs=============================
    poi_names = np.array(["Re", "boundary exponent mult", "penalty strength exp", \
                          "basis speed (TL)", "basis speed (BL)", "basis speed (BR)", \
                          "basis speed (C1)", "basis speed (C2)", "basis speed (S1)", \
                          "basis speed (S2)", \
                          "basis orient (TL)", "basis orient (BL)", "basis orient (BR)", \
                          "basis orient (C1)", "basis orient (C2)", "basis orient (S1)", \
                          "basis orient (S2)", \
                          "basis x-location (TL)", "basis x-location (BL)", "basis x-location (BR)", \
                          "basis x-location (C1)", "basis x-location (C2)", "basis x-location (S1)", \
                          "basis x-location (S2)", \
                          "basis y-location (TL)", "basis y-location (BL)", "basis y-location (BR)", \
                          "basis y-location (C1)", "basis y-location (C2)", "basis y-location (S1)", \
                          "basis y-location (S2)", \
                          "basis extent (TL)", "basis extent (BL)", "basis extent (BR)", \
                          "basis extent (C1)", "basis extent (C2)", "basis extent (S1)", \
                          "basis extent (S2)"])
    if qoi_set.lower() == "integrated measures":
        qoi_names = np.array(["energy", "vorticity", \
                              "local vorticity 0", "local vorticity 1", "local vorticity 2",\
                              "local vorticity 3", "local vorticity 4", "local vorticity 5", \
                              "local vorticity 6"])
    elif qoi_set.lower() == "full velocity":
        qoi_names = np.array(["full velocity"])
    #poi_base = np.array([16000, 0, 1, 1, 0, -.75, .75, 1])
    # poi_base = np.array([20000, -1.5, 0,\
    #                      1, .5, .5, .1, .1, .1, .1, \
    #                      0, 3*np.pi/4, np.pi/4, 0, 0, 0, 0,\
    #                      -.75, -.75, .75, -.2, .2, .5, .75, \
    #                      .75, -.75, -.75, -.2, .2, .5, .75, 
    #                      1, 1.5, 1.5, 1, 1, 1, 1])
    poi_base = np.array([17000, -1.5, -2,\
                          .95, .5, .5, .1, .1, .1, .1, \
                          0, 3*np.pi/4, np.pi/4, 0, 0, 0, 0,\
                          -.75, -.75, .75, -.2, .2, .5, .75, \
                          .75, -.75, -.75, -.2, .2, .5, .75, 
                          1, 1.5, 1.5, 1, 1, 1, 1])
    #initialize ranges with pm .25 of base values
    poi_ranges = np.array([poi_base*.75, poi_base*1.25]).transpose()
    #Set alternate Reynolds range
    poi_ranges[0] = np.array([11000, 20000])
    poi_ranges[1] = np.array([-2, 0])
    poi_ranges[2] = np.array([-12, 0])
    #Set all axis angles to (0, pi) except those that are ovular by assumption (BL and BR)
    poi_ranges[10:17] = np.array([[0, np.pi], [np.pi/2, np.pi], [0, np.pi/2], [0, np.pi], [0, np.pi], [0, np.pi], [0, np.pi]])
    
    center_locations = np.array([poi_base[17:24], poi_base[24:31]]).transpose()
    # poi_ranges = np.array([[12000, 18000], \
    #                        [-2, 0],\
    #                        [0, 1e2],\
    #                        [.8, 1.2],\
    #                        [0, np.pi],\
    #                        [-.2, .2]+poi_base[5],\
    #                        [-.2, .2]+poi_base[6],\
    #                        [1,2]])
    poi_normalized = normalize_pois(poi_base, poi_ranges)
    poi_normalized_ranges = np.array([np.zeros(poi_base.shape), np.ones(poi_base.shape)])
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

    #Convert to meshes for Xi and Eta
    
    #Formulate discretization into dictionary for use in executing function
    discretization = {"Xi":  Xi, 
                      "Eta": Eta, 
                      "Zeta": Zeta, 
                      "cell_centroid": cell_centroid,
                      "num_cell": num_cell,
                      "weights_ND": weights_ND
                  }
    gc.collect()
    #========================Setup Executing function==========================
    eval_fcn = lambda poi_normalized : lid_driven_pod_rom(\
                    poi_normalized, poi_names, qoi_names, poi_ranges, n_modes,\
                    discretization, velocity_unreduced_1D_compact, integration_times,
                    center_mat = center_locations, local_radius = .2)
    
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
    uqOptions.lsa.run=True
    uqOptions.lsa.run_lsa = False
    uqOptions.lsa.run_pss = run_pss
    uqOptions.lsa.pss_decomp_method= decomp_method
    uqOptions.lsa.pss_rel_tol = tolerance
    uqOptions.lsa.pss_algorithm = algorithm
    uqOptions.lsa.x_delta = x_delta
    uqOptions.lsa.deriv_method = deriv_method
    
    uqOptions.gsa.run_sobol = False
    uqOptions.gsa.run=True
    uqOptions.gsa.run_morris = run_morris
    uqOptions.gsa.n_samp_morris = n_samp_morris
    uqOptions.gsa.l_morris = l_morris
    
    uqOptions.save = True
    uqOptions.display = True
    uqOptions.plot = True
    if run_morris and run_pss:
        uqOptions.path = data_folder + "sensitivity/morris_ident_s" + str(n_snapshot) + "m" + \
            str(n_modes) + "_l" + str(int(1/l_morris)) + "_tForward" + str(t_forward) +\
            "_nSamp" + str(n_samp_morris) + "_tol" + str(int(np.log10(tolerance)*1000)/1000) + "_"
    elif run_morris:
        uqOptions.path = data_folder + "sensitivity/morris_s" + str(n_snapshot) + "m" + \
            str(n_modes) + "_l" + str(int(1/l_morris)) + "_tForward" + str(t_forward) +\
            "_nSamp" + str(n_samp_morris) + "_"
    elif run_pss:
        uqOptions.path = data_folder + "sensitivity/ident_s" + str(n_snapshot) + "m" + \
            str(n_modes) + "_tForward" + str(t_forward) + "_tol" + str(int(np.log10(tolerance)*1000)/1000) + "_"
    elif test_reduction:
        uqOptions.path = data_folder + "sensitivity/"+str(algorithm).lower()+\
            "_plots/reduction_s" + str(n_snapshot) + "m" + str(n_modes) + "_tForward"\
            + str(t_forward) + "_tol" +  str(int(np.log10(tolerance)*1000)/1000)\
            + "_nsamp" + str(reduction_nsamp) +  "_"
        
    # if logging>1:
    #     print("Base QOI Values: " + str(np.array([model.name_qoi, model.base_qoi]).transpose()))
    #Run SA
    if run_morris or run_pss:
        print("Running Sensitivity Analysis")
        results=uq.run_uq(model, uqOptions, logging= logging)
    if test_reduction:
        if run_pss:
            uq.lsa.test_model_reduction(model, results.lsa.inactive_set,\
                                        uqOptions.path, tolerance, \
                                        logging = logging)
        elif np.all(inactive_pois == "null"):
            raise Exception("PSS reduction test selected but not inactives set provided.")
        else:
            uq.lsa.test_model_reduction(model, inactive_pois, reduction_nsamp, \
                                        uqOptions.path, tolerance, \
                                        logging = logging)
    
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

