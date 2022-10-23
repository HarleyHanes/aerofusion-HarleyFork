
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
    
    logging = 1
    
    decomp_method = 'svd'
    #Parse Arguments 
    if "x_delta" in sys.argv:
        index = sys.argv.index("x_delta")
        x_delta = float(sys.argv[index+1])
        print("x_delta: " + str(x_delta))
    else:
        x_delta = 1e-6
        
    if "mean_perturbed" in sys.argv:
        mean_type = "mean_perturbed"
    else : 
        #mean_type = "mean_perturbed"
        mean_type = "artificial"
        
    if "n_samp_morris" in sys.argv:
        index = sys.argv.index("n_samp_morris")
        n_samp_morris = int(sys.argv[index+1])
        print("n_samp_morris: " + str(n_samp_morris))
    else : 
        n_samp_morris = 40
            
        
    if "l_morris" in sys.argv:
        index  = sys.argv.index("l_morris")
        l_morris = float(sys.argv[index+1])
        print("l_morris: "+str(x_delta))
    else:
        l_morris = 1/40
    
    if "Re" in sys.argv:
        index = sys.argv.index("Re")
        Re = int(sys.argv[index+1])
        print("x_delta: " + str(x_delta))
    else :
        Re = 17000
        
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
        use_energy = False
        print("n_modes: " + str(n_modes))
    elif "energy" in sys.argv:
        index = sys.argv.index("energy")
        n_modes = float(sys.argv[index+1])
        use_energy = True
        print("energy: " + str(n_modes))
    else:
        use_energy = False
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
        qoi_set = "fullQOI"  #integrated measures
        
    
    
    
    #rom_matrices_filename="../../lid_driven_penalty/rom_matrices_s500_m" + str(modes) + ".npz"
    save_path = "../../lid_driven_data/morris_screening_s"+str(n_snapshot) + \
        "m" + str(n_modes) + "_" 
    data_folder = "../../lid_driven_data/"
    weights_file = "weights_hr.mat"
    velocity_file = "re" + str(Re) + "_hr.mat"
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
    if qoi_set.lower() == "intqoi":
        qoi_names = np.array(["energy", "vorticity", \
                              "local vorticity 0", "local vorticity 1", "local vorticity 2",\
                              "local vorticity 3", "local vorticity 4", "local vorticity 5", \
                              "local vorticity 6"])
    elif qoi_set.lower() == "fullqoi":
        qoi_names = np.array(["full velocity"])
    else: 
        raise Exception("Unrecognized QOI Names: " + str(qoi_names))
           
    #poi_base = np.array([16000, 0, 1, 1, 0, -.75, .75, 1])
    # poi_base = np.array([20000, -1.5, 0,\
    #                      1, .5, .5, .1, .1, .1, .1, \
    #                      0, 3*np.pi/4, np.pi/4, 0, 0, 0, 0,\
    #                      -.75, -.75, .75, -.2, .2, .5, .75, \
    #                      .75, -.75, -.75, -.2, .2, .5, .75, 
    #                      1, 1.5, 1.5, 1, 1, 1, 1])

    if mean_type.lower() == "artificial":
        poi_base = np.array([17000, -1.5, -2,\
                              .95, .5, .5, .1, .1, .1, .1, \
                              0, 3*np.pi/4, np.pi/4, 0, 0, 0, 0,\
                              -.75, -.75, .75, -.2, .2, .5, .75, \
                              .75, -.75, -.75, -.2, .2, .5, .75, 
                              1, 1.5, 1.5, 1, 1, 1, 1])
    elif mean_type.lower() == "mean_perturbed":
        poi_base = np.array([17000, -1.5, -2,\
                              0, 0, 0, 0, 0, 0, 0, \
                              0, 3*np.pi/4, np.pi/4, 0, 0, 0, 0,\
                              -.75, -.75, .75, -.2, .2, .5, .75, \
                              .75, -.75, -.75, -.2, .2, .5, .75, 
                              1, 1.5, 1.5, 1, 1, 1, 1])
            
    if Re == 25000:
        poi_base[0] = 25000
        
        #initialize ranges with pm .25 of base values
    poi_ranges = np.array([poi_base*.75, poi_base*1.25]).transpose()
        #Set alternate Reynolds range
    if Re == 17000:
        poi_ranges[0] = np.array([11000, 20000])
    elif Re == 25000:
        poi_ranges[0] = np.array([19000, 28000])
    poi_ranges[1] = np.array([-2, 0])
    poi_ranges[2] = np.array([-12, 0])
    if mean_type.lower() == "mean_perturbed":
        poi_ranges[3:10,:] =np.array([-.1,.1])
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
    print("Ranges: " + str(poi_ranges))
    poi_normalized = normalize_pois(poi_base, poi_ranges)
    print("Normalized: " + str(poi_normalized))
    poi_normalized_ranges = np.array([np.zeros(poi_base.shape), np.ones(poi_base.shape)])
    print("Normalized Ranges: " + str(poi_normalized_ranges))
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
                    center_mat = center_locations, local_radius = .2, mean_type = mean_type,
                    use_energy = use_energy)
    
    if logging:
        print("Setting up model")
    if qoi_set.lower() == "intqoi":
        model = uq.Model(eval_fcn = eval_fcn,
                          base_poi = poi_normalized,
                          dist_type = "uniform",
                          dist_param = poi_normalized_ranges,
                          name_poi = poi_names,
                          name_qoi = qoi_names
                          )
    else : 
        model = uq.Model(eval_fcn = eval_fcn,
                          base_poi = poi_normalized,
                          dist_type = "uniform",
                          dist_param = poi_normalized_ranges,
                          name_poi = poi_names
                          )
    
    #Determine which aspects of UQLibrary are run
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
        uqOptions.path = data_folder + "sensitivity/morris_ident_Re" + str(Re) +"_s" + str(n_snapshot) + "m" + \
            str(n_modes) + "_l" + str(int(1/l_morris)) + "_tForward" + str(t_forward) +\
            "_nSamp" + str(n_samp_morris) + "_tol" + str(int(np.log10(tolerance)*1000)/1000) + "_" + str(qoi_set) + "_"
    elif run_morris:
        if mean_type.lower() == "artificial":
            uqOptions.path = data_folder + "sensitivity/Re" + str(Re) +"_art_s" + str(n_snapshot) 
        elif mean_type.lower()== "mean_perturbed":
            uqOptions.path = data_folder + "sensitivity/Re" + str(Re) +"_pert_s" + str(n_snapshot)
        else : 
            raise Exception("Unrecgonzied mean type: " + mean_type)
        if use_energy:
            uqOptions.path += "e" + \
                str(n_modes) + "_l" + str(int(1/l_morris)) + "_tForward" + str(t_forward) +\
                "_nSamp" + str(n_samp_morris) + "_" + str(qoi_set) + "_"
        else: 
            uqOptions.path += "m" + \
                str(n_modes) + "_l" + str(int(1/l_morris)) + "_tForward" + str(t_forward) +\
                "_nSamp" + str(n_samp_morris) + "_" + str(qoi_set) + "_"
            
    elif run_pss:
        uqOptions.path = data_folder + "sensitivity/ident_Re" + str(Re) +"_s" + str(n_snapshot) + "m" + \
            str(n_modes) + "_tForward" + str(t_forward) + "_tol" + str(int(np.log10(tolerance)*1000)/1000) + "_"+ str(qoi_set) + "_"
    elif test_reduction:
        uqOptions.path = data_folder + "sensitivity/"+str(algorithm).lower()+\
            "_plots/reduction_Re" + str(Re) +"_s" + str(n_snapshot) + "m" + str(n_modes) + "_tForward"\
            + str(t_forward) + "_tol" +  str(int(np.log10(tolerance)*1000)/1000)\
            + "_nsamp" + str(reduction_nsamp) +  "_" + str(qoi_set) + "_"
        
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
    


if __name__ == "__main__":
    sys.exit(main())

