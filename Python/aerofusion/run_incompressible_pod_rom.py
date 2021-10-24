# -----------------------------------------------------------------------------
# \file process_incompressible_pod_rom.py
# \brief Process incompressible POD-ROM
#
# To run, issue the command:
#   python3 process_incompressible_pod_rom.py \
#           process_incompressible_pod_rom.cfg
# where the first argument is a libconf file with the options to run this
# script, regarding input/output files, POD and ROM settings, etc.
# -----------------------------------------------------------------------------

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
import time
import numba

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
from aerofusion.plot.plot_2D import plot_contour
from aerofusion.numerics import derivatives_curvilinear_grid as curvder
#from aerofusion.numerics.interpolation import interpolate_vectorial_field_in_2D

# -----------------------------------------------------------------------------
def main(argv=None):

  if argv is None:
    argv = sys.argv[1:]

  # Parse arguments and options -----------------------------------------------
  usage_text = (
    "Usage: \n"
    "  python3 " + __file__ + "\n" +
    "Arguments:" + \
    "\n  input_filename[1]:")
  parser = argparse.ArgumentParser(description = usage_text)
  parser.add_argument("input_filename",
    help="Name of input (libconfig) file with postprocessing options")
  parser.add_argument("-d", "--debug",
    action="store_true", dest="debug")
  parser.add_argument("-v", "--verbose",
    action="store_true", dest="verbose")
  parser.add_argument("-q", "--quiet",
    action="store_false", dest="verbose")

  args = parser.parse_args(argv)

  # Execute code --------------------------------------------------------------

 # Read options from input (libconfig) file
  with io.open(args.input_filename) as libconfig_file:
    raw_options = libconf.load(libconfig_file)

  logging_format = '%(asctime)s,%(msecs)d %(levelname)-8s ' + \
                   '[%(pathname)s:%(filename)s:%(lineno)d] %(message)s'
  logging_date_format = '%Y-%m-%d:%H:%M:%S'
  if "logging_level" in raw_options.keys():
    if raw_options["logging_level"] == "DEBUG":
      logging.basicConfig(\
          format  = logging_format,
          datefmt = logging_date_format,
          level   = logging.DEBUG)
  else:
    logging.basicConfig(\
        format  = logging_format,
        datefmt = logging_date_format)

  # ---------------------------------------------------------------------------
  # Initialize environment variables if specified values
  if "environment_variables" in raw_options:
    env_variable_keys = raw_options["environment_variables"].keys()
    for env_variable_key in env_variable_keys:
      os.environ[env_variable_key] = \
        raw_options["environment_variables"][env_variable_key]

  # parse libconfig file
  with io.open(args.input_filename, "r") as fh:
    raw_options_str = fh.read()
  config_str_eval = os.path.expandvars(raw_options_str)
  with io.StringIO(config_str_eval) as fh:
    options = libconf.load(fh)

  #---------------------------------------------------------------------------
  # Create parent directory of output filename prefix, if it does not exist
  os.makedirs(Path(options.output_filename_prefix).parent, exist_ok = True)

  # Set variable on whether to reuse any previously generated postprocess files
  reuse_existing_postprocessing_files = False
  if "reuse_existing_postprocessing_files" in options:
    reuse_existing_postprocessing_files = \
      options["reuse_existing_postprocessing_files"]

  #---------------------------------------------------------------------------
  print("Read options -------------------------------------------------------")
  print(libconf.dumps(options))
  print("--------------------------------------------------------------------")

  # Initialize filenames
  pod_data_filename             = options.io["pod_data_filename"]
  rom_matrices_filename         = options.io["rom_matrices_filename"]
  rom_integration_data_filename = options.io["rom_integration_data_filename"]


  num_dim  = options.num_dimensions

  # import ipdb
  # ipdb.set_trace()
  # Load mesh file first to allocate memory
  print("Reading mesh file")
  input_mesh_filename = \
    options["io"]["snapshot_data_filename_prefix"] + "mesh.npz"
  mesh_data = np.load(input_mesh_filename)
  cell_volume           = mesh_data['cell_volume']
  cell_centroid         = mesh_data['cell_centroid']
  xi                    = mesh_data['xi_index']
  eta                   = mesh_data['eta_index']
  zeta                  = mesh_data['zeta_index']
  mesh_xi_index_range   = mesh_data["mesh_xi_index_range"]
  mesh_eta_index_range  = mesh_data["mesh_eta_index_range"] 
  mesh_zeta_index_range = mesh_data["mesh_zeta_index_range"] 
  mesh_xi_index_min = mesh_xi_index_range[0]
  if mesh_xi_index_min  == "auto":
    mesh_xi_index_min = xi.min()
  mesh_xi_index_max = mesh_xi_index_range[1]
  if mesh_xi_index_max  == "auto":
    mesh_xi_index_max = xi.max()
  mesh_eta_index_min = mesh_eta_index_range[0]
  if mesh_eta_index_min  == "auto":
    mesh_eta_index_min = eta.min()
  mesh_eta_index_max = mesh_eta_index_range[1]
  if mesh_eta_index_max  == "auto":
    mesh_eta_index_max = eta.max()
  mesh_zeta_index_min = mesh_zeta_index_range[0]
  if mesh_zeta_index_min  == "auto":
    mesh_zeta_index_min = zeta.min()
  mesh_zeta_index_max = mesh_zeta_index_range[1]
  if mesh_zeta_index_max  == "auto":
    mesh_zeta_index_max = zeta.max()
  num_xi   = mesh_xi_index_max   - mesh_xi_index_min   + 1
  num_eta  = mesh_eta_index_max  - mesh_eta_index_min  + 1
  num_zeta = mesh_zeta_index_max - mesh_zeta_index_min + 1
  num_cell = cell_volume.shape[0]
  cell_centroid_3D = cell_centroid.reshape((num_xi, num_eta, num_zeta, 3))
  weights_ND = np.zeros([num_cell*num_dim])
  for i_dim in range(num_dim):
    weights_ND[i_dim*num_cell : (i_dim+1)*num_cell] = cell_volume
    
  #ivanDebug
  #mesh_xi_eta_zeta_ranges = \
  #  [mesh_xi_index_min, mesh_xi_index_max,
  #   mesh_eta_index_min, mesh_eta_index_max,
  #   mesh_zeta_index_min, mesh_zeta_index_max]
  xi_relative_array   = xi   - mesh_xi_index_min
  eta_relative_array  = eta  - mesh_eta_index_min
  zeta_relative_array = zeta - mesh_zeta_index_min
  Nxi   = max(xi_relative_array)   - min(xi_relative_array)   + 1
  Neta  = max(eta_relative_array)  - min(eta_relative_array)  + 1
  Nzeta = max(zeta_relative_array) - min(zeta_relative_array) + 1

  #print("DEBUG numba", numba.config.NUMBA_DEFAULT_NUM_THREADS)
  #print("DEBUG", cell_centroid_3D.shape)
  ## Get range of mesh_index_xi_array,eta,zeta
  #cell_volume_3D = cell_volume.reshape((num_xi, num_eta, num_zeta))
  #t_begin = time.time()
  #num_iter = 2
  #for idx in range(num_iter):
  #  cell_volume_1D = arr_conv.array_3D_to_1D(\
  #          xi_relative_array,
  #          eta_relative_array,
  #          zeta_relative_array,
  #          num_cell,
  #          cell_volume_3D)
  #  tmp_cell_volume_3D = arr_conv.array_1D_to_3D(\
  #          xi_relative_array,]
  #          eta_relative_array,
  #          zeta_relative_array,
  #          num_cell,
  #          Nxi, Neta, Nzeta,
  #          cell_volume_1D)
  #t_end = time.time()
  #print("t_end", t_end-t_begin)
  #exit(1)
  ##ivanDebugEnd

  # ---------------------------------------------------------------------------
  # POD
  # ---------------------------------------------------------------------------
  if "pod" in options:

    pod_options = options["pod"]

    print("Loading snapshot data from files")

    #ivanMod
    snapshot_data_options = options["pod"]["input_snapshot_data"]
    padding_zeros = snapshot_data_options["padding_zeros"]
    list_of_time_steps = \
      [tdx for tdx in range(snapshot_data_options["time_step_min"],
                            snapshot_data_options["time_step_max"],
                            snapshot_data_options["time_step_delta"])]
    simulation_time_step_array = np.zeros(len(list_of_time_steps))
    simulation_time_array = np.zeros(len(list_of_time_steps))

    num_snapshots = len(list_of_time_steps)

    ## Assign data to convenience variables
    print("num_snapshots", num_snapshots)
    velocity_1D_compact = np.zeros([num_cell*num_dim, num_snapshots])
    # Loop in timesteps to obtain the snapshot matrix U
    for tdx, time_step in enumerate(tqdm(list_of_time_steps)):

      time_step_str = str(time_step)
      input_solution_fields_filename = \
        options["io"]["snapshot_data_filename_prefix"] + \
        time_step_str.zfill(padding_zeros) + ".npz"
      #print("\nReading solution files at time step",
      #  time_step_str, "from file", input_solution_fields_filename)
      solution_fields_data = np.load(input_solution_fields_filename)
      simulation_time_step = solution_fields_data["simulation_time_step"]
      # Sanity check that the time steps match what we expect
      if simulation_time_step != time_step:
        print("ERROR - Mismatch in read simulation time step",
              simulation_time_step, "vs", time_step) 
      simulation_time = solution_fields_data["simulation_time"]
      velocityData_1D_x = solution_fields_data["velocity_1D_x"]
      velocityData_1D_y = solution_fields_data["velocity_1D_y"]
      velocityData_1D_z = solution_fields_data["velocity_1D_z"]
      simulation_time_step_array[tdx] = simulation_time_step
      simulation_time_array[tdx] = simulation_time
      velocity_components_concatenated = np.concatenate(\
        (velocityData_1D_x,velocityData_1D_y,velocityData_1D_z))
      velocity_1D_compact[:,tdx] = velocity_components_concatenated
        
    num_dof = int(num_cell * num_dim)
    print('mean velocity calculation')
    # mean calculation of velocity
    velocity_mean = np.zeros([num_dof])
    velocity_mean = pod_modes.Find_Mean(velocity_1D_compact)
    ### using first snapshot for time consistant parameter
    #velocity_mean = velocity_1D_compact[:,0] 


    mean_reduced_velocity = np.zeros(velocity_1D_compact.shape)
    for i_mode in range(num_snapshots):
      mean_reduced_velocity[:,i_mode] = \
        velocity_1D_compact[:,i_mode] - velocity_mean[:]
    
    print('Calculating', options.pod.num_modes, 'POD modes')
    (phi, modal_coeff, pod_lambda) = pod_modes.Find_Modes(\
        mean_reduced_velocity,
        weights_ND,
        options.pod.num_modes)
    
    #--------------------------------------------------------------------------
    print('phi, lambda', phi.shape, pod_lambda.shape)
    num_modes = options.pod.num_modes 
   ###---visualizing modes
   # phi_1D = np.zeros([num_cell, num_dim, num_modes]) 
   # for i_mode in range(num_modes):
   #   print("DEBUG", i_mode, num_modes, num_dim, num_cell)
   #   phi_1D[:, :, i_mode] = \
   #     np.reshape(phi[:, i_mode], (num_dim, num_cell)).transpose()
   # 
   # phi_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim, num_modes])
   # for i_dim in range(num_dim):
   #   for i_mode in range(num_modes):
   #     print("DEBUG", i_dim, i_mode, num_dim, num_modes, num_cell)
   #     print("DEBUG", phi_1D[:, i_dim, i_mode].shape)
   #     phi_3D[:, :, :, i_dim, i_mode] = \
   #       arr_conv.array_1D_to_3D(\
   #         xi, eta, zeta, num_cell,\
   #         phi_1D[:, i_dim, i_mode])
   # print('plotting modes and sigma') 
   # plotting modes and sigma
   # num_modes = pod_lambda.shape
   # plt.plot(np.log(pod_lambda[0:num_modes[0]-1]/pod_lambda[0]), '*')
   # plt.xlabel('number of modes')
   # plt.ylabel('ln(lambda/lambda_0)')
   # plt.savefig(options.output_filename_prefix + 'lambda_lambda0.png')
   # plt.clf()
   # # Contour plots
   # plot_contour(\
   #   cell_centroid[:,:,0,0],
   #   cell_centroid[:,:,0,1],
   #   phi_3D[:,:,0,0,0],
   #   options.output_filename_prefix + 'phi_u_0.png',
   #   options.plot.contour.levels,
   #   options.plot.contour.vmin,
   #   options.plot.contour.vmax)
   # plot_contour(\
   #   cell_centroid[:,:,0,0],
   #   cell_centroid[:,:,0,1],
   #   phi_3D[:,:,0,1,0],
   #   options.output_filename_prefix + 'phi_v_0.png',
   #   options.plot.contour.levels,
   #   options.plot.contour.vmin,
   #   options.plot.contour.vmax)
   # plot_contour(\
   #   cell_centroid[:,:,0,0],
   #   cell_centroid[:,:,0,1],
   #   phi_3D[:,:,0,0,1],
   #   options.output_filename_prefix + 'phi_u_1.png',
   #   options.plot.contour.levels,
   #   options.plot.contour.vmin,
   #   options.plot.contour.vmax)
   # plot_contour(\
   #   cell_centroid[:,:,0,0],
   #   cell_centroid[:,:,0,1],
   #   phi_3D[:,:,0,1,1],
   #   options.output_filename_prefix + 'phi_v_1.png',
   #   options.plot.contour.levels,
   #   options.plot.contour.vmin,
   #   options.plot.contour.vmax)
   # 
    print('Saving POD data to file', pod_data_filename)
    np.savez(pod_data_filename,
             simulation_time_array = simulation_time_array,
             phi = phi,
             pod_lambda = pod_lambda,
             modal_coeff = modal_coeff,
             velocity_mean = velocity_mean,
             Nxi = Nxi, Neta = Neta, Nzeta = Nzeta,\
             xi_relative_array = xi_relative_array,\
             eta_relative_array = eta_relative_array, \
             zeta_relative_array = zeta_relative_array)
  #import ipdb
  #ipdb.set_trace()
  # ---------------------------------------------------------------------------
  # ROM
  # ---------------------------------------------------------------------------
  if "rom" in options:

    print("Loading POD data from file", pod_data_filename)
    print("In develop branch")
    pod_data = np.load(pod_data_filename)
    # Assign data to convenience variables
    velocity_mean         = pod_data['velocity_mean']
    simulation_time_array = pod_data['simulation_time_array']
    pod_lambda            = pod_data['pod_lambda']
    phi                   = pod_data['phi']
    modal_coeff           = pod_data['modal_coeff']
    Nxi                   = pod_data['Nxi']
    Neta                  = pod_data['Neta']
    Nzeta                 = pod_data['Nzeta']
    xi_relative_array     = pod_data['xi_relative_array']
    eta_relative_array    = pod_data['eta_relative_array']
    zeta_relative_array   = pod_data['zeta_relative_array']

    num_dim  = options.num_dimensions
    num_cell = len(velocity_mean)//num_dim
    num_dof = int(num_cell * num_dim)
    if options.rom.calculate_matrices:

      print(" - Calculating velocity_mean and velocity_0")
      velocity_mean = \
        (np.reshape(velocity_mean, (num_dim, num_cell))).transpose()
      velocity_0 = np.zeros([num_xi, num_eta, num_zeta, num_dim])
      for i_dim in range(num_dim):
        velocity_0[:,:,:,i_dim] = arr_conv.array_1D_to_3D(\
          xi_relative_array, eta_relative_array, zeta_relative_array, \
           Nxi, Neta, Nzeta, velocity_mean[:,i_dim]) 
      
    print(' - Calculation of Jacobian')
    jacobian = curvder.jacobian_of_grid_3d(\
       xi_relative_array,
       eta_relative_array,
       zeta_relative_array,
       num_cell,
       cell_centroid_3D,
       options.rom.derivatives.order_x,
       options.rom.derivatives.order_y,
       options.rom.derivatives.order_z)
    print(' - Calculation of ROM matrices')
    (L0_calc, LRe_calc, C0_calc, CRe_calc, Q_calc) = \
       incrom.pod_rom_matrices_3d(\
         xi_relative_array,
         eta_relative_array,
         zeta_relative_array,
         cell_centroid_3D,
         num_cell,
         phi,
         weights_ND,
         velocity_0,
         jacobian,
         options.rom.derivatives.order_x,
         options.rom.derivatives.order_y, 
         options.rom.derivatives.order_z)
    print(' - Saving matrices to file', rom_matrices_filename)
    np.savez(rom_matrices_filename,
              L0_calc  = L0_calc,
              LRe_calc = LRe_calc,
              C0_calc  = C0_calc,
              CRe_calc = CRe_calc,
              Q_calc   = Q_calc)

   # print('Reading matrices from file', rom_matrices_filename)
   # matrices = np.load(rom_matrices_filename)
   # L0_calc  = matrices['L0_calc']
   # LRe_calc = matrices['LRe_calc']
   # C0_calc  = matrices['C0_calc']
   # CRe_calc = matrices['CRe_calc']
   # Q_calc   = matrices['Q_calc'

    integration_times = simulation_time_array[1:]
    print('ROM RK45 integration over times', integration_times)
    char_L = 1
    aT = incrom.rom_calc_rk45(\
           options.rom.reynolds_number,
           char_L,
           L0_calc,
           LRe_calc,
           C0_calc,
           CRe_calc,
           Q_calc,
           modal_coeff,
           integration_times)
    mean_reduced_velocity_rom = np.matmul(phi, aT)
    # Save the output
  
#------------------------------------------------------------------------------
if __name__ == "__main__":
  sys.exit(main())

