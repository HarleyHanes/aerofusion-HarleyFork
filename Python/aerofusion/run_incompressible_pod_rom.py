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
  grid_data_filename            = options.io["grid_data_filename"]
  snapshot_data_filename        = options.io["snapshot_data_filename"]
  pod_data_filename             = options.io["pod_data_filename"]
  rom_matrices_filename         = options.io["rom_matrices_filename"]
  rom_integration_data_filename = options.io["rom_integration_data_filename"]
  modal_coefficients_filename    = options.io["modal_coefficients_filename"]


  # ---------------------------------------------------------------------------
  # grid data generation
  # ---------------------------------------------------------------------------
#  if "grid_data" in options:
#    print('in grid data option')
#    grid_data_options = options["grid_data"]
#    padding_zeros = grid_data_options["padding_zeros"]
#    time_step_str = str(grid_data_options["time_step_min"])
#    pod_solution_filename = \
#    grid_data_options["les_solution_prefix"] + \
#    time_step_str.zfill(padding_zeros) + ".hdf5"
##    # Get data fields as 1D arrays and 3D arrays as dictionaries of
##    # - 1D arrays: use the unstructured-grid cell-based data
##    # - 3D arrays: use the structured-grid underlying topology via the
##    #   mesh_index_xi,eta,zeta
#
#    simulation_time_step, simulation_time, \
#      data_field_1D_arrays, data_field_3D_arrays  = \
#        hdf5_cell_data_to_numpy_array.read(
#          pod_solution_filename,
#          fields_to_read = "velocity",
#          xi_min = grid_data_options["mesh_xi_index_range"][0],
#          xi_max = grid_data_options["mesh_xi_index_range"][1],
#          eta_min = grid_data_options["mesh_eta_index_range"][0],
#          eta_max = grid_data_options["mesh_eta_index_range"][1],
#          zeta_min = grid_data_options["mesh_zeta_index_range"][0],
#          zeta_max = grid_data_options["mesh_zeta_index_range"][1])
#
#  # Write data to output file
#  os.makedirs(Path(snapshot_data_filename).parent, exist_ok=True)
#  print("\nWriting grid data to file", grid_data_filename )
#  np.savez(grid_data_filename,
#           cell_volume   = data_field_1D_arrays['cell_volume'],
#           cell_centroid = data_field_3D_arrays['cell_centroid'],
#           xi_index      = data_field_1D_arrays['mesh_index_xi'],
#           eta_index     = data_field_1D_arrays['mesh_index_eta'],
#           zeta_index    = data_field_1D_arrays['mesh_index_zeta']) 
#
##  import ipdb
##  ipdb.set_trace()
  # ---------------------------------------------------------------------------
  # Snapshot data generation
  # ---------------------------------------------------------------------------
#  if "snapshot_data" in options:
#
#    snapshot_data_options = options["snapshot_data"]
#    padding_zeros = snapshot_data_options["padding_zeros"]
#    list_of_time_steps = \
#      [tdx for tdx in range(snapshot_data_options["time_step_min"],
#                          snapshot_data_options["time_step_max"],
#                          snapshot_data_options["time_step_delta"])]
#    simulation_time_step_array = np.zeros(len(list_of_time_steps))
#    simulation_time_array = np.zeros(len(list_of_time_steps))
#
#  # Loop in timesteps to obtain the snapshot matrix U
#  for tdx, time_step in enumerate(tqdm(list_of_time_steps)):
#
#
#    time_step_str = str(time_step)
#    pod_solution_filename = \
#      snapshot_data_options["les_solution_prefix"] + \
#      time_step_str.zfill(padding_zeros) + ".hdf5"
#
#    print("\nProcessing time step", time_step_str, pod_solution_filename)
#    # Get data fields as 1D arrays and 3D arrays as dictionaries of
#    # - 1D arrays: use the unstructured-grid cell-based data
#    # - 3D arrays: use the structured-grid underlying topology via the
#    #   mesh_index_xi,eta,zeta
#    simulation_time_step, simulation_time, \
#      data_field_1D_arrays, data_field_3D_arrays  = \
#        hdf5_cell_data_to_numpy_array.read(
#          pod_solution_filename,
#          fields_to_read = "velocity",
#          xi_min = snapshot_data_options["mesh_xi_index_range"][0],
#          xi_max = snapshot_data_options["mesh_xi_index_range"][1],
#          eta_min = snapshot_data_options["mesh_eta_index_range"][0],
#          eta_max = snapshot_data_options["mesh_eta_index_range"][1],
#          zeta_min = snapshot_data_options["mesh_zeta_index_range"][0],
#          zeta_max = snapshot_data_options["mesh_zeta_index_range"][1])
#
#    if tdx == 0:
#      velocityData_1D = data_field_1D_arrays['velocity']
#      velocity_3D = data_field_3D_arrays['velocity']
#      dim_1D = velocityData_1D.shape
#      dim_3D = velocity_3D.shape
#      velocityData_3D = \
#        np.zeros([dim_3D[0], dim_3D[1],dim_3D[2],dim_3D[3], 1])      
#      velocityData_3D[:,:,:,:,0] = velocity_3D[:,:,:,:]     
#      velocityData_1D = \
#        np.reshape(velocityData_1D.transpose(),(dim_1D[0]*dim_1D[1],1)) 
#      simulation_time_step_array[0] = simulation_time_step
#      simulation_time_array[0] = simulation_time
#    else:
#      velocity_tdx_3D = \
#        np.zeros([dim_3D[0], dim_3D[1],dim_3D[2],dim_3D[3], 1])      
#      velocity_tdx_1D = data_field_1D_arrays['velocity']
#      velocity_tdx_3D[:,:,:,:,0] = data_field_3D_arrays['velocity']
#      velocity_tdx_1D = \
#        np.reshape(velocity_tdx_1D.transpose(),(dim_1D[0]*dim_1D[1],1)) 
#      velocityData_1D = \
#        np.concatenate((velocityData_1D, velocity_tdx_1D),axis = 1)
#      velocityData_3D = \
#        np.concatenate((velocityData_3D,velocity_tdx_3D),axis =4)
#      simulation_time_step_array[tdx] = simulation_time_step
#      simulation_time_array[tdx] = simulation_time
#      
#      print('time step',
#            simulation_time_step_array[tdx],
#            simulation_time_array[tdx])
#       #if tdx == len(list_of_time_steps)-1:
#    if "plot" in snapshot_data_options:
#      # Plot 3D array using structured-grid underlying topology
#      from aerofusion.plot.plot_2D import plot_pcolormesh
#      plot_pcolormesh(
#        data_field_3D_arrays["cell_centroid"][:,:,0,0],
#        data_field_3D_arrays["cell_centroid"][:,:,0,1],
#        data_field_3D_arrays["velocity"][:,:,0,0],
#        options["output_filename_prefix"] + \
#          snapshot_data_options["plot"]["output_filename_midfix"] + \
#          time_step_str.zfill(padding_zeros)+ ".png",
#        font_size = snapshot_data_options["plot"]["font_size"],
#        fig_size = snapshot_data_options["plot"]["fig_size"],
#        title = snapshot_data_options["plot"]["vertical_slice"]["title"],
#        vmin = snapshot_data_options["plot"]["vertical_slice"]["vmin"],
#        vmax = snapshot_data_options["plot"]["vertical_slice"]["vmax"],
#        cmap = snapshot_data_options["plot"]["vertical_slice"]["cmap"],
#        colorbar_label = \
#          snapshot_data_options["plot"]["vertical_slice"]["colorbar_label"],
#        xlabel = snapshot_data_options["plot"]["vertical_slice"]["xlabel"],
#        ylabel = snapshot_data_options["plot"]["vertical_slice"]["ylabel"])
#
#  # Write data to output file
#  os.makedirs(Path(snapshot_data_filename).parent, exist_ok=True)
#  print("\nWriting snapshot data to file", snapshot_data_filename )
#  np.savez(snapshot_data_filename,
#           simulation_time_step_array = simulation_time_step_array,
#           simulation_time_array      = simulation_time_array,
#           velocity_1D   = velocityData_1D,
#           velocity_3D   = velocityData_3D,
#           cell_volume   = data_field_1D_arrays['cell_volume'],
#           cell_centroid = data_field_3D_arrays['cell_centroid'],
#           xi_index      = data_field_1D_arrays['mesh_index_xi'],
#           eta_index     = data_field_1D_arrays['mesh_index_eta'],
#           zeta_index    = data_field_1D_arrays['mesh_index_zeta'])
#
#  import ipdb
#  ipdb.set_trace() 
###########################################
  print("Loading snapshot data from file", snapshot_data_filename)
  directory_snapshots = options.snapshot_data.output_directory
  snapshot_data = np.load(directory_snapshots + snapshot_data_filename)
 # Assign data to convenience variables
  #velocity_1D     = snapshot_data['velocity_1D'] ### only for the purpose of comparison
  velocity_3D     = snapshot_data['velocity_3D']
  cell_volume     = snapshot_data['cell_volume']
  cell_centroid   = snapshot_data['cell_centroid']
  xi              = snapshot_data['xi_index']
  eta             = snapshot_data['eta_index']
  zeta            = snapshot_data['zeta_index']
  simulation_time = snapshot_data['simulation_time_array']
 
  num_dim  = options.num_dimensions
  num_xi   = (cell_centroid.shape)[0]
  num_eta  = (cell_centroid.shape)[1]
  num_zeta = (cell_centroid.shape)[2]
  num_cell = (cell_volume.shape)[0]
  num_snapshots = (velocity_3D.shape)[4]

  ####-----checking 1D velocity structure---------
#  print('checking 1D velocity structure')
#  print('shape of velocity_1D, velocity_3D', velocity_1D.shape, \
#  velocity_3D.shape)
#  print( 'velocity_3D', velocity_3D[0,0,0, 0, 10], velocity_3D[0,0,0,1,10])
#  print(' velocity_1D', velocity_1D[0,10], velocity_1D[1,10])
#  import ipdb
#  ipdb.set_trace()
#  ######-------------------------------------------
#  velocity_1D = np.zeros([num_cell, num_dim, num_snapshots])
  
#  # ---------------------------------------------------------------------------
#  # POD
#  # ---------------------------------------------------------------------------
#  if "pod" in options:
#
#    pod_options = options["pod"]
#
#    print('Shape of velocity_3D', velocity_3D.shape)
#    print('Shape of cell_centroid', cell_centroid.shape)
#    print('Shape of cell_volume', cell_volume.shape)
#    # Convert velocity from 3D to 1D
#    print('converting velocity 3d to 1d')
#    for i_snap in range(num_snapshots):
#      for i_dim in range(num_dim):
#        print('i_snap', 'i_dim', i_snap, i_dim)
#        velocity_1D[:, i_dim, i_snap] = \
#          arr_conv.array_3D_to_1D(\
#            xi, eta, zeta, num_cell, velocity_3D[:,:,:, i_dim, i_snap])
#    velocity_1D_compact = np.zeros([num_cell*num_dim, num_snapshots])
#    print('restructuring velocity_1d')
#    for i_snap in range(num_snapshots):
#      velocity_1D_compact[:,i_snap] = \
#        np.reshape((velocity_1D[:, :, i_snap]).transpose(), (num_dim*num_cell))
#   
#    ### only for the purpose of debugging------------------
#    np.savez('velocity_1D_compact.npz', \
#     velocity_1D_compact = velocity_1D_compact) 
#    #####--------------------------------
#    num_dof = int(num_cell * num_dim)
#    print('mean velocity calculation')
#    # mean calculation of velocity
#    velocity_mean = np.zeros([num_dof])
#    velocity_mean = pod_modes.Find_Mean(velocity_1D_compact)
#    ### using first snapshot for time consistant parameter
#    #velocity_mean = velocity_1D_compact[:,0] 
#
#    mean_reduced_velocity = np.zeros(velocity_1D_compact.shape)
#    for i_mode in range(num_snapshots):
#      mean_reduced_velocity[:,i_mode] = \
#        velocity_1D_compact[:,i_mode] - velocity_mean[:]
#    
#    weights_ND = np.zeros([num_cell*num_dim])
#    for i_dim in range(num_dim):
#      weights_ND[i_dim*num_cell : (i_dim+1)*num_cell] = cell_volume
#  
#    print('caclculating number of modes needed for energy retention of:',\
#       pod_options.energy_retention )
#    num_retained_modes = \
#      pod_modes.find_number_of_modes( mean_reduced_velocity, \
#        weights_ND, pod_options.energy_retention)
#     
#    print('number of retained modes', num_retained_modes)
#    (phi, modal_coeff, pod_lambda) = pod_modes.Find_Modes(\
#       mean_reduced_velocity,
#       weights_ND,
#       num_retained_modes)
#       #options.pod.num_modes)
#   
#    print('phi, lambda', phi.shape, pod_lambda.shape)
#    num_modes = num_retained_modes # options.pod.num_modes 
#    print('Saving POD data to file', pod_data_filename)
#    np.savez(pod_data_filename,
#            simulation_time = simulation_time,
#            phi = phi,
#            pod_lambda = pod_lambda,
#            modal_coeff = modal_coeff,
#            velocity_mean = velocity_mean)
#    import ipdb
#    ipdb.set_trace()
## # # ---------------------------------------------------------------------------
 # ROM
  # ---------------------------------------------------------------------------
  if "rom" in options:

    print("Loading POD data from file", pod_data_filename)
    pod_data = np.load(pod_data_filename)
    # Assign data to convenience variables
    velocity_mean   = pod_data['velocity_mean']
    simulation_time = pod_data['simulation_time']
    pod_lambda      = pod_data['pod_lambda']
    phi             = pod_data['phi']
    modal_coeff     = pod_data['modal_coeff']
    
    #### only for now:
    weights_ND = np.zeros([num_cell*num_dim])
    for i_dim in range(num_dim):
        weights_ND[i_dim*num_cell : (i_dim+1)*num_cell] = cell_volume
    num_xi = num_xi 
    num_eta = num_eta -1
    num_zeta = num_zeta - 1

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
          xi, eta, zeta, num_cell, velocity_mean[:,i_dim])
      
    print(' - Calculation of Jacobian')
    jacobian = curvder.jacobian_of_grid_3d(\
       xi,
       eta,
       zeta,
       num_cell,
       cell_centroid,
       options.rom.jacobian.order_derivatives_x,
       options.rom.jacobian.order_derivatives_y,
       options.rom.jacobian.order_derivatives_z)
    print(' - Calculation of ROM matrices')
    (L0_calc, LRe_calc, C0_calc, CRe_calc, Q_calc) = \
       incrom.pod_rom_matrices_3d(\
         xi,
         eta,
         zeta,
         cell_centroid,
         num_cell,
         phi,
         weights_ND,
         velocity_0,
         jacobian,
         options.rom.jacobian.order_derivatives_x,
         options.rom.jacobian.order_derivatives_y, 
         options.rom.jacobian.order_derivatives_z)
    print(' - Saving matrices to file', rom_matrices_filename)
    np.savez(rom_matrices_filename,
              L0_calc  = L0_calc,
              LRe_calc = LRe_calc,
              C0_calc  = C0_calc,
              CRe_calc = CRe_calc,
              Q_calc   = Q_calc)

    print('Reading matrices from file', rom_matrices_filename)
    matrices = np.load(rom_matrices_filename)
    L0_calc  = matrices['L0_calc']
    LRe_calc = matrices['LRe_calc']
    C0_calc  = matrices['C0_calc']
    CRe_calc = matrices['CRe_calc']
    Q_calc   = matrices['Q_calc']

    integration_times = simulation_time[1:] 
   # print('ROM RK45 integration over times', integration_times)
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
   
#    aT = incrom.rom_calc_odeint(\
#           options.rom.reynolds_number,
#           char_L,
#           L0_calc,
#           LRe_calc,
#           C0_calc,
#           CRe_calc,
#           Q_calc,
#           modal_coeff,
#           integration_times)
#    mean_reduced_velocity_rom = np.matmul(phi, aT)
    # Save the output
    print('- Saving modal coefficienrs to file', modal_coefficients_filename) 
    np.savez(modal_coefficients_filename,
              aT_pod = modal_coeff,
              aT_rom = aT)
#------------------------------------------------------------------------------
if __name__ == "__main__":
  sys.exit(main())

