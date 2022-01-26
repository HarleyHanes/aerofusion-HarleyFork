#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 07:00:07 2021

@author: cjedwar3
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: cjedwar3
"""
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
  argv=None
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
  snapshot_data_filename        = options.io["snapshot_data_filename"]
  #dmd_data_filename             = options.io["dmd_data_filename"]
  #pod_data_filename             = options.io["pod_data_filename"]
  #rom_matrices_filename         = options.io["rom_matrices_filename"]
  #rom_integration_data_filename = options.io["rom_integration_data_filename"]

  # ---------------------------------------------------------------------------
  # Snapshot data generation
  # ---------------------------------------------------------------------------

  # ---------------------------------------------------------------------------
  # ROM
  # ---------------------------------------------------------------------------
  if "rom" in options:
    '''
    snapshot_data = np.load(snapshot_data_filename)
    # Assign data to convenience variables
    velocity_3D     = snapshot_data['velocity_3D']
    cell_volume     = snapshot_data['cell_volume']
    #cell_centroid   = snapshot_data['cell_centroid']
    xi              = snapshot_data['xi_index']
    eta             = snapshot_data['eta_index']
    zeta            = snapshot_data['zeta_index']
    
    num_dim  = options.num_dimensions
    #num_xi   = (cell_centroid.shape)[0]
    #num_eta  = (cell_centroid.shape)[1]
    #num_zeta = (cell_centroid.shape)[2]
    num_cell = (cell_volume.shape)[0]
    num_snapshots = (velocity_3D.shape)[4]
    velocity_1D = np.zeros([num_cell, num_dim, num_snapshots])
    
    for i_snap in range(num_snapshots):
      for i_dim in range(num_dim):
        velocity_1D[:, i_dim, i_snap] = \
          arr_conv.array_3D_to_1D(\
            xi, eta, zeta, num_cell, velocity_3D[:,:,:, i_dim, i_snap])
    velocity_1D_compact = np.zeros([num_cell*num_dim, num_snapshots])
    for i_snap in range(num_snapshots):
      velocity_1D_compact[:,i_snap] = \
        np.reshape((velocity_1D[:, :, i_snap]).transpose(), (num_dim*num_cell))
    
    num_dof = int(num_cell * num_dim)
    # mean calculation of velocity
    velocity_mean = np.zeros([num_dof])
    velocity_mean = pod_modes.Find_Mean(velocity_1D_compact)
    
    mean_reduced_velocity = np.zeros(velocity_1D_compact.shape)
    for i_mode in range(num_snapshots):
      mean_reduced_velocity[:,i_mode] = \
        velocity_1D_compact[:,i_mode] - velocity_mean[:]
        
    weights_ND = np.zeros([num_cell*num_dim])
    for i_dim in range(num_dim):
      weights_ND[i_dim*num_cell : (i_dim+1)*num_cell] = cell_volume
        
    KE=np.zeros((num_snapshots,1))
    for i in range(0,num_snapshots):
        KE[i,0]=np.dot(np.multiply(mean_reduced_velocity[:,i],weights_ND),mean_reduced_velocity[:,i])
    '''
    rom = np.load("coeffs.npz")
    coeffs=rom['coeffs']
    
    num=coeffs.shape[1]
    KE_rom_r50=np.zeros((num,1))
    for i in range(0,num):
        KE_rom_r50[i,0]=np.dot(abs(coeffs[:,i]),abs(coeffs[:,i]))
        
    rom = np.load("coeffs_r20.npz")
    coeffs=rom['coeffs']
    
    num=coeffs.shape[1]
    KE_rom_r20=np.zeros((num,1))
    for i in range(0,num):
        KE_rom_r20[i,0]=np.dot(abs(coeffs[:,i]),abs(coeffs[:,i]))
        
    np.savez("Kinetic_energy_rom",
              KE_rom_r50 = KE_rom_r50,
              KE_rom_r20 = KE_rom_r20)
        
    
    
    #print("Loading POD data from file", pod_data_filename)
    '''
    pod_data = np.load(pod_data_filename)
    # Assign data to convenience variables
    velocity_mean   = pod_data['velocity_mean']
    simulation_time = pod_data['simulation_time']
    pod_lambda      = pod_data['pod_lambda']
    phi             = pod_data['phi']
    modal_coeff     = pod_data['modal_coeff'][:,0:10]

    num_dim  = options.num_dimensions
    num_cell = len(velocity_mean)//num_dim
    num_dof = int(num_cell * num_dim)
    
    A=np.dot(phi,modal_coeff)

    A_1D = np.zeros([num_cell, num_dim, 10])
    for i_mode in range(10):
      A_1D[:, :, i_mode] = \
        np.reshape(A[:, i_mode], (num_dim, num_cell)).transpose()
    
    A_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim, 10])
    for i_dim in range(num_dim):
      for i_mode in range(10):
        A_3D[:, :, :, i_dim, i_mode] = \
          arr_conv.array_1D_to_3D(\
            xi, eta, zeta, num_cell,\
            A_1D[:, i_dim, i_mode])
              
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,0,i],
          options.output_filename_prefix + 'reconstruction_podr50_x'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,1,i],
          options.output_filename_prefix + 'reconstruction_podr50_y'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
        
    phi             = pod_data['phi'][:,0:20]
    modal_coeff     = pod_data['modal_coeff'][0:20,0:10]

    num_dim  = options.num_dimensions
    num_cell = len(velocity_mean)//num_dim
    num_dof = int(num_cell * num_dim)
    
    A=np.dot(phi,modal_coeff)

    A_1D = np.zeros([num_cell, num_dim, 10])
    for i_mode in range(10):
      A_1D[:, :, i_mode] = \
        np.reshape(A[:, i_mode], (num_dim, num_cell)).transpose()
    
    A_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim, 10])
    for i_dim in range(num_dim):
      for i_mode in range(10):
        A_3D[:, :, :, i_dim, i_mode] = \
          arr_conv.array_1D_to_3D(\
            xi, eta, zeta, num_cell,\
            A_1D[:, i_dim, i_mode])
              
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,0,i],
          options.output_filename_prefix + 'reconstruction_podr20_x'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,1,i],
          options.output_filename_prefix + 'reconstruction_podr20_y'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)

    phi             = pod_data['phi'][:,0:10]
    modal_coeff     = pod_data['modal_coeff'][0:10,0:10]

    num_dim  = options.num_dimensions
    num_cell = len(velocity_mean)//num_dim
    num_dof = int(num_cell * num_dim)
    
    A=np.dot(phi,modal_coeff)

    A_1D = np.zeros([num_cell, num_dim, 10])
    for i_mode in range(10):
      A_1D[:, :, i_mode] = \
        np.reshape(A[:, i_mode], (num_dim, num_cell)).transpose()
    
    A_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim, 10])
    for i_dim in range(num_dim):
      for i_mode in range(10):
        A_3D[:, :, :, i_dim, i_mode] = \
          arr_conv.array_1D_to_3D(\
            xi, eta, zeta, num_cell,\
            A_1D[:, i_dim, i_mode])
              
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,0,i],
          options.output_filename_prefix + 'reconstruction_podr10_x'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,1,i],
          options.output_filename_prefix + 'reconstruction_podr10_y'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
        
    phi             = pod_data['phi'][:,0:5]
    modal_coeff     = pod_data['modal_coeff'][0:5,0:10]

    num_dim  = options.num_dimensions
    num_cell = len(velocity_mean)//num_dim
    num_dof = int(num_cell * num_dim)
    
    A=np.dot(phi,modal_coeff)

    A_1D = np.zeros([num_cell, num_dim, 10])
    for i_mode in range(10):
      A_1D[:, :, i_mode] = \
        np.reshape(A[:, i_mode], (num_dim, num_cell)).transpose()
    
    A_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim, 10])
    for i_dim in range(num_dim):
      for i_mode in range(10):
        A_3D[:, :, :, i_dim, i_mode] = \
          arr_conv.array_1D_to_3D(\
            xi, eta, zeta, num_cell,\
            A_1D[:, i_dim, i_mode])
              
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,0,i],
          options.output_filename_prefix + 'reconstruction_podr5_x'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,1,i],
          options.output_filename_prefix + 'reconstruction_podr5_y'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
        
    dmd_data = np.load("dmd_modes_r50.npz")
    # Assign data to convenience variables
    phi             = dmd_data['phi']
    modal_coeff     = dmd_data['modal_coeff'][:,0:10]
    
    A=np.dot(phi,modal_coeff)

    A_1D = np.zeros([num_cell, num_dim, 10],dtype='complex')
    for i_mode in range(10):
      A_1D[:, :, i_mode] = \
        np.reshape(A[:, i_mode], (num_dim, num_cell)).transpose()
    
    A_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim, 10],dtype='complex')
    for i_dim in range(num_dim):
      for i_mode in range(10):
        A_3D[:, :, :, i_dim, i_mode] = \
          arr_conv.array_1D_to_3D(\
            xi, eta, zeta, num_cell,\
            A_1D[:, i_dim, i_mode])
              
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,0,i],
          options.output_filename_prefix + 'reconstruction_dmdr50_x'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,1,i],
          options.output_filename_prefix + 'reconstruction_dmdr50_y'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
        
    dmd_data = np.load("dmd_modes_r20.npz")
    # Assign data to convenience variables
    phi             = dmd_data['phi']
    modal_coeff     = dmd_data['modal_coeff'][:,0:10]
    
    A=np.dot(phi,modal_coeff)

    A_1D = np.zeros([num_cell, num_dim, 10],dtype='complex')
    for i_mode in range(10):
      A_1D[:, :, i_mode] = \
        np.reshape(A[:, i_mode], (num_dim, num_cell)).transpose()
    
    A_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim, 10],dtype='complex')
    for i_dim in range(num_dim):
      for i_mode in range(10):
        A_3D[:, :, :, i_dim, i_mode] = \
          arr_conv.array_1D_to_3D(\
            xi, eta, zeta, num_cell,\
            A_1D[:, i_dim, i_mode])
              
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,0,i],
          options.output_filename_prefix + 'reconstruction_dmdr20_x'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,1,i],
          options.output_filename_prefix + 'reconstruction_dmdr20_y'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
        
    dmd_data = np.load("dmd_modes_r10.npz")
    # Assign data to convenience variables
    phi             = dmd_data['phi']
    modal_coeff     = dmd_data['modal_coeff'][:,0:10]
    
    A=np.dot(phi,modal_coeff)

    A_1D = np.zeros([num_cell, num_dim, 10],dtype='complex')
    for i_mode in range(10):
      A_1D[:, :, i_mode] = \
        np.reshape(A[:, i_mode], (num_dim, num_cell)).transpose()
    
    A_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim, 10],dtype='complex')
    for i_dim in range(num_dim):
      for i_mode in range(10):
        A_3D[:, :, :, i_dim, i_mode] = \
          arr_conv.array_1D_to_3D(\
            xi, eta, zeta, num_cell,\
            A_1D[:, i_dim, i_mode])
              
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,0,i],
          options.output_filename_prefix + 'reconstruction_dmdr10_x'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,1,i],
          options.output_filename_prefix + 'reconstruction_dmdr10_y'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
        
    dmd_data = np.load("dmd_modes_r5.npz")
    # Assign data to convenience variables
    phi             = dmd_data['phi']
    modal_coeff     = dmd_data['modal_coeff'][:,0:10]
    
    A=np.dot(phi,modal_coeff)

    A_1D = np.zeros([num_cell, num_dim, 10],dtype='complex')
    for i_mode in range(10):
      A_1D[:, :, i_mode] = \
        np.reshape(A[:, i_mode], (num_dim, num_cell)).transpose()
    
    A_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim, 10],dtype='complex')
    for i_dim in range(num_dim):
      for i_mode in range(10):
        A_3D[:, :, :, i_dim, i_mode] = \
          arr_conv.array_1D_to_3D(\
            xi, eta, zeta, num_cell,\
            A_1D[:, i_dim, i_mode])
              
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,0,i],
          options.output_filename_prefix + 'reconstruction_dmdr5_x'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
    for i in range(0,10):
        plot_contour(
          cell_centroid[:,:,0,0],
          cell_centroid[:,:,0,1],
          A_3D[:,:,0,1,i],
          options.output_filename_prefix + 'reconstruction_dmdr5_y'+str(i)+'.png',
          options.plot.contour.levels,
          options.plot.contour.vmin,
          options.plot.contour.vmax)
    # Save the output
    '''
  
#------------------------------------------------------------------------------
if __name__ == "__main__":
  sys.exit(main())

