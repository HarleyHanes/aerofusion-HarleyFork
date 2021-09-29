# -----------------------------------------------------------------------------
# \file convert_solution_files_from_uPDE_to_npz.py
# \brief Converts data from solution files written in uPDE hdf5 format to npz
#
# To run, issue the command:
#   python3 convert_solution_files_from_uPDE_to_npz.py \
#           convert_solution_files_from_uPDE_to_npz.cfg
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
import matplotlib.pyplot as plt
from mpi4py import MPI
try:
  from tqdm import tqdm
except:
  # tqdm not present; avoid crashing and simply not report progress bar
  def tqdm(input_arg):
    return input_arg

# aerofusion modules
from aerofusion.io import hdf5_cell_data_to_numpy_array

# -----------------------------------------------------------------------------
# Partition a list among mpi_size returning the sublist of items for mpi_rank
def partition_list(list_to_partition, mpi_rank, mpi_size):
  problem_size = len(list_to_partition)
  partition_chunks = np.linspace(0, problem_size, mpi_size + 1, dtype=int)
  start_idx = partition_chunks[mpi_rank]
  end_idx   = partition_chunks[mpi_rank + 1]
  return list_to_partition[start_idx:end_idx]

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

  #---------------------------------------------------------------------------

  if options["use_mpi"]:
    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()
    print("Running in parallel with ", mpi_size, "processes")
  else:
    print("Running in serial mode")
    comm = None
    mpi_rank = 0
    mpi_size = 1

  # Only the main mpi_rank will print out read options
  if mpi_rank == 0:
    print("Read options -----------------------------------------------------")
    print(libconf.dumps(options))
    print("------------------------------------------------------------------")

  # Initialize filenames
  output_filename_prefix = options["output_filename_prefix"]
  padding_zeros          = options["padding_zeros"]
  list_of_time_steps = \
    [tdx for tdx in range(options["time_step_min"],
                          options["time_step_max"],
                          options["time_step_delta"])]
  simulation_time_step_array = np.zeros(len(list_of_time_steps))
  simulation_time_array = np.zeros(len(list_of_time_steps))

  # Only the main mpi_rank will create the output directory, if needed
  if mpi_rank == 0:
    # Write data to output file
    os.makedirs(Path(output_filename_prefix).parent, exist_ok=True)
  # Ensure the directory is created before all other processes get here (with
  # MPI Barrier)
  comm.Barrier()

  # Retrieve the list of time steps to process by this mpi_rank
  local_list_of_time_steps = \
    partition_list(list_of_time_steps, mpi_rank, mpi_size)

  print("mpi_rank", mpi_rank,
        "local_list_of_time_steps", local_list_of_time_steps)

  # Loop in timesteps to obtain the snapshot matrix U
  for tdx, time_step in enumerate(tqdm(local_list_of_time_steps)):

    print("mpi_rank", mpi_rank,
          "tdx", tdx,
          "time_step", time_step)
    time_step_str = str(time_step)
    input_hdf5_solution_file_name = \
      options["solution_files_prefix"] + \
      time_step_str.zfill(padding_zeros) + ".hdf5"

    # Get data fields as 1D arrays and 3D arrays as dictionaries of
    # - 1D arrays: use the unstructured-grid cell-based data
    # - 3D arrays: use the structured-grid underlying topology via the
    #   mesh_index_xi,eta,zeta
    simulation_time_step, simulation_time, \
      data_field_1D_arrays, data_field_3D_arrays  = \
        hdf5_cell_data_to_numpy_array.read(
          input_hdf5_solution_file_name,
          fields_to_read = "velocity",
          xi_min = options["mesh_xi_index_range"][0],
          xi_max = options["mesh_xi_index_range"][1],
          eta_min = options["mesh_eta_index_range"][0],
          eta_max = options["mesh_eta_index_range"][1],
          zeta_min = options["mesh_zeta_index_range"][0],
          zeta_max = options["mesh_zeta_index_range"][1])

    velocityData_1D = data_field_1D_arrays['velocity']
    velocity_3D = data_field_3D_arrays['velocity']
    dim_1D = velocityData_1D.shape
    simulation_time_step_array[0] = simulation_time_step
    simulation_time_array[0] = simulation_time

    output_solution_fields_npz_filename = \
      output_filename_prefix + \
      time_step_str.zfill(padding_zeros) + ".npz"
    np.savez(output_solution_fields_npz_filename,
             simulation_time_step = simulation_time_step,
             simulation_time      = simulation_time,
             velocity_1D_x        = velocityData_1D[:,0],
             velocity_1D_y        = velocityData_1D[:,1],
             velocity_1D_z        = velocityData_1D[:,2])

  # Only the main mpi_rank will write the converted mesh file
  if mpi_rank == 0:
    # Write mesh data from the last solution_file that has been read
    #ivanFuture Note that this assumes that the mesh does not change between
    #ivanFuture time steps, which will not be the case for FSI problems
    output_solution_mesh_npz_filename = \
      output_filename_prefix + \
      "mesh" + ".npz"
    np.savez(output_solution_mesh_npz_filename,
             cell_centroid = data_field_1D_arrays['cell_centroid'],
             cell_volume   = data_field_1D_arrays['cell_volume'],
             xi_index      = data_field_1D_arrays['mesh_index_xi'],
             eta_index     = data_field_1D_arrays['mesh_index_eta'],
             zeta_index    = data_field_1D_arrays['mesh_index_zeta'],
             mesh_xi_index_range   = options["mesh_xi_index_range"],
             mesh_eta_index_range  = options["mesh_eta_index_range"], 
             mesh_zeta_index_range = options["mesh_zeta_index_range"]) 

#------------------------------------------------------------------------------
if __name__ == "__main__":
  sys.exit(main())
