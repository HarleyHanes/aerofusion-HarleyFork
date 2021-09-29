# \file plot_list_of_npz_files.py
# \brief Converts data from solution files written in uPDE hdf5 format to npz
#
# To run, issue the command:
#   python3 plot_list_of_npz_files.py \
#           plot_list_of_npz_files.cfg
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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
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
# Plot contents of npz files
def plot_npz_file(mesh_filename, 
                  fields_filename,
                  output_filename,
                  options):

  # Figure size
  figsize = options["figsize"]
  # number of subplots
  num_subplots = options["num_subplots"]
  # Index of plane normal to z to plot
  zdx = options["zdx"]
  # Range of y values
  y_min = options["y_min"]
  y_max = options["y_max"]
  # Pad of colorbar
  colorbar_pad = options["colorbar_pad"]
  # Number of ticks in colorbars
  num_colorbar_ticks = 3
  # Titles of subplots
  axs_titles = options["axs_titles"]
  # Labels of colorbars
  colorbar_labels = options["colorbar_labels"]
  # Color map ranges
  colorbar_ranges = options["colorbar_ranges"]
  # Color maps for plots
  colormaps = options["colormaps"]
  # Levels of contour maps
  num_contour_levels = options["num_contour_levels"]
  
  # Read data from input files
  mesh_data = np.load(mesh_filename)
  cell_volume     = mesh_data['cell_volume']
  cell_centroid   = mesh_data['cell_centroid']
  xi              = mesh_data['xi_index']
  eta             = mesh_data['eta_index']
  zeta            = mesh_data['zeta_index']
  mesh_xi_index_range   = mesh_data["mesh_xi_index_range"]
  mesh_eta_index_range  = mesh_data["mesh_eta_index_range"] 
  mesh_zeta_index_range = mesh_data["mesh_zeta_index_range"] 
  
  num_xi   = mesh_xi_index_range[1]   - mesh_xi_index_range[0]   + 1
  num_eta  = mesh_eta_index_range[1]  - mesh_eta_index_range[0]  + 1
  num_zeta = mesh_zeta_index_range[1] - mesh_zeta_index_range[0] + 1
  cell_centroid_3D = cell_centroid.reshape((num_xi, num_eta, num_zeta, 3))
  x = cell_centroid_3D[:,:,:,0]
  y = cell_centroid_3D[:,:,:,1]
  z = cell_centroid_3D[:,:,:,2]
  fields_data = np.load(fields_filename)
  fields = [fields_data['velocity_1D_x'].reshape((num_xi, num_eta, num_zeta)),
            fields_data['velocity_1D_y'].reshape((num_xi, num_eta, num_zeta)),
            fields_data['velocity_1D_z'].reshape((num_xi, num_eta, num_zeta))]
  
  # Create figure
  fig = plt.figure(figsize = figsize)
  tick_locator = ticker.MaxNLocator(nbins=num_colorbar_ticks)
  # Create subplots axes sharing y axis
  axs = fig.subplots(1,num_subplots, sharey = True)
  # Plot contours
  cs    = [None] * num_subplots
  cbars = [None] * num_subplots
  axs[0].set(ylabel='y')
  for adx in range(3):
    colorbar_vmin = colorbar_ranges[adx][0]
    if colorbar_vmin == "auto":
      # set automatically
      colorbar_vmin = fields[adx][:,:,zdx].min()
    colorbar_vmax = colorbar_ranges[adx][1]
    if colorbar_vmax == "auto":
      # set automatically
      colorbar_vmax = fields[adx][:,:,zdx].max()
    levels = np.linspace(colorbar_vmin, colorbar_vmax, num_contour_levels)
    cs[adx] = axs[adx].contourf(x[:,:,zdx], y[:,:,zdx], fields[adx][:,:,zdx],
                                levels,
                                cmap = colormaps[adx],
                                vmin = colorbar_vmin,
                                vmax = colorbar_vmax)
    cbars[adx] = fig.colorbar(cs[adx], ax = axs[adx], orientation="horizontal",
                              pad = colorbar_pad)
    cbars[adx].locator = tick_locator
    cbars[adx].update_ticks()
    cbars[adx].ax.set_xlabel(colorbar_labels[0])
    cbars[adx].set_clim(colorbar_ranges[adx][0],colorbar_ranges[adx][1])
    axs[adx].set_ylim([y_min,y_max])
    axs[adx].set(xlabel='x', title=axs_titles[adx])
  fig.suptitle("Simulation time step: " + \
               str(int(fields_data["simulation_time_step"])) + \
               ", time :" + \
               str(fields_data["simulation_time"]))
  plt.tight_layout(rect=[0, 0.03, 1, 0.95])
  fig.savefig(output_filename)

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
  print("Read options -------------------------------------------------------")
  print(libconf.dumps(options))
  print("--------------------------------------------------------------------")

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

  # Loop in timesteps to obtain the snapshot matrix U
  for tdx, time_step in enumerate(tqdm(local_list_of_time_steps)):

    time_step_str = str(time_step)
    fields_filename = \
      options["npz_filename_prefix"] + \
      time_step_str.zfill(padding_zeros) + ".npz"
    output_filename = \
      options["output_filename_prefix"] + \
      time_step_str.zfill(padding_zeros) + ".png"
    plot_npz_file(options["npz_filename_prefix"] + "mesh.npz", 
                  fields_filename,
                  output_filename,
                  options["plot"])

#------------------------------------------------------------------------------
if __name__ == "__main__":
  sys.exit(main())


