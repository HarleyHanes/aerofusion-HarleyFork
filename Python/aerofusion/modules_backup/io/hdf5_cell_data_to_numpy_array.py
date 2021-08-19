#! \file hdf5_cell_data_to_numpy_array.py
#! \brief Methods to read cell data from hdf5 file into a numpy array
#!

import sys
import os
import numpy as np
import h5py
import logging
from numba import jit

# -----------------------------------------------------------------------------

def read(input_filename,
         *args,
         **kwargs):

  # Default values
  options = \
    {
    "fields_to_read"  : "all",
    "xi_min" : "auto",
    "xi_max" : "auto",
    "eta_min" : "auto",
    "eta_max" : "auto",
    "zeta_min" : "auto",
    "zeta_max" : "auto",
    }
  # Update with kwargs
  for key, value in kwargs.items():
    options[key] = value

  # Check that file exists
  if os.path.isfile(input_filename) == False:
    print("File", input_filename, "does not exist", file = sys.stderr)
    exit(1)

  # Check that mesh_index_xi,eta,zeta fields are requested or add them to list
  fields_to_read = options["fields_to_read"].split()
  for field_name in ["mesh_index_xi", "mesh_index_eta", "mesh_index_zeta"]:
    if field_name not in fields_to_read:
      fields_to_read.append(field_name)

  h5_data = h5py.File(input_filename, "r")

  # Get simulation timestep and time
  simulation_timestep = float(h5_data["/"].attrs["time_step"])
  simulation_time = float(h5_data["/"].attrs["time"])

  # Get the data
  number_of_nodes_of_cell_array = \
    h5_data["mesh"]["number_of_nodes_of_cell"][()]
  number_of_cells         = len(number_of_nodes_of_cell_array)
  cell_id_array           = h5_data["mesh"]["cell_id"][()]
  node_of_cell_id_array   = h5_data["mesh"]["node_of_cell_id"][()]
  node_coor_array         = h5_data["mesh"]["node_coor"][()]
  cell_centroid           = h5_data["mesh"]["cell_centroid"][()]
  cell_volume             = h5_data["mesh"]["cell_geometric_size"][()]

  # Check that mesh_index_xi,eta,zeta fields were read
  mesh_index_xi_array   = h5_data["fields"]["cell_data"]["mesh_index_xi"][()]
  mesh_index_eta_array  = h5_data["fields"]["cell_data"]["mesh_index_eta"][()]
  mesh_index_zeta_array = h5_data["fields"]["cell_data"]["mesh_index_zeta"][()]
  if mesh_index_xi_array is None  or \
     mesh_index_eta_array is None or \
     mesh_index_zeta_array is None:
    logging.error("Missing 'mesh_index_xi,eta,zeta'.")
    return None

  # Get range of mesh_index_xi_array,eta,zeta
  min_mesh_index_xi_array   = min(mesh_index_xi_array)
  max_mesh_index_xi_array   = max(mesh_index_xi_array)
  min_mesh_index_eta_array  = min(mesh_index_eta_array)
  max_mesh_index_eta_array  = max(mesh_index_eta_array)
  min_mesh_index_zeta_array = min(mesh_index_zeta_array)
  max_mesh_index_zeta_array = max(mesh_index_zeta_array)
  if options["xi_min"] != "auto":
    min_mesh_index_xi_array = options["xi_min"]
  if options["xi_max"] != "auto":
    max_mesh_index_xi_array = options["xi_max"]
  if options["eta_min"] != "auto":
    min_mesh_index_eta_array = options["eta_min"]
  if options["eta_max"] != "auto":
    max_mesh_index_eta_array = options["eta_max"]
  if options["zeta_min"] != "auto":
    min_mesh_index_zeta_array = options["zeta_min"]
  if options["zeta_max"] != "auto":
    max_mesh_index_zeta_array = options["zeta_max"]
    
  mesh_index_xi_range   = \
    [min_mesh_index_xi_array, max_mesh_index_xi_array]
  mesh_index_eta_range  = \
    [min_mesh_index_eta_array, max_mesh_index_eta_array]
  mesh_index_zeta_range = \
    [min_mesh_index_zeta_array, max_mesh_index_zeta_array]
  print("mesh_index_{xi,eta,zeta}_range",
        mesh_index_xi_range,
        mesh_index_eta_range,
        mesh_index_zeta_range)

  shape_3D = [ 1 + mesh_index_xi_range[1]  - mesh_index_xi_range[0],
               1 + mesh_index_eta_range[1] - mesh_index_eta_range[0],
               1 + mesh_index_zeta_range[1] - mesh_index_zeta_range[0] ]
  skip_field_names = ["mesh_index_xi_array",
                      "mesh_index_eta_array",
                      "mesh_index_zeta_array"]

  data_fields_1D_arrays = {}
  data_fields_3D_arrays = {}

  cell_centroid_1D_array = []
  cell_volume_1D_array = []
  cell_centroid_3D_array = \
    np.empty([shape_3D[0], shape_3D[1], shape_3D[2], 3], dtype = np.float64)
  cell_volume_3D_array = \
    np.empty([shape_3D[0], shape_3D[1], shape_3D[2]], dtype = np.float64)
  for cdx in range(len(cell_centroid)):
    if mesh_index_xi_array[cdx]   >= min_mesh_index_xi_array   and \
       mesh_index_xi_array[cdx]   <= max_mesh_index_xi_array   and \
       mesh_index_eta_array[cdx]  >= min_mesh_index_eta_array  and \
       mesh_index_eta_array[cdx]  <= max_mesh_index_eta_array  and \
       mesh_index_zeta_array[cdx] >= min_mesh_index_zeta_array and \
       mesh_index_zeta_array[cdx] <= max_mesh_index_zeta_array:
      idx = mesh_index_xi_array[cdx]   - min_mesh_index_xi_array
      jdx = mesh_index_eta_array[cdx]  - min_mesh_index_eta_array
      kdx = mesh_index_zeta_array[cdx] - min_mesh_index_zeta_array
      cell_centroid_3D_array[idx,jdx,kdx] = cell_centroid[cdx]
      cell_centroid_1D_array.append(cell_centroid[cdx])
      cell_volume_3D_array[idx,jdx,kdx] = cell_volume[cdx]
      cell_volume_1D_array.append(cell_volume[cdx])
  data_fields_3D_arrays["cell_centroid"] = cell_centroid_3D_array
  data_fields_1D_arrays["cell_centroid"] = np.array(cell_centroid_1D_array)
  data_fields_3D_arrays["cell_volume"]   = cell_volume_3D_array
  data_fields_1D_arrays["cell_volume"]   = np.array(cell_volume_1D_array)

  for field_name in h5_data["fields"]["cell_data"]:
    if (fields_to_read == "all" or \
        field_name in fields_to_read) and \
       field_name not in skip_field_names:
      data_field = h5_data["fields"]["cell_data"][field_name][()]
      if len(data_field.shape) == 1:
        data_field_3D_array = \
          np.empty([shape_3D[0], shape_3D[1], shape_3D[2]],
            dtype = data_field.dtype)
      elif len(data_field.shape) == 2:
        data_field_3D_array = \
          np.empty([shape_3D[0], shape_3D[1], shape_3D[2],
                    data_field.shape[-1]],
            dtype = data_field.dtype)
      else:
        logging.error("Unhandled shape '" + str(data_field.shape))
        return None
      data_field_1D_array = []
      for cdx in range(len(number_of_nodes_of_cell_array)):
        if mesh_index_xi_array[cdx]   >= min_mesh_index_xi_array   and \
           mesh_index_xi_array[cdx]   <= max_mesh_index_xi_array   and \
           mesh_index_eta_array[cdx]  >= min_mesh_index_eta_array  and \
           mesh_index_eta_array[cdx]  <= max_mesh_index_eta_array  and \
           mesh_index_zeta_array[cdx] >= min_mesh_index_zeta_array and \
           mesh_index_zeta_array[cdx] <= max_mesh_index_zeta_array:
          idx = mesh_index_xi_array[cdx]   - mesh_index_xi_range[0]
          jdx = mesh_index_eta_array[cdx]  - mesh_index_eta_range[0]
          kdx = mesh_index_zeta_array[cdx] - mesh_index_zeta_range[0]
          data_field_3D_array[idx,jdx,kdx] = data_field[cdx]
          data_field_1D_array.append(data_field[cdx])
      # Add to dictionary
      data_fields_1D_arrays[field_name] = np.array(data_field_1D_array)
      data_fields_3D_arrays[field_name] = data_field_3D_array

  return simulation_timestep, simulation_time, \
         data_fields_1D_arrays, data_fields_3D_arrays

# -----------------------------------------------------------------------------

