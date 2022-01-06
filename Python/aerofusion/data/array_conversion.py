# -----------------------------------------------------------------------------
# \file array_conversion.py
# \brief Converts numpy arrays between 1D and 3D format using mesh indices
#
# -----------------------------------------------------------------------------

import numpy as np

# -----------------------------------------------------------------------------

def array_3D_to_1D(xi, eta, zeta, Ncell, data_field):
  data_field_1D_array = np.zeros([Ncell])
  data_field_1D_array[:] = data_field[xi[:], eta[:], zeta[:]]

  return (data_field_1D_array)

# -----------------------------------------------------------------------------

def array_1D_to_3D(xi, eta, zeta, Nxi, Neta, Nzeta, data_field):

  # Get range of mesh_index_xi_array,eta,zeta
  data_field_3D_array = np.zeros([Nxi,Neta,Nzeta])
  data_field_3D_array[xi[:], eta[:], zeta[:]] = data_field[:]

  return (data_field_3D_array)

def array_2D_to_1D(xi, eta, Ncell, data_field):
  data_field_1D_array = np.zeros([Ncell])
  data_field_1D_array[:] = data_field[xi[:], eta[:]]

  return (data_field_1D_array)

def array_1D_to_2D(xi, eta, Nxi, Neta, data_field):

  # Get range of mesh_index_xi_array,eta,zeta
  data_field_2D_array = np.zeros([Nxi,Neta])
  data_field_2D_array[xi[:], eta[:]] = data_field[:]

  return (data_field_2D_array)
