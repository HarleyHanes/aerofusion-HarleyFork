# -----------------------------------------------------------------------------
# \file array_conversion.py
# \brief Converts numpy arrays between 1D and 3D format using mesh indices
#
# -----------------------------------------------------------------------------

import numpy as np

# -----------------------------------------------------------------------------

def array_3D_to_1D(xi_index, eta_index, zeta_index, Ncell, data_field):

  # Get range of mesh_index_xi_array,eta,zeta
  mesh_index_xi_range   = [min(xi_index), max(xi_index)]
  mesh_index_eta_range  = [min(eta_index), max(eta_index)]
  mesh_index_zeta_range = [min(zeta_index), max(zeta_index)]

  data_field_1D_array = np.zeros([Ncell])

  for cdx in range(Ncell):
    idx = xi_index[cdx]   - mesh_index_xi_range[0]
    jdx = eta_index[cdx]  - mesh_index_eta_range[0]
    kdx = zeta_index[cdx] - mesh_index_zeta_range[0]
    data_field_1D_array[cdx] = data_field[idx, jdx, kdx]

  return (data_field_1D_array)

# -----------------------------------------------------------------------------

def array_1D_to_3D(xi_index, eta_index, zeta_index, Ncell, data_field):

  # Get range of mesh_index_xi_array,eta,zeta
  mesh_index_xi_range   = [min(xi_index), max(xi_index)]
  mesh_index_eta_range  = [min(eta_index), max(eta_index)]
  mesh_index_zeta_range = [min(zeta_index), max(zeta_index)]

  Nxi   = max(xi_index)   - min(xi_index) +1
  Neta  = max(eta_index)  - min(eta_index) +1
  Nzeta = max(zeta_index) - min(zeta_index) +1
  data_field_3D_array = np.zeros([Nxi,Neta,Nzeta])

  for cdx in range(Ncell):
    idx = xi_index[cdx]   - mesh_index_xi_range[0]
    jdx = eta_index[cdx]  - mesh_index_eta_range[0]
    kdx = zeta_index[cdx] - mesh_index_zeta_range[0]
    data_field_3D_array[idx, jdx, kdx] = data_field[cdx]

  return (data_field_3D_array)
