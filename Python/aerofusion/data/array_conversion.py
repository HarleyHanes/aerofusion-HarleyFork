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

def array_1D_to_compact(data_1D):
    if data_1D.ndim == 3:
        num_cell = data_1D.shape[0]
        num_dim = data_1D.shape[1]
        n_snap = data_1D.shape[2]
        data_1D_compact = np.zeros([num_cell * num_dim, n_snap])
        for i_snap in range(n_snap):
          data_1D_compact[:,i_snap] = \
            np.reshape((data_1D[:, :, i_snap]).transpose(), (num_dim*num_cell))
    elif data_1D.ndim == 2:
        num_cell = data_1D.shape[0]
        num_dim = data_1D.shape[1]
        data_1D_compact = \
            np.reshape((data_1D).transpose(), (num_dim*num_cell))
    else : 
        raise Exception(str(data_1D.ndim) + " dimensions in 1_D data found (expected 2 or 3)")
    return data_1D_compact

def array_compact_to_1D(data_1D_compact, num_dim, num_cell):
    if data_1D_compact.ndim == 2:
        n_snap = data_1D_compact.shape[1]
        data_1D = np.zeros([num_cell, num_dim, n_snap])
        for i_snap in range(n_snap):
          data_1D[:,:,i_snap] = \
            np.reshape(data_1D_compact[:, i_snap], (num_dim, num_cell)).transpose()
    elif data_1D_compact.ndim == 1:
          data_1D = \
            np.reshape(data_1D_compact, (num_dim, num_cell)).transpose()
    else :
        raise Exception("More than two dimensions in the 1D_compact array")
    return data_1D

def test_compact_to_1D():
    data_1D=np.empty((3,2,3))
    snap = np.array([[1,-1], [2,-2], [3,-3]])
    data_1D[:,:,0]=snap
    data_1D[:,:,1]=2*snap
    data_1D[:,:,2]=3*snap
    data_1D_compact = array_1D_to_compact(data_1D)
    error = data_1D - array_compact_to_1D(data_1D_compact, 2,3)
    assert np.all(error == 0)

