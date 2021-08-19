# -----------------------------------------------------------------------------
# \file derivatives_curvilinar_grid.py
# \brief Calculate derivatives on curvilinear grids using findiff
# -----------------------------------------------------------------------------

import numpy as np
import findiff

from aerofusion.data.array_conversion import array_3D_to_1D

# -----------------------------------------------------------------------------
# Note: - We assume curvilinear and Cartisian grid have one to one mapping
#       - For now we are sending only one dimension of variable
def jacobian_of_grid_2d(xi, eta, zeta, cell_center, accuracy):

  var_dim = cell_center.shape
  num_xi = var_dim[0]
  num_eta = var_dim[1]
  num_zeta = var_dim[2]
  num_dim = var_dim[3]
  num_cell = num_xi * num_eta
  # dx/dxi, dx/deta, dy/dxi, dy/deta
  xi_delta = 1
  eta_delta = 1
  dx_dxi = np.zeros([num_xi, num_eta, num_zeta])
  dy_dxi = np.zeros([num_xi, num_eta, num_zeta])
  d_dxi = findiff.FinDiff(0, xi_delta, acc=accuracy)
  for i_eta in range(num_eta):
    dx_dxi[:,i_eta,0] = d_dxi(cell_center[:, i_eta, 0, 0])
    dy_dxi[:,i_eta,0] = d_dxi(cell_center[:, i_eta, 0, 1])

  dx_deta = np.zeros([num_xi, num_eta, num_zeta])
  dy_deta = np.zeros([num_xi, num_eta, num_zeta])
  d_deta = findiff.FinDiff(0, eta_delta, acc=accuracy)
  for i_xi in range(num_xi):
    dx_deta[i_xi, :, 0] = d_deta(cell_center[i_xi, :, 0, 0])
    dy_deta[i_xi, :, 0] = d_deta(cell_center[i_xi, :, 0, 1])

  dx_dxi_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dx_dxi)
  dy_dxi_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dy_dxi)
  dx_deta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dx_deta)
  dy_deta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dy_deta)

  jacobian = np.zeros([2, 2, num_cell])
  # j00 = dxi/dx j01 = dxi/dy j10 = deta/dx j11 = deta/dy
  for i_cell in range(num_cell):
    det = dx_dxi_1D[i_cell]*dy_deta_1D[i_cell] - \
          dx_deta_1D[i_cell]*dy_dxi_1D[i_cell]
    inv_det = 1.0 / det
    jacobian[0, 0, i_cell] =   inv_det * dy_deta_1D[i_cell]
    jacobian[0, 1, i_cell] = - inv_det * dx_deta_1D[i_cell]
    jacobian[1, 0, i_cell] = - inv_det * dy_dxi_1D[i_cell]
    jacobian[1, 1, i_cell] =   inv_det * dx_dxi_1D[i_cell]

  return(jacobian)

# -----------------------------------------------------------------------------
def jacobian_of_grid_3d(xi, eta, zeta, num_cell, cell_center, accuracy):

  var_dim = cell_center.shape
  num_xi = var_dim[0]
  num_eta = var_dim[1]
  num_zeta = var_dim[2]
  num_dim = var_dim[3]
  # dx/dxi, dx/deta, dx/dzeta, dy/dxi, dy/deta, dy/dzeta, dz/dxi, dz/deta, dz/dzeta
  xi_delta = 1
  eta_delta = 1
  zeta_delta = 1
  dx_dxi = np.zeros([num_xi, num_eta, num_zeta])
  dy_dxi = np.zeros([num_xi, num_eta, num_zeta])
  dz_dxi = np.zeros([num_xi, num_eta, num_zeta])
  d_dxi = findiff.FinDiff(0, xi_delta, acc=accuracy)

  # d/dxi
  for i_zeta in range(num_zeta):
    for i_eta in range(num_eta):
      dx_dxi[:,i_eta,i_zeta]   = d_dxi(cell_center[:, i_eta, i_zeta, 0])
      dy_dxi[:,i_eta, i_zeta]  = d_dxi(cell_center[:, i_eta, i_zeta, 1])
      dz_dxi[:, i_eta, i_zeta] = d_dxi(cell_center[:, i_eta, i_zeta, 2])

  dx_deta = np.zeros([num_xi, num_eta, num_zeta])
  dy_deta = np.zeros([num_xi, num_eta, num_zeta])
  dz_deta = np.zeros([num_xi, num_eta, num_zeta])
  d_deta = findiff.FinDiff(0, eta_delta, acc=accuracy)

  # d/deta
  for i_zeta in range(num_zeta):
    for i_xi in range(num_xi):
      dx_deta[i_xi, :, i_zeta] = d_deta(cell_center[i_xi, :, i_zeta, 0])
      dy_deta[i_xi, :, i_zeta] = d_deta(cell_center[i_xi, :, i_zeta, 1])
      dz_deta[i_xi, :, i_zeta] = d_deta(cell_center[i_xi, :, i_zeta, 2])

  dx_dzeta = np.zeros([num_xi, num_eta, num_zeta])
  dy_dzeta = np.zeros([num_xi, num_eta, num_zeta])
  dz_dzeta = np.zeros([num_xi, num_eta, num_zeta])
  d_dzeta = findiff.FinDiff(0, zeta_delta, acc=accuracy)

  # d/dzeta
  for i_eta in range(num_eta):
    for i_xi in range(num_xi):
      dx_dzeta[i_xi, i_eta, :] = d_dzeta(cell_center[i_xi, i_eta, :, 0])
      dy_dzeta[i_xi, i_eta, :] = d_dzeta(cell_center[i_xi, i_eta, :, 1])
      dz_dzeta[i_xi, i_eta, :] = d_dzeta(cell_center[i_xi, i_eta, :, 2])


  dx_dxi_1D   = array_3D_to_1D(xi, eta, zeta, num_cell, dx_dxi)
  dy_dxi_1D   = array_3D_to_1D(xi, eta, zeta, num_cell, dy_dxi)
  dz_dxi_1D   = array_3D_to_1D(xi, eta, zeta, num_cell, dz_dxi)
  dx_deta_1D  = array_3D_to_1D(xi, eta, zeta, num_cell, dx_deta)
  dy_deta_1D  = array_3D_to_1D(xi, eta, zeta, num_cell, dy_deta)
  dz_deta_1D  = array_3D_to_1D(xi, eta, zeta, num_cell, dz_deta)
  dx_dzeta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dx_dzeta)
  dy_dzeta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dy_dzeta)
  dz_dzeta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dz_dzeta)

  jacobian = np.zeros([3, 3, num_cell])

  # j00 = dxi/dx, j01 = dxi/dy j10=, j02 = dxi/dz
  # j10 = deta / dx, j11 = deta / dy, j12 = deta/dz,
  # j20 = dzeta / dx, j21 = dzeta / dy, j22 = dzeta/dz,
  for i_cell in range(num_cell):
    det = \
      dx_dxi_1D[i_cell] * dy_deta_1D[i_cell] * dz_dzeta_1D[i_cell] + \
      dy_dxi_1D[i_cell] * dz_deta_1D[i_cell] * dx_dzeta_1D[i_cell] + \
      dz_dxi_1D[i_cell] * dx_deta_1D[i_cell] * dy_dzeta_1D[i_cell] - \
      dz_dxi_1D[i_cell] * dy_deta_1D[i_cell] * dx_dzeta_1D[i_cell] - \
      dy_dxi_1D[i_cell] * dx_deta_1D[i_cell] * dz_dzeta_1D[i_cell] - \
      dx_dxi_1D[i_cell] * dz_deta_1D[i_cell] * dy_dzeta_1D[i_cell]
    inv_det = 1.0 / det

    jacobian[0, 0, i_cell] = \
      inv_det * (dy_deta_1D[i_cell] * dz_dzeta_1D[i_cell] - \
                 dz_deta_1D[i_cell]*dy_dzeta_1D[i_cell])
    jacobian[0, 1, i_cell] = \
      inv_det * (dx_dzeta_1D[i_cell] * dz_deta_1D[i_cell] - \
                 dx_deta_1D[i_cell] * dz_dzeta_1D[i_cell])
    jacobian[0, 2, i_cell] = \
      inv_det * (dx_deta_1D[i_cell] * dy_dzeta_1D[i_cell] - \
                 dx_dzeta_1D[i_cell] * dy_deta_1D[i_cell])
    jacobian[1, 0, i_cell] = \
      inv_det * (dy_dzeta_1D[i_cell] * dz_dxi_1D[i_cell] - \
                 dy_dxi_1D[i_cell] * dz_dzeta_1D[i_cell])
    jacobian[1, 1, i_cell] = \
      inv_det * (dx_dxi_1D[i_cell] * dz_dzeta_1D[i_cell] - \
                 dx_dzeta_1D[i_cell] * dz_dxi_1D[i_cell])
    jacobian[1, 2, i_cell] = \
      inv_det * (dx_dxi_1D[i_cell] * dy_dzeta_1D[i_cell] - \
                 dx_dzeta_1D[i_cell] * dy_dxi_1D[i_cell])
    jacobian[2, 0, i_cell] = \
      inv_det * (dy_dxi_1D[i_cell] * dz_deta_1D[i_cell] - \
                 dy_deta_1D[i_cell] * dz_dxi_1D[i_cell])
    jacobian[2, 1, i_cell] = \
      inv_det * (dx_deta_1D[i_cell] * dz_dxi_1D[i_cell] - \
                 dx_dxi_1D[i_cell] * dz_deta_1D[i_cell])
    jacobian[2, 2, i_cell] = \
      inv_det * (dx_dxi_1D[i_cell] * dy_deta_1D[i_cell] - \
                 dx_deta_1D[i_cell] * dy_dxi_1D[i_cell])

  # print('shape of jacobian', jacobian.shape)
  return(jacobian)

# -----------------------------------------------------------------------------
def derivative(var, xi, eta, zeta, jacobian, accuracy):

  var_dim = var.shape
  num_xi = var_dim[0]
  num_eta = var_dim[1]
  num_zeta = var_dim[2]
  num_dim = var_dim[3]
  num_cell = num_xi * num_eta

  # dvar/dxi, dvar/deta
  xi_delta = 1
  eta_delta = 1
  dvar_dxi = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  dvar_deta = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  d_dxi = findiff.FinDiff(0, xi_delta, acc=accuracy)
  d_deta = findiff.FinDiff(0, eta_delta, acc=accuracy)

  #donyaFuture change this part to one loop instead of 2
  # var[num_dof*nxi, num_dof*neta, num_dof*neta]
  for i_dim in range(num_dim):
    for i_eta in range(num_eta):
      dvar_dxi[:,i_eta,0,i_dim] = d_dxi(var[:,i_eta,0,i_dim])
    for i_xi in range(num_xi):
      dvar_deta[i_xi, :, 0, i_dim] = d_deta(var[i_xi, :, 0, i_dim])

  dvar_dxi_1D = np.zeros([num_cell, num_dim])
  dvar_deta_1D = np.zeros([num_cell, num_dim])

  #donyaFuture change this from 2 loops of n_dim and n_snap to one loop of n_dof
  for i_dim in range(num_dim):
    dvar_dxi_1D[:,i_dim] = \
      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_dxi[:, :, :, i_dim])
    dvar_deta_1D[:,i_dim] = \
      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_deta[:, :, :, i_dim])

  #donyaFuture change this from 2 loops of n_dim and n_snap to one loop of n_dof
  # dvar/dx = dvar/dxi*dxi/dx + dvar/deta*deta/dx
  dvar_dx = np.zeros([num_cell, num_dim])
  dvar_dy = np.zeros([num_cell, num_dim])
  for i_dim in range(num_dim):
    for i_cell in range(num_cell):
      dvar_dx[i_cell, i_dim] = \
        dvar_dxi_1D[i_cell, i_dim]  * jacobian[0,0,i_cell] + \
        dvar_deta_1D[i_cell, i_dim] * jacobian[1, 0, i_cell]
      dvar_dy[i_cell, i_dim] = \
        dvar_dxi_1D[i_cell, i_dim]  * jacobian[0, 1, i_cell] + \
        dvar_deta_1D[i_cell, i_dim] * jacobian[1, 1, i_cell]

  return(dvar_dx, dvar_dy)

# -----------------------------------------------------------------------------
def derivative_3d(var, xi, eta, zeta, num_cell, jacobian, accuracy):
  # structure of input is [num_xi, num_eta, num_zeta, num_dof]
  # structure of output is [num_cell, num_dof]
  var_dim = var.shape
  num_xi = var_dim[0]
  num_eta = var_dim[1]
  num_zeta = var_dim[2]
  num_dof = var_dim[3]

  # dvar/dxi, dvar/deta, dvar_dzeta
  xi_delta = 1
  eta_delta = 1
  zeta_delta = 1
  # var_xi = [i_xi, (num_dof*num_zeta)*i_eta + (num_dof)*i_zeta + i_dof)]
  var_xi = np.reshape(var, (num_xi, num_eta * num_zeta*num_dof))
  # var_eta = [i_eta, (num_dof*num_zeta)*i_xi + (num_dof)*i_zeta + i_dof)]
  var_eta = np.reshape(var.transpose(1, 0, 2, 3),
                       (num_eta, num_xi * num_zeta * num_dof))
  # var_zeta = [i_zeta, (num_dof*num_xi)*i_eta + (num_dof)*i_xi + i_dof)]
  var_zeta = np.reshape(var.transpose(2, 1, 0, 3),
                        (num_zeta, num_xi * num_eta * num_dof))

  d_dxi = findiff.FinDiff(0, xi_delta, acc=accuracy)
  d_deta = findiff.FinDiff(0, eta_delta, acc=accuracy)
  d_dzeta = findiff.FinDiff(0, zeta_delta, acc=accuracy)

  dvar_dxi = np.zeros([num_xi, num_eta*num_zeta*num_dof])
  for i_xi_dof in range(num_eta*num_zeta*num_dof):
    dvar_dxi[:,i_xi_dof] =  d_dxi(var_xi[:,i_xi_dof])

  dvar_deta = np.zeros([num_eta, num_xi * num_zeta * num_dof])
  for i_eta_dof in range(num_xi * num_zeta * num_dof):
    dvar_deta[:,i_eta_dof] = d_deta(var_eta[:, i_eta_dof])

  dvar_dzeta = np.zeros([num_zeta, num_xi * num_eta * num_dof])
  for i_zeta_dof in range(num_xi * num_eta * num_dof):
    dvar_dzeta[:,i_zeta_dof] = d_deta(var_zeta[:, i_zeta_dof])

  # reshape derivative to 3d
  # var_xi = [i_xi, (num_dof*num_zeta)*i_eta + (num_dof)*i_zeta + i_dof)]
  dvar_dxi = np.reshape(dvar_dxi, (num_xi, num_eta, num_zeta, num_dof))

  # var_eta = [i_eta, (num_dof*num_zeta)*i_xi + (num_dof)*i_zeta + i_dof)]
  dvar_deta = np.reshape(dvar_deta, (num_eta, num_xi, num_zeta, num_dof))
  dvar_deta = dvar_deta.transpose(1, 0, 2, 3)

  # var_zeta = [i_zeta, (num_dof*num_xi)*i_eta + (num_dof)*i_xi + i_dof)]
  dvar_dzeta = np.reshape(dvar_dzeta, (num_zeta, num_eta, num_xi, num_dof))
  dvar_dzeta = dvar_dzeta.transpose(2, 1, 0, 3)

  dvar_dxi_1D = np.zeros([num_cell, num_dof])
  dvar_deta_1D = np.zeros([num_cell, num_dof])
  dvar_dzeta_1D = np.zeros([num_cell, num_dof])

  #donyaFuture change this from 2 loops of n_dim and n_snap to one loop of n_dof
  for i_dof in range(num_dof):
    dvar_dxi_1D[:,i_dof]    = \
      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_dxi[:, :, :, i_dof])
    dvar_deta_1D[:,i_dof]   = \
      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_deta[:, :, :, i_dof])
    dvar_dzeta_1D[:, i_dof] = \
      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_dzeta[:, :, :, i_dof])

  #donyaFuture change this from 2 loops of n_dim and n_snap to one loop of n_dof
  # dvar/dx = dvar/dxi*dxi/dx + dvar/deta*deta/dx
  # j00 = dxi/dx, j01 = dxi/dy , j02 = dxi/dz
  # j10 = deta / dx, j11 = deta / dy, j12 = deta/dz,
  # j20 = dzeta / dx, j21 = dzeta / dy, j22 = dzeta/dz,
  dvar_dx = np.zeros([num_cell, num_dof])
  dvar_dy = np.zeros([num_cell, num_dof])
  dvar_dz = np.zeros([num_cell, num_dof])
  for i_dof in range(num_dof):
   for i_cell in range(num_cell):
     dvar_dx[i_cell, i_dof] = \
       dvar_dxi_1D[i_cell, i_dof]   * jacobian[0, 0, i_cell] + \
       dvar_deta_1D[i_cell, i_dof]  * jacobian[1, 0, i_cell] + \
       dvar_dzeta_1D[i_cell, i_dof] * jacobian[2, 0, i_cell]
     dvar_dy[i_cell, i_dof] = \
       dvar_dxi_1D[i_cell, i_dof]   * jacobian[0, 1, i_cell] + \
       dvar_deta_1D[i_cell, i_dof]  * jacobian[1, 1, i_cell] + \
       dvar_dzeta_1D[i_cell, i_dof] * jacobian[2, 1, i_cell]
     dvar_dz[i_cell, i_dof] = \
       dvar_dxi_1D[i_cell, i_dof]   * jacobian[0, 2, i_cell] + \
       dvar_deta_1D[i_cell, i_dof]  * jacobian[1, 2, i_cell] + \
       dvar_dzeta_1D[i_cell, i_dof] * jacobian[2, 2, i_cell]

  return(dvar_dx, dvar_dy, dvar_dz)
