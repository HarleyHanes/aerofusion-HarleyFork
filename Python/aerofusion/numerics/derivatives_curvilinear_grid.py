# -----------------------------------------------------------------------------
# \file derivatives_curvilinar_grid.py
# \brief Calculate derivatives on curvilinear grids using findiff
# -----------------------------------------------------------------------------

import numpy as np
import findiff
import time

from aerofusion.data.array_conversion import array_3D_to_1D
from aerofusion.numerics import derivatives as der

# -----------------------------------------------------------------------------
# Note: - We assume curvilinear and Cartisian grid have one to one mapping
#       - For now we are sending only one dimension of variable
def jacobian_of_grid_2d2(xi, eta, zeta, cell_center, accuracy):

  var_dim = cell_center.shape
  num_xi = var_dim[0]
  num_eta = var_dim[1]
  num_zeta = var_dim[2]
  #num_dim = var_dim[3]
  num_cell = num_xi * num_eta

  dx_dxi = np.zeros([num_xi, num_eta, num_zeta])
  dy_dxi = np.zeros([num_xi, num_eta, num_zeta])
  dy_deta = np.zeros([num_xi, num_eta, num_zeta])
  dx_deta = np.zeros([num_xi, num_eta, num_zeta])
  if accuracy==6:
      (dcell_dxi, dcell_deta) = der.FD_derivative_6th_order(cell_center)
  elif accuracy==4:
      (dcell_dxi, dcell_deta) = der.FD_derivative_4th_order(cell_center)
  else:
      (dcell_dxi, dcell_deta) = der.FD_derivative_2nd_order(cell_center)
  dx_dxi[:,:,:] = dcell_dxi[:,:,:,0]
  dy_dxi=dcell_dxi[:,:,:,1]
  dy_deta[:, :, :] = dcell_deta[:, :, :, 1]
  dx_deta[:, :, :] = dcell_deta[:, :, :, 0]

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
    jacobian[0, 1, i_cell] =  -inv_det * dx_deta_1D[i_cell]
    jacobian[1, 0, i_cell] =  -inv_det * dy_dxi_1D[i_cell]
    jacobian[1, 1, i_cell] =   inv_det * dx_dxi_1D[i_cell]

  return(jacobian)




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
###----output of the following is [3,3, num_xi, num_eta, num_zeta]
def jacobian_of_grid_3d_restructured(xi, eta, zeta, num_cell, cell_center, \
      accuracy_x, accuracy_y, accuracy_z):


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
  d_dxi = findiff.FinDiff(0, xi_delta, acc=accuracy_x)

  # d/dxi
  t_begin = time.time()
  for i_zeta in range(num_zeta):
    for i_eta in range(num_eta):
      dx_dxi[:,i_eta,i_zeta] = d_dxi(cell_center[:, i_eta, i_zeta, 0])
      dy_dxi[:,i_eta,i_zeta] = d_dxi(cell_center[:, i_eta, i_zeta, 1])
      dz_dxi[:,i_eta,i_zeta] = d_dxi(cell_center[:, i_eta, i_zeta, 2])
  t_end = time.time()
  print("DEBUG jacobian d/dxi", t_end - t_begin)

  dx_deta = np.zeros([num_xi, num_eta, num_zeta])
  dy_deta = np.zeros([num_xi, num_eta, num_zeta])
  dz_deta = np.zeros([num_xi, num_eta, num_zeta])
  d_deta = findiff.FinDiff(0, eta_delta, acc=accuracy_y)

  # d/deta
  t_begin = time.time()
  for i_zeta in range(num_zeta):
    for i_xi in range(num_xi):
      dx_deta[i_xi, :, i_zeta] = d_deta(cell_center[i_xi, :, i_zeta, 0])
      dy_deta[i_xi, :, i_zeta] = d_deta(cell_center[i_xi, :, i_zeta, 1])
      dz_deta[i_xi, :, i_zeta] = d_deta(cell_center[i_xi, :, i_zeta, 2])
  t_end = time.time()
  print("DEBUG jacobian d/deta", t_end - t_begin)

  dx_dzeta = np.zeros([num_xi, num_eta, num_zeta])
  dy_dzeta = np.zeros([num_xi, num_eta, num_zeta])
  dz_dzeta = np.zeros([num_xi, num_eta, num_zeta])
  d_dzeta = findiff.FinDiff(0, zeta_delta, acc=accuracy_z)

  # d/dzeta
  t_brgin = time.time()
  for i_eta in range(num_eta):
    for i_xi in range(num_xi):
      dx_dzeta[i_xi, i_eta, :] = d_dzeta(cell_center[i_xi, i_eta, :, 0])
      dy_dzeta[i_xi, i_eta, :] = d_dzeta(cell_center[i_xi, i_eta, :, 1])
      dz_dzeta[i_xi, i_eta, :] = d_dzeta(cell_center[i_xi, i_eta, :, 2])
  t_end = time.time() 
  print("DEBUG jacobian d/dzeta", t_end - t_begin)

 # t_begin = time.time()
 # dx_dxi_1D   = array_3D_to_1D(xi, eta, zeta, num_cell, dx_dxi)
 # dy_dxi_1D   = array_3D_to_1D(xi, eta, zeta, num_cell, dy_dxi)
 # dz_dxi_1D   = array_3D_to_1D(xi, eta, zeta, num_cell, dz_dxi)
 # dx_deta_1D  = array_3D_to_1D(xi, eta, zeta, num_cell, dx_deta)
 # dy_deta_1D  = array_3D_to_1D(xi, eta, zeta, num_cell, dy_deta)
 # dz_deta_1D  = array_3D_to_1D(xi, eta, zeta, num_cell, dz_deta)
 # dx_dzeta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dx_dzeta)
 # dy_dzeta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dy_dzeta)
 # dz_dzeta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dz_dzeta)
 # t_end = time.time()
 # print("DEBUG jacobian array_3D_to_1D", t_end -t_begin)

  #jacobian = np.zeros([3, 3, num_cell])
  jacobian = np.zeros([3, 3, num_xi, num_eta, num_zeta])

  # j00 = dxi/dx, j01 = dxi/dy j10=, j02 = dxi/dz
  # j10 = deta / dx, j11 = deta / dy, j12 = deta/dz,
  # j20 = dzeta / dx, j21 = dzeta / dy, j22 = dzeta/dz,
  #ivanComment Calculating det at once to be able to use slicing of numpy
  #ivanComment arrays rather than for-loops.
  #### changing 1d structure to 3d i
  det = np.zeros([num_xi, num_eta, num_zeta])
  t_begin = time.time()
  det[:,:,:] = \
    dx_dxi[:,:,:] * dy_deta[:,:,:] * dz_dzeta[:,:,:] + \
    dy_dxi[:,:,:] * dz_deta[:,:,:] * dx_dzeta[:,:,:] + \
    dz_dxi[:,:,:] * dx_deta[:,:,:] * dy_dzeta[:,:,:] - \
    dz_dxi[:,:,:] * dy_deta[:,:,:] * dx_dzeta[:,:,:] - \
    dy_dxi[:,:,:] * dx_deta[:,:,:] * dz_dzeta[:,:,:] - \
    dx_dxi[:,:,:] * dz_deta[:,:,:] * dy_dzeta[:,:,:]
 
  print('shape of determinant', det.shape)
  inv_det = 1.0 / det
  jacobian[0, 0, :,:,:] = \
    inv_det[:,:,:] * (dy_deta_1D[:,:,:]  * dz_dzeta_1D[:,:,:] - \
               dz_deta_1D[:,:,:]  * dy_dzeta_1D[:,:,:])
  jacobian[0, 1, :,:,:] = \
    inv_det[:,:,:] * (dx_dzeta_1D[:,:,:] * dz_deta_1D[:,:,:] - \
               dx_deta_1D[:,:,:]  * dz_dzeta_1D[:,:,:])
  jacobian[0, 2, :,:,:] = \
    inv_det[:,:,:] * (dx_deta_1D[:,:,:]  * dy_dzeta_1D[:,:,:] - \
               dx_dzeta_1D[:,:,:] * dy_deta_1D[:,:,:])
  jacobian[1, 0, :,:,:] = \
    inv_det[:,:,:] * (dy_dzeta_1D[:,:,:] * dz_dxi_1D[:,:,:] - \
               dy_dxi_1D[:,:,:]   * dz_dzeta_1D[:,:,:])
  jacobian[1, 1, :,:,:] = \
    inv_det[:,:,:] * (dx_dxi_1D[:,:,:]   * dz_dzeta_1D[:,:,:] - \
               dx_dzeta_1D[:,:,:] * dz_dxi_1D[:,:,:])
  jacobian[1, 2, :,:,:] = \
    inv_det[:,:,:] * (dx_dxi_1D[:,:,:]   * dy_dzeta_1D[:,:,:] - \
               dx_dzeta_1D[:,:,:] * dy_dxi_1D[:,:,:])
  jacobian[2, 0, :,:,:] = \
    inv_det[:,:,:] * (dy_dxi_1D[:,:,:]   * dz_deta_1D[:,:,:] - \
               dy_deta_1D[:,:,:]  * dz_dxi_1D[:,:,:])
  jacobian[2, 1, :, :,:] = \
    inv_det[:,:,:] * (dx_deta_1D[:,:,:]  * dz_dxi_1D[:,:,:] - \
               dx_dxi_1D[:,:,:]   * dz_deta_1D[:,:,:])
  jacobian[2, 2, :,:,:] = \
    inv_det[:,:,:] * (dx_dxi_1D[:,:,:]   * dy_deta_1D[:,:,:] - \
               dx_deta_1D[:,:,:]  * dy_dxi_1D[:,:,:])
  t_end = time.time()
  print("DEBUG creating jacobian", t_end - t_begin)
 # t_begin = time.time()
 # det = \
 #   dx_dxi_1D * dy_deta_1D * dz_dzeta_1D + \
 #   dy_dxi_1D * dz_deta_1D * dx_dzeta_1D + \
 #   dz_dxi_1D * dx_deta_1D * dy_dzeta_1D - \
 #   dz_dxi_1D * dy_deta_1D * dx_dzeta_1D - \
 #   dy_dxi_1D * dx_deta_1D * dz_dzeta_1D - \
 #   dx_dxi_1D * dz_deta_1D * dy_dzeta_1D
 # inv_det = 1.0 / det
 # jacobian[0, 0, :] = \
 #   inv_det * (dy_deta_1D[:]  * dz_dzeta_1D[:] - \
 #              dz_deta_1D[:]  * dy_dzeta_1D[:])
 # jacobian[0, 1, :] = \
 #   inv_det * (dx_dzeta_1D[:] * dz_deta_1D[:] - \
 #              dx_deta_1D[:]  * dz_dzeta_1D[:])
 # jacobian[0, 2, :] = \
 #   inv_det * (dx_deta_1D[:]  * dy_dzeta_1D[:] - \
 #              dx_dzeta_1D[:] * dy_deta_1D[:])
 # jacobian[1, 0, :] = \
 #   inv_det * (dy_dzeta_1D[:] * dz_dxi_1D[:] - \
 #              dy_dxi_1D[:]   * dz_dzeta_1D[:])
 # jacobian[1, 1, :] = \
 #   inv_det * (dx_dxi_1D[:]   * dz_dzeta_1D[:] - \
 #              dx_dzeta_1D[:] * dz_dxi_1D[:])
 # jacobian[1, 2, :] = \
 #   inv_det * (dx_dxi_1D[:]   * dy_dzeta_1D[:] - \
 #              dx_dzeta_1D[:] * dy_dxi_1D[:])
 # jacobian[2, 0, :] = \
 #   inv_det * (dy_dxi_1D[:]   * dz_deta_1D[:] - \
 #              dy_deta_1D[:]  * dz_dxi_1D[:])
 # jacobian[2, 1, :] = \
 #   inv_det * (dx_deta_1D[:]  * dz_dxi_1D[:] - \
 #              dx_dxi_1D[:]   * dz_deta_1D[:])
 # jacobian[2, 2, :] = \
 #   inv_det * (dx_dxi_1D[:]   * dy_deta_1D[:] - \
 #              dx_deta_1D[:]  * dy_dxi_1D[:])
 # t_end = time.time()
 # print("DEBUG creating jacobian", t_end - t_begin)

  # print('shape of jacobian', jacobian.shape)
  return(jacobian)
#------------------------------------------------------------------------------
## output of the following is [3,3, num_cell]
def jacobian_of_grid_3d(xi, eta, zeta, num_cell, cell_center, \
      accuracy_x, accuracy_y, accuracy_z):


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
  d_dxi = findiff.FinDiff(0, xi_delta, acc=accuracy_x)

  # d/dxi
  t_begin = time.time()
  for i_zeta in range(num_zeta):
    for i_eta in range(num_eta):
      dx_dxi[:,i_eta,i_zeta] = d_dxi(cell_center[:, i_eta, i_zeta, 0])
      dy_dxi[:,i_eta,i_zeta] = d_dxi(cell_center[:, i_eta, i_zeta, 1])
      dz_dxi[:,i_eta,i_zeta] = d_dxi(cell_center[:, i_eta, i_zeta, 2])
  t_end = time.time()
  print("DEBUG jacobian d/dxi", t_end - t_begin)

  dx_deta = np.zeros([num_xi, num_eta, num_zeta])
  dy_deta = np.zeros([num_xi, num_eta, num_zeta])
  dz_deta = np.zeros([num_xi, num_eta, num_zeta])
  d_deta = findiff.FinDiff(0, eta_delta, acc=accuracy_y)

  # d/deta
  t_begin = time.time()
  for i_zeta in range(num_zeta):
    for i_xi in range(num_xi):
      dx_deta[i_xi, :, i_zeta] = d_deta(cell_center[i_xi, :, i_zeta, 0])
      dy_deta[i_xi, :, i_zeta] = d_deta(cell_center[i_xi, :, i_zeta, 1])
      dz_deta[i_xi, :, i_zeta] = d_deta(cell_center[i_xi, :, i_zeta, 2])
  t_end = time.time()
  print("DEBUG jacobian d/deta", t_end - t_begin)

  dx_dzeta = np.zeros([num_xi, num_eta, num_zeta])
  dy_dzeta = np.zeros([num_xi, num_eta, num_zeta])
  dz_dzeta = np.zeros([num_xi, num_eta, num_zeta])
  d_dzeta = findiff.FinDiff(0, zeta_delta, acc=accuracy_z)

  # d/dzeta
  t_brgin = time.time()
  for i_eta in range(num_eta):
    for i_xi in range(num_xi):
      dx_dzeta[i_xi, i_eta, :] = d_dzeta(cell_center[i_xi, i_eta, :, 0])
      dy_dzeta[i_xi, i_eta, :] = d_dzeta(cell_center[i_xi, i_eta, :, 1])
      dz_dzeta[i_xi, i_eta, :] = d_dzeta(cell_center[i_xi, i_eta, :, 2])
  t_end = time.time() 
  print("DEBUG jacobian d/dzeta", t_end - t_begin)

  t_begin = time.time()
  dx_dxi_1D   = array_3D_to_1D(xi, eta, zeta, num_cell, dx_dxi)
  dy_dxi_1D   = array_3D_to_1D(xi, eta, zeta, num_cell, dy_dxi)
  dz_dxi_1D   = array_3D_to_1D(xi, eta, zeta, num_cell, dz_dxi)
  dx_deta_1D  = array_3D_to_1D(xi, eta, zeta, num_cell, dx_deta)
  dy_deta_1D  = array_3D_to_1D(xi, eta, zeta, num_cell, dy_deta)
  dz_deta_1D  = array_3D_to_1D(xi, eta, zeta, num_cell, dz_deta)
  dx_dzeta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dx_dzeta)
  dy_dzeta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dy_dzeta)
  dz_dzeta_1D = array_3D_to_1D(xi, eta, zeta, num_cell, dz_dzeta)
  t_end = time.time()
  print("DEBUG jacobian array_3D_to_1D", t_end -t_begin)

  jacobian = np.zeros([3, 3, num_cell])

  # j00 = dxi/dx, j01 = dxi/dy j10=, j02 = dxi/dz
  # j10 = deta / dx, j11 = deta / dy, j12 = deta/dz,
  # j20 = dzeta / dx, j21 = dzeta / dy, j22 = dzeta/dz,
  #ivanComment Calculating det at once to be able to use slicing of numpy
  #ivanComment arrays rather than for-loops.
  t_begin = time.time()
  det = \
    dx_dxi_1D * dy_deta_1D * dz_dzeta_1D + \
    dy_dxi_1D * dz_deta_1D * dx_dzeta_1D + \
    dz_dxi_1D * dx_deta_1D * dy_dzeta_1D - \
    dz_dxi_1D * dy_deta_1D * dx_dzeta_1D - \
    dy_dxi_1D * dx_deta_1D * dz_dzeta_1D - \
    dx_dxi_1D * dz_deta_1D * dy_dzeta_1D
  inv_det = 1.0 / det
  jacobian[0, 0, :] = \
    inv_det * (dy_deta_1D[:]  * dz_dzeta_1D[:] - \
               dz_deta_1D[:]  * dy_dzeta_1D[:])
  jacobian[0, 1, :] = \
    inv_det * (dx_dzeta_1D[:] * dz_deta_1D[:] - \
               dx_deta_1D[:]  * dz_dzeta_1D[:])
  jacobian[0, 2, :] = \
    inv_det * (dx_deta_1D[:]  * dy_dzeta_1D[:] - \
               dx_dzeta_1D[:] * dy_deta_1D[:])
  jacobian[1, 0, :] = \
    inv_det * (dy_dzeta_1D[:] * dz_dxi_1D[:] - \
               dy_dxi_1D[:]   * dz_dzeta_1D[:])
  jacobian[1, 1, :] = \
    inv_det * (dx_dxi_1D[:]   * dz_dzeta_1D[:] - \
               dx_dzeta_1D[:] * dz_dxi_1D[:])
  jacobian[1, 2, :] = \
    inv_det * (dx_dxi_1D[:]   * dy_dzeta_1D[:] - \
               dx_dzeta_1D[:] * dy_dxi_1D[:])
  jacobian[2, 0, :] = \
    inv_det * (dy_dxi_1D[:]   * dz_deta_1D[:] - \
               dy_deta_1D[:]  * dz_dxi_1D[:])
  jacobian[2, 1, :] = \
    inv_det * (dx_deta_1D[:]  * dz_dxi_1D[:] - \
               dx_dxi_1D[:]   * dz_deta_1D[:])
  jacobian[2, 2, :] = \
    inv_det * (dx_dxi_1D[:]   * dy_deta_1D[:] - \
               dx_deta_1D[:]  * dy_dxi_1D[:])
  t_end = time.time()
  print("DEBUG creating jacobian", t_end - t_begin)

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
    dvar_dx[:, i_dim] = \
      dvar_dxi_1D[:, i_dim]  * jacobian[0, 0, :] + \
      dvar_deta_1D[:, i_dim] * jacobian[1, 0, :]
    dvar_dy[:, i_dim] = \
      dvar_dxi_1D[:, i_dim]  * jacobian[0, 1, :] + \
      dvar_deta_1D[:, i_dim] * jacobian[1, 1, :]

  return(dvar_dx, dvar_dy)

# -----------------------------------------------------------------------------
### output of restructured derivative function is [num_xi,num_eta, num_zeta, num_dof]
def derivative_3d_restructured(var, xi, eta, zeta, num_cell, jacobian, accuracy_x, \
      accuracy_y, accuracy_z):

  t_begin = time.time()

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
  t_end = time.time()
  print("DEBUG reshaping arrays for derivative calculation", t_end - t_begin)
  
  d_dxi = findiff.FinDiff(0, xi_delta, acc=accuracy_x)
  d_deta = findiff.FinDiff(0, eta_delta, acc=accuracy_y)
  d_dzeta = findiff.FinDiff(0, zeta_delta, acc=accuracy_z)

  dvar_dxi = np.zeros([num_xi, num_eta*num_zeta*num_dof])
  for i_xi_dof in range(num_eta*num_zeta*num_dof):
    dvar_dxi[:,i_xi_dof] =  d_dxi(var_xi[:,i_xi_dof])
  t_dxi = time.time()
  print("DEBUG derivative_3D dvar_dxi", t_dxi - t_end)

  dvar_deta = np.zeros([num_eta, num_xi * num_zeta * num_dof])
  for i_eta_dof in range(num_xi * num_zeta * num_dof):
    dvar_deta[:,i_eta_dof] = d_deta(var_eta[:, i_eta_dof])
  t_deta = time.time()
  print("DEBUG derivative_3D dvar_deta", t_deta - t_dxi)

  dvar_dzeta = np.zeros([num_zeta, num_xi * num_eta * num_dof])
  for i_zeta_dof in range(num_xi * num_eta * num_dof):
    dvar_dzeta[:,i_zeta_dof] = d_dzeta(var_zeta[:, i_zeta_dof])
  t_dzeta = time.time()
  print("DEBUG derivative_3D dvar_dzeta", t_dzeta - t_deta)

  # reshape derivative to 3d
  # var_xi = [i_xi, (num_dof*num_zeta)*i_eta + (num_dof)*i_zeta + i_dof)]
  dvar_dxi = np.reshape(dvar_dxi, (num_xi, num_eta, num_zeta, num_dof))

  # var_eta = [i_eta, (num_dof*num_zeta)*i_xi + (num_dof)*i_zeta + i_dof)]
  dvar_deta = np.reshape(dvar_deta, (num_eta, num_xi, num_zeta, num_dof))
  dvar_deta = dvar_deta.transpose(1, 0, 2, 3)

  # var_zeta = [i_zeta, (num_dof*num_xi)*i_eta + (num_dof)*i_xi + i_dof)]
  dvar_dzeta = np.reshape(dvar_dzeta, (num_zeta, num_eta, num_xi, num_dof))
  dvar_dzeta = dvar_dzeta.transpose(2, 1, 0, 3)

#  dvar_dxi_1D = np.zeros([num_cell, num_dof])
#  dvar_deta_1D = np.zeros([num_cell, num_dof])
#  dvar_dzeta_1D = np.zeros([num_cell, num_dof])
#
#  #donyaFuture change this from 2 loops of n_dim and n_snap to one loop of n_dof
#  t_begin = time.time()
#  for i_dof in range(num_dof):
#    dvar_dxi_1D[:,i_dof]    = \
#      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_dxi[:, :, :, i_dof])
#    dvar_deta_1D[:,i_dof]   = \
#      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_deta[:, :, :, i_dof])
#    dvar_dzeta_1D[:, i_dof] = \
#      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_dzeta[:, :, :, i_dof])
#  t_end = time.time()
#  print("DEBUG array_3D_to_1D", t_end - t_begin)

  #donyaFuture change this from 2 loops of n_dim and n_snap to one loop of n_dof
  # dvar/dx = dvar/dxi*dxi/dx + dvar/deta*deta/dx
  # j00 = dxi/dx, j01 = dxi/dy , j02 = dxi/dz
  # j10 = deta / dx, j11 = deta / dy, j12 = deta/dz,
  # j20 = dzeta / dx, j21 = dzeta / dy, j22 = dzeta/dz,

  dvar_dx = np.zeros([num_xi, num_eta, num_zeta, num_dof])
  dvar_dy = np.zeros([num_xi, num_eta, num_zeta, num_dof])
  dvar_dz = np.zeros([num_xi, num_eta, num_zeta, num_dof])
  t_begin = time.time()
  for i_dof in range(num_dof):
    dvar_dx[:,:,:,i_dof] = \
      dvar_dxi[:,:,:, i_dof]   * jacobian[0, 0, :,:,:] + \
      dvar_deta[:,:,:, i_dof]  * jacobian[1, 0, :,:,:] + \
      dvar_dzeta[:,:,:, i_dof] * jacobian[2, 0, :,:,:]
    dvar_dy[:, :,:,i_dof] = \
      dvar_dxi[:,:,:, i_dof]   * jacobian[0, 1, :,:,:] + \
      dvar_deta[:,:,:, i_dof]  * jacobian[1, 1, :,:,:] + \
      dvar_dzeta[:,:,:, i_dof] * jacobian[2, 1, :,:,:]
    dvar_dz[:,:,:, i_dof] = \
      dvar_dxi[:,:,:, i_dof]   * jacobian[0, 2, :,:,:] + \
      dvar_deta[:,:,:, i_dof]  * jacobian[1, 2, :,:,:] + \
      dvar_dzeta[:,:,:, i_dof] * jacobian[2, 2, :,:,:]

  t_end = time.time()
  print("DEBUG jacobian into dvar-dx,y,z", t_end - t_begin)



 # dvar_dx = np.zeros([num_cell, num_dof])
 # dvar_dy = np.zeros([num_cell, num_dof])
 # dvar_dz = np.zeros([num_cell, num_dof])
 # t_begin = time.time()
 # for i_dof in range(num_dof):
 #   dvar_dx[:, i_dof] = \
 #     dvar_dxi_1D[:, i_dof]   * jacobian[0, 0, :] + \
 #     dvar_deta_1D[:, i_dof]  * jacobian[1, 0, :] + \
 #     dvar_dzeta_1D[:, i_dof] * jacobian[2, 0, :]
 #   dvar_dy[:, i_dof] = \
 #     dvar_dxi_1D[:, i_dof]   * jacobian[0, 1, :] + \
 #     dvar_deta_1D[:, i_dof]  * jacobian[1, 1, :] + \
 #     dvar_dzeta_1D[:, i_dof] * jacobian[2, 1, :]
 #   dvar_dz[:, i_dof] = \
 #     dvar_dxi_1D[:, i_dof]   * jacobian[0, 2, :] + \
 #     dvar_deta_1D[:, i_dof]  * jacobian[1, 2, :] + \
 #     dvar_dzeta_1D[:, i_dof] * jacobian[2, 2, :]

 # t_end = time.time()
 # print("DEBUG jacobian into dvar-dx,y,z", t_end - t_begin)

  return(dvar_dx, dvar_dy, dvar_dz)
# ------------------------------------------------------------------------------
### output of the following derivative function is [num_cell, num_dof]
def derivative_3d(var, xi, eta, zeta, num_cell, jacobian, accuracy_x, \
      accuracy_y, accuracy_z):

  t_begin = time.time()

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
  t_end = time.time()
  print("DEBUG reshaping arrays for derivative calculation", t_end - t_begin)
  
  d_dxi = findiff.FinDiff(0, xi_delta, acc=accuracy_x)
  d_deta = findiff.FinDiff(0, eta_delta, acc=accuracy_y)
  d_dzeta = findiff.FinDiff(0, zeta_delta, acc=accuracy_z)

  dvar_dxi = np.zeros([num_xi, num_eta*num_zeta*num_dof])
  for i_xi_dof in range(num_eta*num_zeta*num_dof):
    dvar_dxi[:,i_xi_dof] =  d_dxi(var_xi[:,i_xi_dof])
  t_dxi = time.time()
  print("DEBUG derivative_3D dvar_dxi", t_dxi - t_end)

  dvar_deta = np.zeros([num_eta, num_xi * num_zeta * num_dof])
  for i_eta_dof in range(num_xi * num_zeta * num_dof):
    dvar_deta[:,i_eta_dof] = d_deta(var_eta[:, i_eta_dof])
  t_deta = time.time()
  print("DEBUG derivative_3D dvar_deta", t_deta - t_dxi)

  dvar_dzeta = np.zeros([num_zeta, num_xi * num_eta * num_dof])
  for i_zeta_dof in range(num_xi * num_eta * num_dof):
    dvar_dzeta[:,i_zeta_dof] = d_dzeta(var_zeta[:, i_zeta_dof])
  t_dzeta = time.time()
  print("DEBUG derivative_3D dvar_dzeta", t_dzeta - t_deta)

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
  t_begin = time.time()
  for i_dof in range(num_dof):
    dvar_dxi_1D[:,i_dof]    = \
      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_dxi[:, :, :, i_dof])
    dvar_deta_1D[:,i_dof]   = \
      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_deta[:, :, :, i_dof])
    dvar_dzeta_1D[:, i_dof] = \
      array_3D_to_1D(xi, eta, zeta, num_cell, dvar_dzeta[:, :, :, i_dof])
  t_end = time.time()
  print("DEBUG array_3D_to_1D", t_end - t_begin)

  #donyaFuture change this from 2 loops of n_dim and n_snap to one loop of n_dof
  # dvar/dx = dvar/dxi*dxi/dx + dvar/deta*deta/dx
  # j00 = dxi/dx, j01 = dxi/dy , j02 = dxi/dz
  # j10 = deta / dx, j11 = deta / dy, j12 = deta/dz,
  # j20 = dzeta / dx, j21 = dzeta / dy, j22 = dzeta/dz,
  dvar_dx = np.zeros([num_cell, num_dof])
  dvar_dy = np.zeros([num_cell, num_dof])
  dvar_dz = np.zeros([num_cell, num_dof])
  t_begin = time.time()
  for i_dof in range(num_dof):
    dvar_dx[:, i_dof] = \
      dvar_dxi_1D[:, i_dof]   * jacobian[0, 0, :] + \
      dvar_deta_1D[:, i_dof]  * jacobian[1, 0, :] + \
      dvar_dzeta_1D[:, i_dof] * jacobian[2, 0, :]
    dvar_dy[:, i_dof] = \
      dvar_dxi_1D[:, i_dof]   * jacobian[0, 1, :] + \
      dvar_deta_1D[:, i_dof]  * jacobian[1, 1, :] + \
      dvar_dzeta_1D[:, i_dof] * jacobian[2, 1, :]
    dvar_dz[:, i_dof] = \
      dvar_dxi_1D[:, i_dof]   * jacobian[0, 2, :] + \
      dvar_deta_1D[:, i_dof]  * jacobian[1, 2, :] + \
      dvar_dzeta_1D[:, i_dof] * jacobian[2, 2, :]

  t_end = time.time()
  print("DEBUG jacobian into dvar-dx,y,z", t_end - t_begin)

  return(dvar_dx, dvar_dy, dvar_dz)
