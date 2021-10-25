# -----------------------------------------------------------------------------
# \file incompressible_navier_stokes_rom.py
# \brief Methods involved in computation of reduced-order model of
#        incompressible Navier-Stokes equations
#
# Reference:
# Lee (2020) "On Improving the Predictable Accuracy of Reduced-order Models for
# Fluid Flows," DOI: 10.13140/RG.2.2.26712.52489 PhD Thesis
# -----------------------------------------------------------------------------
import numpy as np

from aerofusion.data import array_conversion as arr_conv
from aerofusion.numerics import derivatives_curvilinear_grid as curvder
import time

# -----------------------------------------------------------------------------
def pod_rom_matrices_2d(xi_index, eta_index, zeta_index, cell_center,
             num_cell, phi, weights, velocity_0, jacobian, accuracy):

  dim = velocity_0.shape
  num_xi   = dim[0]
  num_eta  = dim[1]
  num_zeta = dim[2]
  num_dim  = dim[3]
  num_snapshots = phi.shape[1]

  velocity_0_1D = np.zeros([num_cell, num_dim])
  for i_dim in range(num_dim):
    velocity_0_1D[:, i_dim] = \
      arr_conv.array_3D_to_1D(\
        xi_index, eta_index, zeta_index, num_cell,
        velocity_0[:, :, :, i_dim])

  u0_ND = velocity_0_1D[:, 0]
  v0_ND = velocity_0_1D[:, 1]
  w0_ND = velocity_0_1D[:, 2]

  for i_dim in range(num_dim - 1):
    u0_ND = np.append(u0_ND, velocity_0_1D[:, 0], axis=0)
    v0_ND = np.append(v0_ND, velocity_0_1D[:, 1], axis=0)

  ##----- for a 2d problem with 3d approach----
  u0_ND[2 * num_cell + 1:3 * num_cell] = 0
  v0_ND[2 * num_cell + 1:3 * num_cell] = 0

  ##-------derivative on curvilinear grid-----------
  print('Calculating velocity derivatives')
  dvel_dx_1D = np.zeros([num_cell, num_dim])
  dvel_dy_1D = np.zeros([num_cell, num_dim])
  ddvel_dx2_1D = np.zeros([num_cell, num_dim])
  ddvel_dy2_1D = np.zeros([num_cell, num_dim])
  dvel_dx_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  dvel_dy_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  (dvel_dx_1D, dvel_dy_1D) = curvder.derivative(\
    velocity_0, xi_index, eta_index, zeta_index, jacobian, accuracy)
  for i_dim in range(num_dim):
    dvel_dx_3D[:,:,:,i_dim] = arr_conv.array_1D_to_3D(\
      xi_index, eta_index, zeta_index, num_cell, dvel_dx_1D[:,i_dim])
    dvel_dy_3D[:, :, :, i_dim] = arr_conv.array_1D_to_3D(\
      xi_index, eta_index, zeta_index, num_cell, dvel_dy_1D[:, i_dim])

  (ddvel_dx2_1D, nul) = curvder.derivative(\
    dvel_dx_3D, xi_index, eta_index, zeta_index, jacobian, accuracy)
  (nul, ddvel_dy2_1D) = curvder.derivative(\
    dvel_dy_3D, xi_index, eta_index, zeta_index, jacobian, accuracy)

  dvel_dx_1D   = np.reshape(dvel_dx_1D.transpose(),   (num_cell * num_dim))
  dvel_dy_1D   = np.reshape(dvel_dy_1D.transpose(),   (num_cell * num_dim))
  ddvel_dx2_1D = np.reshape(ddvel_dx2_1D.transpose(), (num_cell * num_dim))
  ddvel_dy2_1D = np.reshape(ddvel_dy2_1D.transpose(), (num_cell * num_dim))

  phi_dim1_ND = np.zeros([num_cell, num_snapshots])
  phi_dim2_ND = np.zeros([num_cell, num_snapshots])
  phi_dim1_ND[:,:] = phi[0:num_cell, :]
  phi_dim2_ND[:,:] = phi[num_cell:2*num_cell, :]

  for i_dim in range(num_dim-1):
    phi_dim1_ND = np.append(phi_dim1_ND, phi[0:num_cell, :], axis=0)
    phi_dim2_ND = np.append(phi_dim2_ND, phi[num_cell:2*num_cell, :], axis=0)

  phi_1D = np.zeros([num_cell, num_dim, num_snapshots])
  for i_mode in range(num_snapshots):
      phi_1D[:, :, i_mode] = \
        np.reshape(phi[:, i_mode], (num_dim, num_cell)).transpose()

  phi_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim, num_snapshots])
  for i_dim in range(num_dim):
      for i_mode in range(num_snapshots):
          phi_3D[:, :, :, i_dim, i_mode] = \
            arr_conv.array_1D_to_3D(\
              xi_index, eta_index, zeta_index, num_cell,\
                phi_1D[:, i_dim, i_mode])

  # using FD for derivative
  dphidx_1D   = np.zeros([num_cell, num_dim, num_snapshots])
  dphidy_1D   = np.zeros([num_cell, num_dim, num_snapshots])
  dphidx_3D   = np.zeros([num_xi, num_eta, num_zeta, num_dim, num_snapshots])
  dphidy_3D   = np.zeros([num_xi, num_eta, num_zeta, num_dim, num_snapshots])
  ddphidx2_1D = np.zeros([num_cell, num_dim, num_snapshots])
  ddphidy2_1D = np.zeros([num_cell, num_dim, num_snapshots])

  print('Calculating derivative of phi')
  for i_snap in range(num_snapshots):
    (dphidx_1D[:,:,i_snap], dphidy_1D[:,:,i_snap]) = \
       curvder.derivative(\
         phi_3D[:,:,:,:,i_snap],
         xi_index,
         eta_index,
         zeta_index,
         jacobian,
         accuracy)
    for i_dim in range(num_dim):
      dphidx_3D[:,:,:,i_dim, i_snap] = \
        arr_conv.array_1D_to_3D(\
          xi_index,
          eta_index,
          zeta_index,
          num_cell,
          dphidx_1D[:,i_dim, i_snap])
      dphidy_3D[:, :, :, i_dim, i_snap] = \
        arr_conv.array_1D_to_3D(\
          xi_index,
          eta_index,
          zeta_index,
          num_cell,
          dphidy_1D[:, i_dim, i_snap])
    (ddphidx2_1D[:,:,i_snap], nul) = \
      curvder.derivative(\
        dphidx_3D[:,:,:,:, i_snap],
        xi_index,
        eta_index,
        zeta_index,
        jacobian,
        accuracy)
    (nul, ddphidy2_1D[:, :, i_snap]) = \
      curvder.derivative(\
        dphidy_3D[:, :, :, :, i_snap],
        xi_index,
        eta_index,
        zeta_index,
        jacobian,
        accuracy)

  dphi_dx_1D = np.zeros([num_cell*num_dim, num_snapshots])
  dphi_dy_1D = np.zeros([num_cell * num_dim, num_snapshots])
  ddphi_dx2_1D = np.zeros([num_cell * num_dim, num_snapshots])
  ddphi_dy2_1D = np.zeros([num_cell * num_dim, num_snapshots])

  for i_mode in range(num_snapshots):
    dphi_dx_1D[:,i_mode] = \
      np.reshape((dphidx_1D[:,:,i_mode]).transpose(), (num_cell * num_dim))
    dphi_dy_1D[:,i_mode] = \
      np.reshape((dphidy_1D[:,:,i_mode]).transpose(), (num_cell * num_dim))
    ddphi_dx2_1D[:,i_mode] = \
      np.reshape((ddphidx2_1D[:,:,i_mode]).transpose(), (num_cell * num_dim))
    ddphi_dy2_1D[:,i_mode] = \
      np.reshape((ddphidy2_1D[:,:,i_mode]).transpose(), (num_cell * num_dim))

  #print('Calculating numerical C0 ')
  phi_transpose_by_weight = np.multiply(phi.transpose(), weights)
  C0 = - np.matmul(phi_transpose_by_weight,
                   np.multiply(u0_ND, dvel_dx_1D ) + \
                   np.multiply(v0_ND, dvel_dy_1D) )

  print('Shape of C0', C0.shape)
  # print('Numerical calc of L0')
  L0 = - np.matmul(\
    phi_transpose_by_weight,
    (np.multiply(u0_ND, dphi_dx_1D.transpose())).transpose() + \
    (np.multiply(v0_ND, dphi_dy_1D.transpose())).transpose() + \
    (np.multiply(phi_dim1_ND.transpose(), dvel_dx_1D)).transpose() + \
    (np.multiply(phi_dim2_ND.transpose(), dvel_dy_1D)).transpose())

  print('Shape of L0', L0.shape)
  # creating LRe[num_snapshots,num_snapshots]
  LRe = np.matmul(phi_transpose_by_weight,
                    ddphi_dx2_1D + ddphi_dy2_1D)
  print('Shape of LRe', LRe.shape)

  # creating CRe
  CRe = np.matmul(phi_transpose_by_weight, ddvel_dx2_1D) + \
        np.matmul(phi_transpose_by_weight, ddvel_dy2_1D)
  print('Shape of CRe', CRe.shape)

  # creating Q[num_snapshots,num_snapshots,num_snapshots]
  Q = np.zeros([num_snapshots, num_snapshots, num_snapshots])
  for i_mode in range(num_snapshots):
    Q[:, i_mode, :] = \
      - np.matmul(phi_transpose_by_weight,
                  (np.multiply(phi_dim1_ND[:, i_mode],\
                    dphi_dx_1D.transpose())).transpose() + \
                     (np.multiply(phi_dim2_ND[:, i_mode], \
                       dphi_dy_1D.transpose())).transpose())
  print('Shape of Q', Q.shape)

  return (L0, LRe, C0, CRe, Q)


# -----------------------------------------------------------------------------
def pod_rom_matrices_3d(xi_index, eta_index, zeta_index, cell_center,
             num_cell, phi, weights, velocity_0_3d, jacobian,
               accuracy_x, accuracy_y, accuracy_z):

  dim = velocity_0_3d.shape
  num_xi   = dim[0]
  num_eta  = dim[1]
  num_zeta = dim[2]
  num_dim  = dim[3]
  num_snapshots = phi.shape[1]
  num_dof = num_dim *num_snapshots

  velocity_0_1D = np.zeros([num_cell, num_dim])
  t_begin = time.time()
  for i_dim in range(num_dim):
    velocity_0_1D[:, i_dim] = \
      arr_conv.array_3D_to_1D(\
        xi_index, eta_index, zeta_index, num_cell,
        velocity_0_3d[:, :, :, i_dim])
  t_end =time.time()
  print('DEBUG in matrices calc: v0 rehsape', t_end - t_begin)

  u0_ND = velocity_0_1D[:, 0]
  v0_ND = velocity_0_1D[:, 1]
  w0_ND = velocity_0_1D[:, 2]

  for i_dim in range(num_dim - 1):
    u0_ND = np.append(u0_ND, velocity_0_1D[:, 0], axis=0)
    v0_ND = np.append(v0_ND, velocity_0_1D[:, 1], axis=0)
    w0_ND = np.append(w0_ND, velocity_0_1D[:, 2], axis=0)


  ##-------derivative on curvilinear grid-----------
  print('Calculating derivative of velocity')
  dvel_dx_1D = np.zeros([num_cell, num_dim])
  dvel_dy_1D = np.zeros([num_cell, num_dim])
  dvel_dz_1D = np.zeros([num_cell, num_dim])
  ddvel_dx2_1D = np.zeros([num_cell, num_dim])
  ddvel_dy2_1D = np.zeros([num_cell, num_dim])
  ddvel_dz2_1D = np.zeros([num_cell, num_dim])

  (dvel_dx_1D, dvel_dy_1D, dvel_dz_1D) = \
    curvder.derivative_3d(\
      velocity_0_3d,
      xi_index,
      eta_index,
      zeta_index,
      num_cell,
      jacobian,
      accuracy_x,
      accuracy_y,
      accuracy_z)

  dvel_dx_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  dvel_dy_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  dvel_dz_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  
  t_begin = time.time()
  for i_dim in range(num_dim):
    dvel_dx_3D[:,:,:,i_dim] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index,\
        num_xi, num_eta, num_zeta, dvel_dx_1D[:,i_dim])
    dvel_dy_3D[:, :, :, i_dim] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, \
        num_xi, num_eta, num_zeta, dvel_dy_1D[:, i_dim])
    dvel_dz_3D[:, :, :, i_dim] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, \
        num_xi, num_eta, num_zeta, dvel_dz_1D[:, i_dim])
  t_end = time.time()
  print('DEBUG in matrices cal: reconstructing dvel', t_end - t_begin)
 
  (ddvel_dx2_1D, nul, nul) = curvder.derivative_3d \
    (dvel_dx_3D, xi_index, eta_index, zeta_index, num_cell, jacobian,\
      accuracy_x, accuracy_y, accuracy_z)
  (nul, ddvel_dy2_1D, nul) = curvder.derivative_3d \
    (dvel_dy_3D, xi_index, eta_index, zeta_index, num_cell, jacobian,\
      accuracy_x, accuracy_y, accuracy_z)
  (nul, nul, ddvel_dz2_1D) = curvder.derivative_3d \
    (dvel_dz_3D, xi_index, eta_index, zeta_index, num_cell, jacobian,\
      accuracy_x, accuracy_y, accuracy_z)

  print('Reconstructing derivative of velocity')
  dvel_dx_1D = np.reshape(dvel_dx_1D.transpose(), (num_cell * num_dim))
  dvel_dy_1D = np.reshape(dvel_dy_1D.transpose(), (num_cell * num_dim))
  dvel_dz_1D = np.reshape(dvel_dz_1D.transpose(), (num_cell * num_dim))
  ddvel_dx2_1D = np.reshape(ddvel_dx2_1D.transpose(), (num_cell * num_dim))
  ddvel_dy2_1D = np.reshape(ddvel_dy2_1D.transpose(), (num_cell * num_dim))
  ddvel_dz2_1D = np.reshape(ddvel_dz2_1D.transpose(), (num_cell * num_dim))

  print('Preparing phi')
  phi_x_ND = np.zeros([num_cell, num_snapshots])
  phi_y_ND = np.zeros([num_cell, num_snapshots])
  phi_z_ND = np.zeros([num_cell, num_snapshots])
  phi_x_ND[:,:] = phi[0: num_cell, :]
  phi_y_ND[:,:] = phi[num_cell: 2*num_cell, :]
  phi_z_ND[:, :] = phi[2*num_cell: 3*num_cell, :]

  t_begin = time.time()
  for i_dim in range(num_dim-1):
    phi_x_ND = np.append(phi_x_ND, phi[0: num_cell, :], axis=0)
    phi_y_ND = np.append(phi_y_ND, phi[num_cell: 2*num_cell, :], axis=0)
    phi_z_ND = np.append(phi_z_ND, phi[2*num_cell: 3* num_cell, :], axis=0)
  t_end = time.time()
  print('DEBUG in matrices calc: appending phi', t_end - t_begin)
  # Reshape phi_1D
  t_begin = time.time()
  temp = np.zeros([num_cell, num_dim, num_snapshots])
  for i_dim in range(num_dim):
    temp[:,i_dim, :] = phi[i_dim*num_cell: (i_dim+1)*num_cell,:]

  phi_1D = np.zeros([num_cell, num_dof])
  # Structure of phi_1D will be phi_1D = [i_cell, (num_snap)*i_dim + i_snap]
  phi_1D = np.reshape(temp, (num_cell, num_dof))

  # Structure of phi_3D will be 
  #   phi_3D = [i_xi, i_eta, i_zeta, (num_snap)*i_dim + i_snap]
  phi_3D = np.zeros([num_xi, num_eta, num_zeta, num_dof])
  for i_dof in range(num_dof):
    phi_3D[:, :, :, i_dof] = arr_conv.array_1D_to_3D(\
      xi_index, eta_index, zeta_index, num_xi, num_eta, num_zeta, \
       phi_1D[:, i_dof])
  t_end = time.time()
  print('DEBUG in matrices calc: reshaping and reconstructing phi', \
         t_end-t_begin)
  # Using fd for derivative
  dphidx_1D   = np.zeros([num_cell, num_dof])
  dphidy_1D   = np.zeros([num_cell, num_dof])
  dphidz_1D   = np.zeros([num_cell, num_dof])
  dphidx_3D   = np.zeros([num_xi, num_eta, num_zeta, num_dof])
  dphidy_3D   = np.zeros([num_xi, num_eta, num_zeta, num_dof])
  dphidz_3D   = np.zeros([num_xi, num_eta, num_zeta, num_dof])
  ddphidx2_1D = np.zeros([num_cell, num_dof])
  ddphidy2_1D = np.zeros([num_cell, num_dof])
  ddphidz2_1D = np.zeros([num_cell, num_dof])

  print('Calculating derivative of phi--- in develop branch')
#  ####---------DEBUG-------------------------
#  np.savez('phi_3D_develop.npz', phi_3D = phi_3D)
#  #####----------------------------------------------
  (dphidx_1D, dphidy_1D, dphidz_1D) = curvder.derivative_3d \
    (phi_3D, xi_index, eta_index, zeta_index, num_cell, jacobian,\
       accuracy_x, accuracy_y, accuracy_z)
#  ####---------DEBUG-------------------------
#  np.savez('dphi_1D_develop.npz', dphidx_1D = dphidx_1D, dphidy_1D = dphidy_1D,\
#                          dphidz_1D = dphidz_1D)
#  #####----------------------------------------------
  for i_dof in range(num_dof):
    dphidx_3D[:,:,:,i_dof] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_xi, num_eta, num_zeta,\
        dphidx_1D[:,i_dof])
    dphidy_3D[:, :, :, i_dof] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_xi, num_eta, num_zeta, \
        dphidy_1D[:, i_dof])
    dphidz_3D[:, :, :, i_dof] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_xi, num_eta, num_zeta, \
        dphidz_1D[:, i_dof])

  ####---------DEBUG-------------------------
 # np.savez('dphi_3D_develop.npz', dphidx_3D = dphidx_3D, dphidy_3D = dphidy_3D,\
 #                         dphidz_3D = dphidz_3D)
 # print('matrices saved in develop branch')
 # exit(1)
 # ####--------------------------------------------------
  (ddphidx2_1D, nul, nul) = curvder.derivative_3d \
    (dphidx_3D, xi_index, eta_index, zeta_index, num_cell, jacobian,\
       accuracy_x, accuracy_y, accuracy_z)
  (nul, ddphidy2_1D, nul) = curvder.derivative_3d \
    (dphidy_3D, xi_index, eta_index, zeta_index, num_cell, jacobian,\
      accuracy_x, accuracy_y, accuracy_z)
  (nul, nul, ddphidz2_1D) = curvder.derivative_3d \
    (dphidz_3D, xi_index, eta_index, zeta_index, num_cell, jacobian,\
       accuracy_x, accuracy_y, accuracy_z)

  ####---------DEBUG-------------------------
  np.savez('dphi_1D_final_develop.npz', dphidx_1D = dphidx_1D, dphidy_1D = dphidy_1D,\
                          dphidz_1D = dphidz_1D)
  ####--------------------------------------------------
  print('Reconstructing derivative of phi')
  t_begin = time.time()
  dphi_dx_1D = np.reshape(dphidx_1D, (num_cell, num_dim, num_snapshots))
  dphi_dy_1D = np.reshape(dphidy_1D, (num_cell, num_dim, num_snapshots))
  dphi_dz_1D = np.reshape(dphidz_1D, (num_cell, num_dim, num_snapshots))
  ddphi_dx2_1D = np.reshape(ddphidx2_1D, (num_cell, num_dim, num_snapshots))
  ddphi_dy2_1D = np.reshape(ddphidy2_1D, (num_cell, num_dim, num_snapshots))
  ddphi_dz2_1D = np.reshape(ddphidz2_1D, (num_cell, num_dim, num_snapshots))

  ####---------DEBUG-------------------------
  np.savez('dphi_1D_reshape_develop.npz', dphidx_1D = dphi_dx_1D, dphidy_1D =
            dphi_dy_1D, dphidz_1D = dphi_dz_1D)
  ####--------------------------------------------------
  # reconstructing to have the shape of
  # dphi_d_1D = [(num_cell*i_dim)+i_cell, i_snap]
  dphi_dx_1D = np.reshape(\
    dphi_dx_1D.transpose(1,0,2), (num_cell*num_dim, num_snapshots))
  dphi_dy_1D = np.reshape(\
    dphi_dy_1D.transpose(1, 0, 2), (num_cell * num_dim, num_snapshots))
  dphi_dz_1D = np.reshape(\
    dphi_dz_1D.transpose(1, 0, 2), (num_cell * num_dim, num_snapshots))
  ddphi_dx2_1D = np.reshape(\
    ddphi_dx2_1D.transpose(1, 0, 2), (num_cell * num_dim, num_snapshots))
  ddphi_dy2_1D = np.reshape(\
    ddphi_dy2_1D.transpose(1, 0, 2), (num_cell * num_dim, num_snapshots))
  ddphi_dz2_1D = np.reshape(\
    ddphi_dz2_1D.transpose(1, 0, 2), (num_cell * num_dim, num_snapshots))
  ####---------DEBUG-------------------------
  np.savez('dphi_1D_reshape_develop_2.npz', dphidx_1D = dphi_dx_1D, dphidy_1D =
            dphi_dy_1D, dphidz_1D = dphi_dz_1D)

  print('debug matrices are saved')
  exit(1)
  ####--------------------------------------------------
  t_end = time.time()
  print('DEBUG in matrices calc: reshaping dphi', t_end - t_begin)
  phi_transpose_by_weight = np.multiply(phi.transpose(), weights)
  ####DEBUG
  np.savez('matrices_param_orig.npz', dveldx_1D = dvel_dx_1D,\
            dveldy_1D = dvel_dy_1D, dveldz_1D = dvel_dz_1D,\
            ddvelddx_1D = ddvel_dx2_1D, 
            ddvelddy_1D = ddvel_dy2_1D, 
            ddvelddz_1D = ddvel_dz2_1D,
            dphidx_1D = dphi_dx_1D,
            dphidy_1D = dphi_dy_1D,
            dphidz_1D = dphi_dz_1D,
            ddphiddx_1D = ddphi_dx2_1D,
            ddphiddy_1D = ddphi_dy2_1D,
            ddphiddz_1D = ddphi_dz2_1D)

  print('matrices parameters are saved')
  exit(1)
  print('Calculating C0')
  t_begin = time.time()
  C0 = - np.matmul(phi_transpose_by_weight,
                   np.multiply(u0_ND, dvel_dx_1D) + \
                   np.multiply(v0_ND, dvel_dy_1D) + \
                   np.multiply(w0_ND, dvel_dz_1D))
  t_end = time.time()
  print('DEBUG C0 calc', t_end -t_begin)
  print('Shape of C0', C0.shape)
  print('Calculating L0')
  t_begin = time.time()
  L0 = - np.matmul(\
    phi_transpose_by_weight,
    (np.multiply(u0_ND, dphi_dx_1D.transpose())).transpose() + \
    (np.multiply(v0_ND, dphi_dy_1D.transpose())).transpose() + \
    (np.multiply(w0_ND, dphi_dz_1D.transpose())).transpose() + \
    (np.multiply(phi_x_ND.transpose(), dvel_dx_1D)).transpose() + \
    (np.multiply(phi_y_ND.transpose(), dvel_dy_1D)).transpose() + \
    (np.multiply(phi_z_ND.transpose(), dvel_dz_1D)).transpose())
  t_end = time.time()
  print('DEBUG L0 calc', t_end -t_begin)
  print('Shape of L0', L0.shape)
  print('Calculating LRe')
  # creating LRe[num_snapshots,num_snapshots]
  t_begin = time.time()
  LRe = np.matmul(phi_transpose_by_weight,
                    ddphi_dx2_1D + ddphi_dy2_1D + ddphi_dz2_1D)
  t_end = time.time()
  print('DEBUG LRE calc', t_end -t_begin)
  print('Shape of LRe', LRe.shape)
  print('Calculating CRe')
  # creating CRe
  t_begin = time.time()
  CRe = np.matmul(phi_transpose_by_weight, ddvel_dx2_1D) + \
        np.matmul(phi_transpose_by_weight, ddvel_dy2_1D) + \
        np.matmul(phi_transpose_by_weight, ddvel_dz2_1D)
  t_end = time.time()
  print('DEBUG CRe calc', t_end - t_begin)
  print('Shape of CRe', CRe.shape)
  print('Calculating Q')
  # creating Q[num_snapshots,num_snapshots,num_snapshots]
  Q = np.zeros([num_snapshots, num_snapshots, num_snapshots])
  t_begin = time.time()
  for i_mode in range(num_snapshots):
    Q[:, i_mode, :] = \
      - np.matmul(phi_transpose_by_weight,
                  (np.multiply(phi_x_ND[:, i_mode],
                               dphi_dx_1D.transpose())).transpose() + \
                  (np.multiply(phi_y_ND[:, i_mode], \
                               dphi_dy_1D.transpose())).transpose() + \
                  (np.multiply(phi_z_ND[:, i_mode], \
                               dphi_dz_1D.transpose())).transpose())
  t_end = time.time()
  print('DEBUG Q calc', t_end - t_begin)
  print('shape of Q', Q.shape)

  return (L0, LRe, C0, CRe, Q)

#------------2 basis approach
def pod_rom_2basis_matrices_3d(xi_index, eta_index, zeta_index, cell_center,
             num_cell, phi, psi, weights, velocity_0_3d, jacobian, accuracy):

  dim = velocity_0_3d.shape
  num_xi   = dim[0]
  num_eta  = dim[1]
  num_zeta = dim[2]
  num_dim  = dim[3]
  num_mode_phi = phi.shape[1]
  num_mode_psi = psi.shape[1]
  num_dof_phi = num_dim * num_mode_phi
  num_dof_psi = num_dim * num_mode_psi

  velocity_0_1D = np.zeros([num_cell, num_dim])
  for i_dim in range(num_dim):
    velocity_0_1D[:, i_dim] = \
      arr_conv.array_3D_to_1D(\
        xi_index, eta_index, zeta_index, num_cell,
        velocity_0_3d[:, :, :, i_dim])

  u0_ND = velocity_0_1D[:, 0]
  v0_ND = velocity_0_1D[:, 1]
  w0_ND = velocity_0_1D[:, 2]

  for i_dim in range(num_dim - 1):
    u0_ND = np.append(u0_ND, velocity_0_1D[:, 0], axis=0)
    v0_ND = np.append(v0_ND, velocity_0_1D[:, 1], axis=0)
    w0_ND = np.append(w0_ND, velocity_0_1D[:, 2], axis=0)


  ##-------derivative on curvilinear grid-----------
  print('Calculating derivative of velocity-- in develop branch')
  dvel_dx_1D = np.zeros([num_cell, num_dim])
  dvel_dy_1D = np.zeros([num_cell, num_dim])
  dvel_dz_1D = np.zeros([num_cell, num_dim])
  ddvel_dx2_1D = np.zeros([num_cell, num_dim])
  ddvel_dy2_1D = np.zeros([num_cell, num_dim])
  ddvel_dz2_1D = np.zeros([num_cell, num_dim])

  (dvel_dx_1D, dvel_dy_1D, dvel_dz_1D) = \
    curvder.derivative_3d(\
      velocity_0_3d,
      xi_index,
      eta_index,
      zeta_index,
      num_cell,
      jacobian,
      accuracy)

  dvel_dx_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  dvel_dy_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  dvel_dz_3D = np.zeros([num_xi, num_eta, num_zeta, num_dim])

  for i_dim in range(num_dim):
    dvel_dx_3D[:,:,:,i_dim] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_cell, dvel_dx_1D[:,i_dim])
    dvel_dy_3D[:, :, :, i_dim] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_cell, dvel_dy_1D[:, i_dim])
    dvel_dz_3D[:, :, :, i_dim] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_cell, dvel_dz_1D[:, i_dim])

  (ddvel_dx2_1D, nul, nul) = curvder.derivative_3d \
    (dvel_dx_3D, xi_index, eta_index, zeta_index, num_cell, jacobian, accuracy)
  (nul, ddvel_dy2_1D, nul) = curvder.derivative_3d \
    (dvel_dy_3D, xi_index, eta_index, zeta_index, num_cell, jacobian, accuracy)
  (nul, nul, ddvel_dz2_1D) = curvder.derivative_3d \
    (dvel_dz_3D, xi_index, eta_index, zeta_index, num_cell, jacobian, accuracy)

  print('Reconstructing derivative of velocity')
  dvel_dx_1D = np.reshape(dvel_dx_1D.transpose(), (num_cell * num_dim))
  dvel_dy_1D = np.reshape(dvel_dy_1D.transpose(), (num_cell * num_dim))
  dvel_dz_1D = np.reshape(dvel_dz_1D.transpose(), (num_cell * num_dim))
  ddvel_dx2_1D = np.reshape(ddvel_dx2_1D.transpose(), (num_cell * num_dim))
  ddvel_dy2_1D = np.reshape(ddvel_dy2_1D.transpose(), (num_cell * num_dim))
  ddvel_dz2_1D = np.reshape(ddvel_dz2_1D.transpose(), (num_cell * num_dim))

  print('Preparing phi')
  phi_x_ND = np.zeros([num_cell, num_mode_phi])
  phi_y_ND = np.zeros([num_cell, num_mode_phi])
  phi_z_ND = np.zeros([num_cell, num_mode_phi])
  phi_x_ND[:,:] = phi[0: num_cell, :]
  phi_y_ND[:,:] = phi[num_cell: 2*num_cell, :]
  phi_z_ND[:, :] = phi[2*num_cell: 3*num_cell, :]

  for i_dim in range(num_dim-1):
    phi_x_ND = np.append(phi_x_ND, phi[0: num_cell, :], axis=0)
    phi_y_ND = np.append(phi_y_ND, phi[num_cell: 2*num_cell, :], axis=0)
    phi_z_ND = np.append(phi_z_ND, phi[2*num_cell: 3* num_cell, :], axis=0)

  # Reshape phi_1D
  temp = np.zeros([num_cell, num_dim, num_mode_phi])
  for i_dim in range(num_dim):
    temp[:,i_dim, :] = phi[i_dim*num_cell: (i_dim+1)*num_cell,:]

  phi_1D = np.zeros([num_cell, num_dof_phi])
  # Structure of phi_1D will be phi_1D = [i_cell, (num_mode_phi)*i_dim + i_snap]
  phi_1D = np.reshape(temp, (num_cell, num_dof_phi))

  # Structure of phi_3D will be
  #   phi_3D = [i_xi, i_eta, i_zeta, (num_mode_phi)*i_dim + i_snap]
  phi_3D = np.zeros([num_xi, num_eta, num_zeta, num_dof_phi])
  for i_dof in range(num_dof_phi):
    phi_3D[:, :, :, i_dof] = arr_conv.array_1D_to_3D(\
      xi_index, eta_index, zeta_index, num_cell, phi_1D[:, i_dof])

  # Using fd for derivative
  dphidx_1D   = np.zeros([num_cell, num_dof_phi])
  dphidy_1D   = np.zeros([num_cell, num_dof_phi])
  dphidz_1D   = np.zeros([num_cell, num_dof_phi])
  dphidx_3D   = np.zeros([num_xi, num_eta, num_zeta, num_dof_phi])
  dphidy_3D   = np.zeros([num_xi, num_eta, num_zeta, num_dof_phi])
  dphidz_3D   = np.zeros([num_xi, num_eta, num_zeta, num_dof_phi])
  ddphidx2_1D = np.zeros([num_cell, num_dof_phi])
  ddphidy2_1D = np.zeros([num_cell, num_dof_phi])
  ddphidz2_1D = np.zeros([num_cell, num_dof_phi])

  print('Calculating derivative of phi --- develop branch')
  (dphidx_1D, dphidy_1D, dphidz_1D) = curvder.derivative_3d \
    (phi_3D, xi_index, eta_index, zeta_index, num_cell, jacobian, accuracy)
  for i_dof in range(num_dof_phi):
    dphidx_3D[:,:,:,i_dof] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_cell, dphidx_1D[:,i_dof])
    dphidy_3D[:, :, :, i_dof] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_cell, dphidy_1D[:, i_dof])
    dphidz_3D[:, :, :, i_dof] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_cell, dphidz_1D[:, i_dof])

  #####----------------------------------------------
  (ddphidx2_1D, nul, nul) = curvder.derivative_3d \
    (dphidx_3D, xi_index, eta_index, zeta_index, num_cell, jacobian, accuracy)
  (nul, ddphidy2_1D, nul) = curvder.derivative_3d \
    (dphidy_3D, xi_index, eta_index, zeta_index, num_cell, jacobian, accuracy)
  (nul, nul, ddphidz2_1D) = curvder.derivative_3d \
    (dphidz_3D, xi_index, eta_index, zeta_index, num_cell, jacobian, accuracy)

  print('Reconstructing derivative of phi')
  dphi_dx_1D = np.reshape(dphidx_1D, (num_cell, num_dim, num_mode_phi))
  dphi_dy_1D = np.reshape(dphidy_1D, (num_cell, num_dim, num_mode_phi))
  dphi_dz_1D = np.reshape(dphidz_1D, (num_cell, num_dim, num_mode_phi))
  ddphi_dx2_1D = np.reshape(ddphidx2_1D, (num_cell, num_dim, num_mode_phi))
  ddphi_dy2_1D = np.reshape(ddphidy2_1D, (num_cell, num_dim, num_mode_phi))
  ddphi_dz2_1D = np.reshape(ddphidz2_1D, (num_cell, num_dim, num_mode_phi))

  # reconstructing to have the shape of
  # dphi_d_1D = [(num_cell*i_dim)+i_cell, i_snap]
  dphi_dx_1D = np.reshape(\
    dphi_dx_1D.transpose(1,0,2), (num_cell*num_dim, num_mode_phi))
  dphi_dy_1D = np.reshape(\
    dphi_dy_1D.transpose(1, 0, 2), (num_cell * num_dim, num_mode_phi))
  dphi_dz_1D = np.reshape(\
    dphi_dz_1D.transpose(1, 0, 2), (num_cell * num_dim, num_mode_phi))
  ddphi_dx2_1D = np.reshape(\
    ddphi_dx2_1D.transpose(1, 0, 2), (num_cell * num_dim, num_mode_phi))
  ddphi_dy2_1D = np.reshape(\
    ddphi_dy2_1D.transpose(1, 0, 2), (num_cell * num_dim, num_mode_phi))
  ddphi_dz2_1D = np.reshape(\
    ddphi_dz2_1D.transpose(1, 0, 2), (num_cell * num_dim, num_mode_phi))

  #### calculating derivative of psi
  #### Calculating derivativ of psi_x and psi_y--------
  psi_x = np.zeros([num_dof_psi, num_mode_psi])
  psi_y = np.zeros([num_dof_psi, num_mode_psi])
  psi_z = np.zeros([num_dof_psi, num_mode_psi])

  psi_x[:, :] = psi[0 : num_dof_psi, :]
  psi_y[:, :] = psi[num_dof_psi : 2*num_dof_psi, :]
  psi_z[:, :] = psi[num_dof_psi*2 : 3* num_dof_psi, :]

  # Reshape psi_1D
  temp = np.zeros([num_cell, num_dim, num_mode_phi])
  for i_dim in range(num_dim):
    temp[:, i_dim, :] = psi[i_dim * num_cell: (i_dim + 1) * num_cell, :]

  psi_1D = np.zeros([num_cell, num_dof])
  # Structure of psi_1D will be psi_1D = [i_cell, (num_mode_psi)*i_dim + i_snap]
  phi_1D = np.reshape(temp, (num_cell, num_dof_psi))

  # Structure of psi_3D will be
  #   psi_3D = [i_xi, i_eta, i_zeta, (num_mode_psi)*i_dim + i_snap]
  psi_3D = np.zeros([num_xi, num_eta, num_zeta, num_dof_psi])
  for i_dof in range(num_dof_psi):
    psi_3D[:, :, :, i_dof] = arr_conv.array_1D_to_3D( \
      xi_index, eta_index, zeta_index, num_cell, psi_1D[:, i_dof])

  # Using fd for derivative
  dpsidx_1D = np.zeros([num_cell, num_dof_psi])
  dpsidy_1D = np.zeros([num_cell, num_dof_psi])
  dpsidz_1D = np.zeros([num_cell, num_dof_psi])
  dpsidx_3D = np.zeros([num_xi, num_eta, num_zeta, num_dof_psi])
  dpsidy_3D = np.zeros([num_xi, num_eta, num_zeta, num_dof_psi])
  dpsidz_3D = np.zeros([num_xi, num_eta, num_zeta, num_dof_psi])

  print('Calculating derivative of psi')
  (dpsidx_1D, dpsidy_1D, dpsidz_1D) = curvder.derivative_3d \
    (psi_3D, xi_index, eta_index, zeta_index, num_cell, jacobian, accuracy)
  for i_dof in range(num_dof_psi):
    dpsidx_3D[:, :, :, i_dof] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_cell, dpsidx_1D[:, i_dof])
    dpsidy_3D[:, :, :, i_dof] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_cell, dpsidy_1D[:, i_dof])
    dpsidz_3D[:, :, :, i_dof] = arr_conv.array_1D_to_3D \
      (xi_index, eta_index, zeta_index, num_cell, dpsidz_1D[:, i_dof])

  print('Reconstructing derivative of phi')
  dpsi_dx_1D = np.reshape(dpsidx_1D, (num_cell, num_dim, num_mode_psi))
  dpsi_dy_1D = np.reshape(dpsidy_1D, (num_cell, num_dim, num_mode_psi))
  dpsi_dz_1D = np.reshape(dpsidz_1D, (num_cell, num_dim, num_mode_psi))

  # reconstructing to have the shape of
  # dphi_d_1D = [(num_cell*i_dim)+i_cell, i_snap]
  dpsi_dx_1D = np.reshape( \
    dpsi_dx_1D.transpose(1, 0, 2), (num_cell * num_dim, num_mode_phi))
  dpsi_dy_1D = np.reshape( \
    dpsi_dy_1D.transpose(1, 0, 2), (num_cell * num_dim, num_mode_phi))
  dpsi_dz_1D = np.reshape( \
    dpsi_dz_1D.transpose(1, 0, 2), (num_cell * num_dim, num_mode_phi))

  phi_transpose_by_weight = np.multiply(phi.transpose(), weights)

             

  
  print('Calculating C0')
  C0 = - np.matmul(phi_transpose_by_weight,
                   np.multiply(u0_ND, dvel_dx_1D) + \
                   np.multiply(v0_ND, dvel_dy_1D) + \
                   np.multiply(w0_ND, dvel_dz_1D))


  print('shape of C0', C0.shape)
  print('Calculating La')
  La = - np.matmul(\
    phi_transpose_by_weight,
    (np.multiply(phi_x_ND.transpose(), dvel_dx_1D)).transpose() + \
    (np.multiply(phi_y_ND.transpose(), dvel_dy_1D)).transpose() + \
    (np.multiply(phi_z_ND.transpose(), dvel_dz_1D)).transpose())


  print('shape of La', La.shape)
  Lb = - np.matmul(phi_transpose_by_weight,
                   (np.multiply(u0_ND, psi_x.transpose())).transpose() + \
                   (np.multiply(v0_ND, psi_y.transpose())).transpose() + \
                   (np.multiply(w0_ND, psi_z.transpose())).transpose())



  print('shape of Lb', Lb.shape)
  # ------------ creating LRe[num_mode_phi,num_mode_psi]
  LRe = np.matmul(phi_transpose_by_weight,
                  dpsi_dx_1D + dpsi_dy_1D + dpsi_dz_1D)

  print('shape of LRe', LRe.shape)
  # ------------ creating CRe
  print('Calculating CRe')
  CRe = np.matmul(phi_transpose_by_weight, ddvel_dx2_1D) + \
        np.matmul(phi_transpose_by_weight, ddvel_dy2_1D) + \
        np.matmul(phi_transpose_by_weight, ddvel_dz2_1D)

  print('shape of CRe', CRe.shape)
  # ------------ creating Q[num_snapshots,num_snapshots,num_snapshots]
  Q = np.zeros([num_mode_phi, num_mode_phi, num_mode_psi])
  print('Calculating Q')

  for i_mode in range(num_mode_phi):
    Q[:, i_mode, :] = \
      - np.matmul(phi_transpose_by_weight,
                  (np.multiply(phi_x_ND[:, i_mode],
                               psi_x.transpose())).transpose() + \
                  (np.multiply(phi_y_ND[:, i_mode], \
                               psi_y.transpose())).transpose() + \
                  (np.multiply(phi_z_ND[:, i_mode], \
                               psi_z.transpose())).transpose())

  print('shape of Q', Q.shape)
  return (L0, LRe, C0, CRe, Q)

# -----------------------------------------------------------------------------
def RHS_rk45(t, a, Re, char_L, L0, LRe, C0, CRe, Q):
  #rhs = np.matmul((L0 + LRe * (2 / Re)), a) + C0 + CRe * (2 / Re)
  rhs = np.matmul((L0 + LRe * (char_L / Re)), a) + C0 + CRe * (char_L / Re)
  num_modes = (a.shape)[0]
  aT = np.zeros([1,num_modes])
  aT[0,:] = a[:]
  t_begin = time.time()  
  for i_modes in range(num_modes):
    rhs[i_modes] = rhs[i_modes] + np.matmul(np.matmul(aT, Q[i_modes,:,:]), a)
  t_end = time.time()
  print('DEBUG in RHS of rk45', t_end - t_begin)
  return rhs

# -----------------------------------------------------------------------------
def rom_calc_rk45(Re, char_L, L0, LRe, C0, CRe, Q, modal_coef, t_eval):

  from scipy import integrate
  t_span = (t_eval[0], t_eval[len(t_eval)-1])
  sol = \
    integrate.solve_ivp(\
      lambda t,
      y: RHS_rk45(t, y, Re, char_L, L0, LRe, C0, CRe, Q),
      t_span,
      modal_coef[:,0],
      method = 'RK45',
      t_eval = t_eval,
      rtol = 1e-8)
  
  aT = sol.y
  time = sol.t
  #print('Time', time)
  print('Size of aT', aT.shape, modal_coef.shape)
   
#  import matplotlib.pyplot as plt
#  plt.plot(modal_coef[:,0])
#  plt.plot(aT[:,0], '*')
#  plt.savefig("tmp_modal_coeff_0.png")
#  
#  plt.plot(modal_coef[:, 1])
#  plt.plot(aT[:, 1], '*')
#  plt.savefig("tmp_modal_coeff_1.png")
  return (aT)

# -----------------------------------------------------------------------------
def RHS_odeint(a, t, Re, char_L, L0, LRe, C0, CRe, Q):

  rhs = np.matmul((L0 + LRe * (char_L / Re)), a) + C0 + CRe * (char_L / Re)
  num_modes = (a.shape)[0]
  aT = np.zeros([1,num_modes])
  aT[0,:] = a[:]
  for i_modes in range(num_modes):
      rhs[i_modes] = \
       rhs[i_modes] + np.matmul(np.matmul(aT, Q[i_modes,:,:]), a)
  return rhs

# -----------------------------------------------------------------------------
def rom_calc_odeint(Re, char_L, L0, LRe, C0, CRe, Q, modal_coef, t):

 from scipy.integrate import odeint
 a = odeint(RHS_odeint, modal_coef[:,0], t, args=(Re, char_L, L0, LRe, C0, CRe, Q))

 print('shape of odeint aT', (a.transpose()).shape)
 return (a.transpose())

