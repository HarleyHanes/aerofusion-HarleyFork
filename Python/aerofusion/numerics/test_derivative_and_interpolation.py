# -----------------------------------------------------------------------------
# \file test_derivative_and_interpolation.py
# \brief Test derivative and interpolation methods
# -----------------------------------------------------------------------------

import numpy as np
import findiff
import structure_format
import derivative_curvilinear_grid
import Derivative_Calc
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# comparison of derivative0f curvilinear and interpolated
def deriv_curv_interp_comparison_test(\
  var_interp_lin,
  var_interp_nearest,
  var_interp_cubic,
  xi,
  eta,
  zeta,
  cell_center,
  num_x,
  num_y,
  domain_x_min,
  domain_x_max,
  domain_y_min,
  domain_y_max,
  accuracy):

  var_dim = cell_center.shape
  num_xi = var_dim[0]
  num_eta = var_dim[1]
  num_zeta = var_dim[2]
  num_dim = var_dim[3]
  num_cell = num_xi * num_eta

  x = np.linspace(domain_x_min, domain_x_max, num_x)
  y = np.linspace(domain_y_min, domain_y_max, num_y)
  grid_x, grid_y = np.meshgrid(x, y)
  grid_x = grid_x.transpose()
  grid_y = grid_y.transpose()
  # dvel/dx and dvel/dy of interpolated domain
  d_dx = findiff.FinDiff(0, grid_x[:,0], acc=accuracy)
  dver_dx_interp_lin = np.zeros([num_x, num_y])
  dver_dx_interp_lin[:,:] = d_dx(var_interp_lin[:, :])
  dver_dx_interp_nearest = np.zeros([num_x, num_y])
  dver_dx_interp_nearest[:, :] = d_dx(var_interp_nearest[:, :])
  dver_dx_interp_cubic = np.zeros([num_x, num_y])
  dver_dx_interp_cubic[:, :] = d_dx(var_interp_cubic[:, :])

  vmin = -10#np.min(velocity_3D[:, :, 0, dir])
  vmax = 10#np.max(velocity_3D[:, :, 0, dir])
  Derivative_Calc.plot_contour(grid_x, grid_y, dver_dx_interp_cubic[:,:],
    '../plots/dv_dx_interp_cubic.png', 256, vmin, vmax)
  Derivative_Calc.plot_contour(grid_x, grid_y, dver_dx_interp_lin[:, :],
    '../plots/dv_dx_interp_linear.png', 256, vmin, vmax)
  Derivative_Calc.plot_contour(grid_x, grid_y, dver_dx_interp_nearest[:, :],
    '../plots/dv_dx_interp_nearest.png', 256, vmin, vmax)
  Derivative_Calc.plot_contour(grid_x, grid_y, var_interp_lin[:, :],
    '../plots/v_lin.png', 256, vmin, vmax)

  return()

# -----------------------------------------------------------------------------
def deriv_curv_study(var, xi, eta, zeta, cell_center, accuracy):
  var_dim = var.shape
  num_xi = 100#var_dim[0]
  num_eta = 80#var_dim[1]
  num_zeta = 60#var_dim[2]
  num_dim = var_dim[3]
  num_snap = var_dim[4]
  num_cell = num_xi * num_eta*num_zeta

  # testing jacobian when x, y and var are smooth functions
  cell_center = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  xi = np.zeros([num_cell])
  eta = np.zeros([num_cell])
  zeta = np.zeros([num_cell])

  for i_xi in range(num_xi):
    for i_eta in range(num_eta):
      for i_zeta in range(num_zeta):
        ldx = i_xi + num_xi * (i_eta + num_eta * i_zeta)
        xi[ldx] = int(i_xi)
        eta[ldx] = int(i_eta)
        zeta[ldx] = int(i_zeta)
        cell_center[i_xi, :, :, 0]   = \
          i_xi / num_xi + (8*i_eta/num_eta) + (4* i_zeta/num_zeta)
        cell_center[:, i_eta, :, 1]  = \
          5*i_eta / num_eta + i_xi/num_xi + 10* i_zeta/num_zeta
        cell_center[:, :, i_zeta, 2] = \
          i_zeta / num_zeta + 2*i_eta/num_eta + 7* i_xi/num_xi
        # cell_center[i_xi, :, :, 0] = i_xi/num_xi
        # cell_center[:, i_eta, :, 1] = i_eta/num_eta
        # cell_center[:, :, i_zeta, 2] = i_zeta/num_zeta

  xi   = xi.astype('int')
  eta  = eta.astype('int')
  zeta = zeta.astype('int')

  #####--------2d
  # dvar_dx_analy = np.zeros(var.shape)
  # dvar_dy_analy = np.zeros(var.shape)
  # for i_xi in range(num_xi):
  #   for i_eta in range(num_eta):
  #     # var[i_xi, i_eta, 0, 0, 0] = \
  #     #   cell_center[i_xi,i_eta, 0, 0] * cell_center[i_xi,i_eta, 0, 0]
  #     # dvar_dx_analy[i_xi, i_eta, 0, 0, 0] = \
  #     #   2 * cell_center[i_xi,i_eta, 0, 0]
  #     var[i_xi, i_eta, 0, 0, 0] = \
  #       np.sin(np.pi * cell_center[i_xi, i_eta, 0, 0]) + \
  #       np.sin(np.pi * cell_center[i_xi, i_eta, 0, 1])
  #     dvar_dx_analy[i_xi, i_eta, 0, 0, 1] = \
  #       np.pi* np.cos(np.pi *cell_center[i_xi, i_eta, 0, 0])
  #     dvar_dy_analy[i_xi, i_eta, 0, 0, 0] = \
  #       np.pi * np.cos(np.pi * cell_center[i_xi, i_eta, 0, 1])
  ##------------3d
  var_3d = np.zeros([num_xi, num_eta, num_zeta, num_dim, num_snap])
  dvar_dx_analy = np.zeros(var_3d.shape)
  dvar_dy_analy = np.zeros(var_3d.shape)
  dvar_dz_analy = np.zeros(var_3d.shape)
  for i_xi in range(num_xi):
    for i_eta in range(num_eta):
      for i_zeta in range(num_zeta):
        var_3d[i_xi, i_eta, i_zeta, :, :] = \
          np.sin(np.pi * cell_center[i_xi, i_eta, i_zeta, 0]) + \
          np.cos(np.pi * cell_center[i_xi, i_eta, i_zeta, 1]) + \
          np.sin(2*np.pi * cell_center[i_xi, i_eta, i_zeta, 2])
        dvar_dx_analy[i_xi, i_eta, i_zeta, :, :] = \
          np.pi * np.cos(np.pi * cell_center[i_xi, i_eta, i_zeta, 0])
        dvar_dy_analy[i_xi, i_eta, i_zeta, :, :] = \
          - np.pi * np.sin(np.pi * cell_center[i_xi, i_eta, i_zeta, 1])
        dvar_dz_analy[i_xi, i_eta, i_zeta, :, :] = \
          2*np.pi * np.cos(2*np.pi * cell_center[i_xi, i_eta, i_zeta, 2])


  print('cell_center shape', cell_center.shape)
  jacobian = derivative_curvilinear_grid.jacobian_of_grid_3d(\
    xi, eta, zeta, cell_center, accuracy)
  (dvel_dx_1D, dvel_dy_1D, dvel_dz_1D) = \
    derivative_curvilinear_grid.derivative_updated(\
      var_3d[:,:,:,:,0], xi, eta, zeta, jacobian, accuracy)

  print('shape of derivative',
        dvel_dx_1D.shape, xi.shape, eta.shape, zeta.shape, num_cell)
  dvel_dx_3D = structure_format.array_1D_to_3D(\
    xi, eta, zeta, num_cell, dvel_dx_1D[:, 0])
  dvel_dy_3D = structure_format.array_1D_to_3D(\
    xi, eta, zeta, num_cell, dvel_dy_1D[:, 0])
  dvel_dz_3D = structure_format.array_1D_to_3D(\
    xi, eta, zeta, num_cell, dvel_dz_1D[:, 0])

  #### analytical w.r.t xi, eta
  dvar_dxi_analy = np.zeros([num_xi,num_eta])
  dvar_deta_analy = np.zeros([num_xi,num_eta])
  for i_xi in range(num_xi):
    for i_eta in range(num_eta):
      # dvar_dxi_analy[i_xi, i_eta] = 8*(4*i_xi + 10*i_eta)
      # dvar_deta_analy[i_xi, i_eta] = 20 * (4 * i_xi + 10 * i_eta)
      dvar_dxi_analy[i_xi, i_eta] = \
        (np.pi/num_xi) * 4 * np.cos((np.pi/num_xi) *(4 * i_xi + 10 * i_eta))
      dvar_deta_analy[i_xi, i_eta] = \
        (np.pi/num_xi) * 10 * np.cos((np.pi/num_xi) *(4 * i_xi + 10 * i_eta))


  fig, [ax1, ax2] = plt.subplots(1,2)
  ax1.set_xlabel('nodes')
  ax1.set_ylabel('variable', color='k')
  ax1.plot(var_3d[:,1,0,0,0], color='g', label= 'var_x')
  ax1.plot(var_3d[1, :, 0, 0, 0], color='g', label = 'var_y')
  ax1.plot(var_3d[0, 1, :, 0, 0], color='g', label = 'var_z')
  ax1.legend()
  #ax1.tick_params(axis='y', labelcolor=color)
  #plt.grid()

  #ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  ax2.set_xlabel('nodes')
  ax2.set_ylabel('dvar', color='k')  # we already handled the x-label with ax1
  ax2.plot(dvel_dx_3D[:,1,0], 'r', linewidth = 4, label = 'dvdx, calc')
  ax2.plot(dvel_dy_3D[1, :, 1], 'r', linewidth=4, label='dvdy, calc')
  ax2.plot(dvel_dz_3D[0, 1, :], 'r', linewidth=4, label='dvdz, calc')
  ax2.plot(dvar_dx_analy[:,1,0,0,0],
           color='g', linewidth = 2, label = 'dvdx, analy')
  ax2.plot(dvar_dy_analy[1, :, 0, 0, 1],
           color='g', linewidth=2, label='dvdy, analy')
  ax2.plot(dvar_dz_analy[0, 1, :, 0, 2],
           color='g', linewidth=2, label='dvdz, analy')
  ax2.legend()
  plt.savefig('../plots/deriv_test_3d_harmonic_nonUniform.png')

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.show()
  import ipdb
  ipdb.set_trace()
  #plt.savefig('../plots/curv_deriv_harmnicfunc_data_grid.png')
  # vmin = 0
  # vmax = 0
  # Derivative_Calc.plot_contour(\
  #   cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], var[:,:,0,0,0],
  #   '../plots/curv_harmnicfunc_v.png', 256, vmin, vmax)
  #
  # Derivative_Calc.plot_contour(\
  #   cell_center[:,:,0,0], cell_center[:,:,0,1], dvel_dx_3D[:,:,0],
  #   '../plots/curv_harmnicfunc_dvdx.png', 256, vmin, vmax)
  #
  # Derivative_Calc.plot_contour(\
  #   cell_center[:, :, 0, 0], cell_center[:, :, 0, 1], dvel_dy_3D[:, :, 0],
  #   '../plots/curv_harmnicfunc_dvdy.png', 256, vmin, vmax)
  #
  # Derivative_Calc.plot_contour(\
  #   cell_center[:,:,0,0], cell_center[:,:,0,1], dvar_dx_analy[:,:,0,0,0],
  #   '../plots/curv_harmnicfunc_dvdx_analy.png', 256, vmin, vmax)
  #
  # Derivative_Calc.plot_contour(\
  #   cell_center[:, :, 0, 0], cell_center[:, :, 0, 1],
  #   dvar_dy_analy[:, :, 0, 0, 0],
  #   '../plots/curv_harmnicfunc_dvdy_analy.png', 256, vmin, vmax)
  #
  # Derivative_Calc.plot_contour(\
  #   cell_center[:,:,0,0], cell_center[:,:,0,1],
  #   (dvar_dx_analy[:,:,0,0,0] - dvel_dx_3D[:,:,0]),
  #   '../plots/curv_harmnicfunc_dvdx_diff.png', 256, vmin, vmax)
  #
  # Derivative_Calc.plot_contour(\
  #   cell_center[:,:,0,0], cell_center[:,:,0,1],
  #   (dvar_dy_analy[:,:,0,0,0] - dvel_dy_3D[:,:,0]),
  #   '../plots/curv_harmnicfunc_dvdy_diff.png', 256, vmin, vmax)
  # print('velocity_1D', velocity_3D[:,50,0,0])
  return()

# -----------------------------------------------------------------------------
def manufactured_3d_field():
  num_xi = 50
  num_eta = 30
  num_zeta = 20
  num_dim = 3
  num_snap = 5
  num_cell = num_xi * num_eta*num_zeta

  ### testing jacobian when x, y and var are smooth functions
  cell_center = np.zeros([num_xi, num_eta, num_zeta, num_dim])
  xi = np.zeros([num_cell])
  eta = np.zeros([num_cell])
  zeta = np.zeros([num_cell])
  for i_xi in range(num_xi):
    for i_eta in range(num_eta):
      for i_zeta in range(num_zeta):
        ldx = i_xi + num_xi * (i_eta + num_eta * i_zeta)
        xi[ldx] = int(i_xi)
        eta[ldx] = int(i_eta)
        zeta[ldx] = int(i_zeta)
        # cell_center[i_xi, :, :, 0] = \
        #   i_xi / num_xi + (8*i_eta/num_eta) + (4* i_zeta/num_zeta)
        # cell_center[:, i_eta, :, 1] = \
        #   5*i_eta / num_eta + i_xi/num_xi + 10* i_zeta/num_zeta
        # cell_center[:, :, i_zeta, 2] = \
        #   i_zeta / num_zeta + 2*i_eta/num_eta + 7* i_xi/num_xi
        cell_center[i_xi, :, :, 0] = i_xi/num_xi
        cell_center[:, i_eta, :, 1] = i_eta/num_eta
        cell_center[:, :, i_zeta, 2] = i_zeta/num_zeta

  xi = xi.astype('int')
  eta = eta.astype('int')
  zeta = zeta.astype('int')

  var_3d = np.zeros([num_xi, num_eta, num_zeta, num_dim, num_snap])
  for i_snap in range(num_snap):
    for i_dim in range(num_dim):
      for i_xi in range(num_xi):
        for i_eta in range(num_eta):
          for i_zeta in range(num_zeta):
            var_3d[i_xi, i_eta, i_zeta, i_dim, i_snap] = \
              (np.sin(np.pi * cell_center[i_xi, i_eta, i_zeta, 0]) + \
               np.cos(np.pi * cell_center[i_xi, i_eta, i_zeta, 1]) + \
               np.sin(2 * np.pi * cell_center[i_xi, i_eta, i_zeta, 2]))* \
              (i_dim+1)*(i_snap+1)*((i_xi+1)/num_xi)*((i_eta+1)/num_eta) * \
              ((i_zeta+1)/num_zeta)

  return (var_3d, cell_center, xi, eta, zeta)
