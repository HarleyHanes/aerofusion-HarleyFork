import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
def plot_contour(X, Y, Z, output_filename, n_levels, v_min, v_max):

  fig, axs = plt.subplots(nrows=1, ncols=1)
  fig.subplots_adjust(hspace=0.3)
  levels = np.linspace(v_min, v_max, n_levels)

  #cset1 = axs.contourf(X, Y, Z, levels=levels, cmap='bwr', extend = 'both')
  cset1 = axs.contourf(X, Y, Z, levels = n_levels, cmap='bwr')
  #axs.set_xlim([0.4, 1])
  fig.colorbar(cset1)
  fig.tight_layout()
  plt.savefig(output_filename)

# -----------------------------------------------------------------------------
def Find_Derivative_FD(var, cell_center, order):
  dim = var.shape
  Nxi = dim[0]
  Neta = dim[1]
  Nzeta = dim[2]
  Ndim = dim[3]
  dvar_dx = np.zeros(var.shape)
  dvar_dy = np.zeros(var.shape)
  (dcell_dxi, dcell_deta) = FD_derivative_2nd_order(cell_center)
  dx_dxi = np.zeros([Nxi, Neta, Nzeta])
  dx_dxi[:,:,:] = dcell_dxi[:,:,:,0]
  dy_deta = np.zeros([Nxi, Neta, Nzeta])
  dy_deta[:, :, :] = dcell_deta[:, :, :, 1]
  if order == 2 :
    (dvel_dxi, dvel_deta) = FD_derivative_2nd_order(var)
  elif order == 4 :
    (dvel_dxi, dvel_deta) = FD_derivative_4th_order(var)
  elif order == 6 :
    (dvel_dxi, dvel_deta) = FD_derivative_6th_order(var)
  else:
    print("The requested order is wrong. Choose between 2, 4, or 6.")

  for i_dim in range(Ndim):
    dvar_dx[:, :, 0, i_dim] = \
      np.divide(dvel_dxi[:, :, 0, i_dim] ,dx_dxi[:, :, 0])
    dvar_dy[:, :, 0, i_dim] = \
      np.divide(dvel_deta[:, :, 0, i_dim], dy_deta[:, :, 0])

  return (dvar_dx, dvar_dy)

# -----------------------------------------------------------------------------
def FD_derivative_2nd_order(Var):
  dim = Var.shape
  Nxi = dim[0]
  Neta = dim[1]
  Nzeta = dim[2]
  Ndim = dim[3]
  dVar_dxi = np.zeros(dim)
  dVar_deta = np.zeros(dim)

  for i_dim in range(Ndim):
    for xi in range(Nxi):
      if xi == 0:
        dVar_dxi[xi, :, :, i_dim] = \
          Var[xi + 1, :, :, i_dim] - Var[xi, :, :, i_dim]
      elif xi == Nxi - 1:
        dVar_dxi[xi, :, :, i_dim] = \
          Var[xi, :, :, i_dim] - Var[xi - 1, :, :, i_dim]
      else:
        dVar_dxi[xi, :, :, i_dim] = \
          + 1.0/2.0 * Var[xi + 1, :, :, i_dim] \
          - 1.0/2.0 * Var[xi - 1, :, :, i_dim]

  for i_dim in range(Ndim):
    for eta in range(Neta):
      if eta == 0:
        dVar_deta[:, eta, :, i_dim] = \
          Var[:, eta+1, :, i_dim] - Var[:, eta, :, i_dim]
      elif eta == Neta - 1:
        dVar_deta[:, eta, :, i_dim] = \
          Var[:, eta, :, i_dim] - Var[:,eta - 1, :, i_dim]
      else:
        dVar_deta[:, eta, :, i_dim] = \
          + 1.0/2.0 * Var[:, eta + 1, :, i_dim] \
          - 1.0/2.0 * Var[:, eta - 1, :, i_dim]


  return (dVar_dxi, dVar_deta)

# -----------------------------------------------------------------------------
def FD_derivative_4th_order(Var):
  dim = Var.shape
  Nxi = dim[0]
  Neta = dim[1]
  Nzeta = dim[2]
  Ndim = dim[3]
  dVar_dxi = np.zeros(dim)
  dVar_deta = np.zeros(dim)

  for i_dim in range(Ndim):
    for xi in range(Nxi):
      if xi == 0:
        dVar_dxi[xi, :, :, i_dim] = \
          Var[xi+1, :, :, i_dim] - Var[xi, :, :, i_dim]
      elif xi == Nxi-1:
        dVar_dxi[xi,:,:,i_dim] = \
          Var[xi,:,:,i_dim] - Var[xi-1,:,:,i_dim]
      elif 0 < xi < 3:
        dVar_dxi[xi, :, :, i_dim] = \
          - 25.0 / 12.0 * Var[xi, :, :, i_dim] \
          +  4.0        * Var[xi+1, :, :, i_dim] \
          -  3.0        * Var[xi+2, :, :, i_dim] \
          +  4.0 / 3.0  * Var[xi+3, :, :, i_dim] \
          -  1.0 / 4.0  * Var[xi+4, :, :, i_dim]
      elif Nxi-1 > xi > Nxi-4:
        dVar_dxi[xi, :, :, i_dim] = \
          + 25.0 / 12.0 * Var[xi, :, :, i_dim] \
          - 4.0         * Var[xi-1, :, :, i_dim] \
          + 3.0         * Var[xi-1, :, :, i_dim] \
          - 4.0 / 3.0   * Var[xi-3, :, :, i_dim] \
          + 1.0 / 4.0   * Var[xi-4, :, :, i_dim]
      else:
        dVar_dxi[xi,:,:,i_dim] = \
          + 1.0/12.0 * Var[xi-2,:,:,i_dim] \
          - 2.0/3.0  * Var[xi-1,:,:,i_dim] \
          + 2.0/3.0  * Var[xi+1,:,:,i_dim] \
          - 1.0/12.0 * Var[xi+2,:,:,i_dim]

  for i_dim in range(Ndim):
    for eta in range(Neta):
      if eta == 0:
        dVar_deta[:, eta, :, i_dim] = \
          Var[:, eta + 1, :, i_dim] - Var[:,eta, :, i_dim]
      elif eta == Neta - 1:
        dVar_deta[:, eta, :, i_dim] = \
          Var[:, eta, :, i_dim] - Var[:,eta - 1, :, i_dim]
      elif 0 < eta < 3:
        dVar_deta[:, eta, :, i_dim] = \
          - 25.0 / 12.0 * Var[:, eta, :, i_dim] \
          +  4.0        * Var[:, eta + 1, :, i_dim] \
          -  3.0        * Var[:, eta + 2, :, i_dim] \
          +  4.0 / 3.0  * Var[:, eta + 3, :, i_dim] \
          -  1.0 / 4.0  * Var[:, eta + 4, :, i_dim]
      elif Neta - 1 > eta > Neta - 4:
        dVar_deta[:, eta, :, i_dim] = \
          + 25.0 / 12.0 * Var[:, eta, :, i_dim] \
          -  4.0        * Var[:, eta - 1, :, i_dim] \
          +  3.0        * Var[:, eta - 2, :, i_dim] \
          -  4.0 / 3.0  * Var[:, eta - 3, :, i_dim] \
          +  1.0 / 4.0  * Var[:, eta - 4, :, i_dim]
      else:
        dVar_deta[:, eta, :, i_dim] = \
          + 1.0 / 12.0 * Var[:, eta - 2, :, i_dim] \
          - 2.0 /  3.0 * Var[:, eta - 1, :, i_dim] \
          + 2.0 /  3.0 * Var[:, eta + 1, :, i_dim] \
          - 1.0 / 12.0 * Var[:, eta + 2, :, i_dim]

  return (dVar_dxi, dVar_deta)

# -----------------------------------------------------------------------------
def FD_derivative_6th_order(Var):
  dim = Var.shape
  Nxi = dim[0]
  Neta = dim[1]
  Nzeta = dim[2]
  Ndim = dim[3]
  dVar_dxi = np.zeros(dim)
  dVar_deta = np.zeros(dim)

  for i_dim in range(Ndim):
    for xi in range(Nxi):
      if xi == 0:
        dVar_dxi[xi, :, :, i_dim] = \
          (Var[xi+1, :, :, i_dim] - Var[xi, :, :, i_dim])
      elif xi == Nxi-1:
        dVar_dxi[xi,:,:,i_dim] = \
          (Var[xi,:,:,i_dim] - Var[xi-1,:,:,i_dim])
      elif 0 < xi < 3:
        dVar_dxi[xi, :, :, i_dim] = \
          - 49.0/20.0 * Var[xi, :, :, i_dim]   \
          +  6.0      * Var[xi+1, :, :, i_dim] \
          - 15.0/2.0  * Var[xi+2, :, :, i_dim] \
          + 20.0/3.0  * Var[xi+3, :, :, i_dim] \
          - 15.0/4.0  * Var[xi+4, :, :, i_dim] \
          +  6.0/5.0  * Var[xi+5, :, :, i_dim] \
          -  1.0/6.0  * Var[xi+6, :, :, i_dim]
      elif Nxi-1 > xi > Nxi-4:
        dVar_dxi[xi, :, :, i_dim] = \
          + 49.0 / 20.0 * Var[xi, :, :, i_dim] \
          -  6.0        * Var[xi - 1, :, :, i_dim] \
          + 15.0 / 2.0  * Var[xi - 2, :, :, i_dim] \
          - 20.0 / 3.0  * Var[xi - 3, :, :, i_dim] \
          + 15.0 / 4.0  * Var[xi - 4, :, :, i_dim] \
          -  6.0 / 5.0  * Var[xi - 5, :, :, i_dim] \
          +  1.0 / 6.0  * Var[xi - 6, :, :, i_dim]
      else:
        dVar_dxi[xi,:,:,i_dim] = \
          - 1.0/60.0 * Var[xi-3,:,:,i_dim] \
          + 3.0/20.0 * Var[xi-2,:,:,i_dim] \
          - 3.0/4.0  * Var[xi-1,:,:,i_dim] \
          + 3.0/4.0  * Var[xi+1,:,:,i_dim] \
          - 3.0/20.0 * Var[xi+2,:,:,i_dim] \
          + 1.0/60.0 * Var[xi+3,:,:,i_dim]

  for i_dim in range(Ndim):
    for eta in range(Neta):
       if eta == 0:
         dVar_deta[:, eta, :, i_dim] = \
           (Var[:, eta+1, :, i_dim] - Var[:, eta, :, i_dim])
       elif eta == Neta-1:
         dVar_deta[:, eta, :, i_dim] = \
           (Var[:,eta,:,i_dim] - Var[:, eta-1,:,i_dim])
       elif 0 < eta < 3:
         dVar_deta[:, eta, :, i_dim] = \
           - 49.0/20.0 * Var[:, eta, :, i_dim] \
           +  6.0      * Var[:, eta+1, :, i_dim] \
           - 15.0/2.0  * Var[:, eta+2, :, i_dim] \
           + 20.0/3.0  * Var[:, eta+3, :, i_dim] \
           - 15.0/4.0  * Var[:, eta+4, :, i_dim] \
           +  6.0/5.0  * Var[:, eta+5, :, i_dim] \
           -  1.0/6.0  * Var[:, eta+6, :, i_dim]
       elif Neta-1 > eta > Neta-4:
         dVar_deta[:, eta, :, i_dim] = \
           + 49.0 / 20.0 * Var[:, eta, :, i_dim] \
           - 6.0         * Var[:, eta - 1, :, i_dim] \
           + 15.0 / 2.0  * Var[:, eta - 2, :, i_dim] \
           - 20.0 / 3.0  * Var[:, eta - 3, :, i_dim] \
           + 15.0 / 4.0  * Var[:, eta - 4, :, i_dim] \
           -  6.0 / 5.0  * Var[:, eta - 5, :, i_dim] \
           +  1.0 / 6.0  * Var[:, eta - 6, :, i_dim]
       else:
         dVar_deta[:, eta, :, i_dim] = \
           - 1.0/60.0 * Var[:, eta-3,:,i_dim] \
           + 3.0/20.0 * Var[:, eta-2,:,i_dim] \
           - 3.0/4.0  * Var[:, eta-1,:,i_dim] \
           + 3.0/4.0  * Var[:, eta+1,:,i_dim] \
           - 3.0/20.0 * Var[:, eta+2,:,i_dim] \
           + 1.0/60.0 * Var[:, eta+3,:,i_dim]

    return (dVar_dxi, dVar_deta)

# -----------------------------------------------------------------------------
def cheb_derivative(var):
  dim = var.shape
  Nxi = dim[0]
  Neta = dim[1]
  Nzeta = dim[2]
  Ndim = dim[3]
  dvar_dxi = np.zeros(dim)
  dvar_deta = np.zeros(dim)

  # for now we assume , Nxi=Neta
  import cheb_function
  D = cheb_function.cheb(Nxi)
  # print('size of D', D.shape, Nxi, Neta)
  # plot_contour(var[:, :, 0, 0], var[:, :, 0, 1], D, "D.png")
  for i_dim in range(Ndim):
    dvar_dxi[:,:,0,i_dim] = np.matmul(-var[:,:,0,i_dim], D.transpose())
    dvar_deta[:, :, 0, i_dim] = np.matmul(D, var[:, :, 0, i_dim])

  return (dvar_dxi, dvar_deta)
