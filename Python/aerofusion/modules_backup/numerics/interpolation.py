# -----------------------------------------------------------------------------
# \file interpolation.py
# \brief Interpolate fields
# -----------------------------------------------------------------------------

import numpy as np
from scipy.interpolate import griddata

from aerofusion.data import array_conversion as arr_conv
from aerofusion.plot.plot_2D import plot_contour

# -----------------------------------------------------------------------------
#ivanComment This function has the y_bump hardcoded, and should be generalized
def interpolate_vectorial_field_in_2D(\
  domain_x_min,
  domain_x_max,
  domain_y_min,
  domain_y_max,
  num_cell,
  num_dimensions,
  xi,
  eta,
  zeta,
  vector_1D,
  vector_3D,
  cell_center,
  num_x,
  num_y,
  direction,
  snap):

  # interpolating values into Cartesian grid
  x = np.linspace(domain_x_min, domain_x_max, num_x)
  y = np.linspace(domain_y_min, domain_y_max, num_y)
  grid_x, grid_y = np.meshgrid(x,y)
  grid_x = grid_x.transpose()
  grid_y = grid_y.transpose()

  cell_center_1D = np.zeros([num_cell, num_dimensions])
  for i_dim in range(num_dimensions):
    cell_center_1D[:, i_dim] = \
      array_3D_to_1D(\
        xi, eta, zeta, num_cell, cell_center[:, :, :, i_dim])

  vector_interp_nearest = \
    griddata((cell_center_1D[:, 0], cell_center_1D[:, 1]),
             vector_1D[:, direction, snap],
             (grid_x, grid_y),
             method='nearest')
  vector_interp_linear = \
    griddata((cell_center_1D[:, 0], cell_center_1D[:, 1]),
             vector_1D[:, direction, snap],
             (grid_x, grid_y),
             method='linear')
  vector_interp_cubic = \
    griddata((cell_center_1D[:, 0], cell_center_1D[:, 1]),
             vector_1D[:, direction, snap],
             (grid_x, grid_y),
             method='cubic')

  #---- recreating bump and set the values inside bum equal to zero
  y_bump = 0.085 * np.exp(-(grid_x[:, 0] / 0.195) ** 2)

  for i_x in range(num_x):
      for i_y in range(num_y):
          if grid_y[i_x, i_y] < y_bump[i_x]:
              vector_interp_nearest[i_x, i_y] = 0.0
              vector_interp_linear[i_x, i_y]  = 0.0
              vector_interp_cubic[i_x, i_y]   = 0.0

  #vmin = np.min(vector_3D[:,:,0,direction])
  #vmax = np.max(vector_3D[:,:,0,direction])
  #plot_contour(\
  #  cell_center[:, :, 0, 0],
  #  cell_center[:, :, 0, 1],
  #  vector_3D[:, :, 0, direction, snap],
  #  'no-interpolation_dir' + str(direction) + '.png',
  #  256,
  #  vmin,
  #  vmax)
  #plot_contour(\
  #  grid_x,
  #  grid_y,
  #  vector_interp_nearest,
  #  'nearest_interpol_dir' + str(direction) + '.png',
  #  256,
  #  vmin,
  #  vmax)
  #plot_contour(\
  #  grid_x,
  #  grid_y,
  #  vector_interp_linear,
  #  'linear_interpol_dir' + str(direction) + '.png',
  #  256,
  #  vmin,
  #  vmax)
  #plot_contour(\
  #  grid_x,
  #  grid_y,
  #  vector_interp_cubic,
  #  'cubic_interpol_dir' + str(direction) + '.png',
  #  256,
  #  vmin,
  #  vmax)

  return (vector_interp_nearest, vector_interp_linear, vector_interp_cubic)
