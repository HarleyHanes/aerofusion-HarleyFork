import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

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
def plot_pcolormesh(\
      x,
      y,
      field,
      output_filename,
      *args,
      **kwargs):

  # Default values
  options = \
    {
    "fig_size" : (16,12),
    "title" : "",
    "font_size" : 12,
    "vmin" : "auto",
    "vmax" : "auto",
    "cmap" : "jet",
    "colorbar_label" : "field",
    "xlabel" : "$x$",
    "ylabel" : "$y$",
    }
  # Update with kwargs
  for key, value in kwargs.items():
    options[key] = value

  # Plot 3D array using structured-grid underlying topology
  import matplotlib.pyplot as plt
  plt.rcParams.update({'font.size': options["font_size"]})
  fig = plt.figure(figsize = options["fig_size"])
  ax = fig.add_subplot()
  vmin = options["vmin"]
  if vmin == "auto":
    vmin = field.min()
  vmax = options["vmax"]
  if vmax == "auto":
    vmax = field.max()
  im = ax.pcolormesh(\
               x,
               y,
               field,
               cmap = options["cmap"],
               vmin = vmin,
               vmax = vmax)
  cbar = fig.colorbar(im, ax=ax)
  cbar.set_label(options["colorbar_label"], rotation = 270, labelpad = options["font_size"]+4, fontsize = options["font_size"])
 # fig.colorbar(im, label = options["colorbar_label"], rotation='180')
  ax.set_title(options["title"])
  ax.set_xlabel(options["xlabel"])
  ax.set_ylabel(options["ylabel"])
  # Create parent directory if it does not exist
  os.makedirs(Path(output_filename).parent, exist_ok = True)
  plt.savefig(output_filename)
  fig.clf()

