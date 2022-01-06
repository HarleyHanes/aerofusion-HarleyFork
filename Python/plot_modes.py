import sys
import io
import os
import UQtoolbox as uq
import aerofusion.data.array_conversion as arr_conv
from aerofusion.rom import incompressible_navier_stokes_rom as incrom
import numpy as np
import logging
import argparse
import libconf
from aerofusion.plot.plot_2D import plot_contour
from aerofusion.plot.plot_2D import plot_pcolormesh
import matplotlib.pyplot as plt
import scipy.io as mio

def main(argv=None):
    #Variables to load
    filename = "../../lid_driven_snapshots/modal coeff/results_lsa.npz"
    points = np.array([0, 50, 128, 150])
    #Loop through modes
    results = np.load(filename)
    jac = results["sensitivities"]
    modes_plotted = np.array([0, 1, 5, 10, 49])
    
    #Load mesh data
    rom_matrices_filename="../../lid_driven_penalty/rom_matrices_50.npz"
    #penalty=10.0**4 penalty defined explicitly in function
    QOI_type = "modal coeff"
    tmax=100
    penalty_exp=4
    
    #Specify when integration takes place 
    integration_times = np.arange(.1,tmax,.1)
    integration_indices = np.arange(1,len(integration_times)+1)
    num_points = len(points)
    num_times = len(integration_times)
    

    data_folder = "../../lid_driven_snapshots/"
    plot_folder = "../../lid_driven_snapshots/modal coeff/"
    pod_data = np.load(data_folder + 'pod_lid_driven_50.npz')
    # Assign data to convenience variables
    vel_0  = pod_data['velocity_mean']
    simulation_time = pod_data['simulation_time']
    phi             = pod_data['phi']
    modal_coeff     = pod_data['modal_coeff']
    #integration_indices = np.arange(1,500)
    #integration_times = simulation_time[integration_indices]
    num_time = len(integration_times)
    
    #Seperate Jac into modes
    jac_seperated = np.empty((num_points, 50, num_times))
    for i in range(num_points):
        jac_seperated[i, :, :]= np.reshape(jac[:,i], (50, num_times))
    
    #Plot Modes
    for i in range(num_points):
        for j in range(len(modes_plotted)):
            plt.figure()
            plt.plot(integration_times[0:600], jac_seperated[i,j,0:600])
            plt.xlabel("Time (s)")
            plt.ylabel(r"$\alpha_" + str(j+1)+"$")
            plt.title("Sensitivity of POD modes to Xi=" + str(points[i]))
            plt.savefig(plot_folder + "LocalSensmode_" + str(j) + "_Xi=" + str(points[i]) + ".png")

if __name__ == "__main__":
    sys.exit(main())
