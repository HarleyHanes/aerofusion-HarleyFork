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

def main(argv=None):
    #Variables to load
    num_modes=np.array([2, 3,4, 5, 6, 7,8,9,10,15,20,25,30,40,50]) #4 6
    mean_prefix = 'sensitivities/morris_mu'
    std_prefix = 'sensitivities/morris_sigma'
    data_prefix = 'solutions/mean_reduced'
    distance_fig_name='plots/sens_distance'
    #Loop through modes
    mean=[]
    std=[]
    data=[]
    for i in range(len(num_modes)):
        #Load sensitivity
        i_modes=num_modes[i]
        i_mean=np.load(mean_prefix + '_m' + str(i_modes)+ '.npy')
        i_std = np.load(std_prefix + '_m' + str(i_modes)+ '.npy')
        i_data=np.load(data_prefix + '_m' + str(i_modes)+ '.npy')
        print("Mode " + str(i_modes) + " base shape:" + str(i_data.shape))
        print("Mode " + str(i_modes) + " std shape: " + str(i_std.shape))
        print("Mode " + str(i_modes) + " sens: " + str(i_mean[:2,:2,:2]))
        print("Mode " + str(i_modes) + " data: " + str(i_data[:2,:2,:2]))
        print("Mode " + str(i_modes) + " std: " + str(i_std[:2,:2,:2]))
        #Store sensitivities
        mean.append(i_mean)
        std.append(i_std)
        data.append(i_data)
    #Caclulate Distances
    #sens_error=np.sum((sensitivities[:]-sensitivities[-1])**2, axis=(1,2,3))/np.sum(sensitivities[-1]**2)
   # data_error=np.sum((data[:]-data[-1])**2,axis=(1,2,3))/np.linalg.norm(data[-1]**2)
    #print(sensitivities[:].shape)
    #print(sensitivities[-1].shape)
    std_sum=np.sum((std[:]-std[-1])**2, axis=(1,2,3))
    std_error=std_sum/np.sum(std[-1]**2)

    mean_sum=np.sum((mean[:]-mean[-1])**2, axis=(1,2,3))
    mean_error=mean_sum/np.sum(mean[-1]**2)

    data_error=np.sum((data[:]-data[-1])**2,axis=(1,2,3))/np.sum(data[-1]**2)
    print(data_error)
    print(mean_error)
    print(std_error)
    #print("Sensitivity: " + str(sensitivities))
    #print("Sensitivity Error: " +str(sens_error))
    #print("Data Error: " + str(data_error))
    #sens_distance=np.empty((len(num_modes)))
    #for i in range(len(num_modes)):
    #    sens_distance[i]=np.sum((sensitivities[i]**2)-(sensitivities[-1]**2))/np.sum((sensitivities[-1]**2))
    plt.plot(num_modes, mean_error,'b-')
    plt.plot(num_modes, std_error,'r--')
    plt.plot(num_modes, data_error,'g:')
    plt.xlabel('Number of Modes')
    plt.title('Streamline Velocity Errors Compared to 50 Mode System')
    plt.ylabel('Relative MSE')
    plt.legend(['Morris Mean', 'Morris Standard Deviation','Reconstructed Solution'])
    plt.savefig(distance_fig_name + '.png')
    plt.savefig(distance_fig_name + '.pdf')
    plt.savefig(distance_fig_name + '.eps', format = "eps")
 #fig, ax = plt.subplots(1,2, sharey=True)
    #ax1.plot(num_modes, sens_error)
    #ax2.plot(num_modes, data_error)
    #ax1.ylabel('Relative

if __name__ == "__main__":
    sys.exit(main())
