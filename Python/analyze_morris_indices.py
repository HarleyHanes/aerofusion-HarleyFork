# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:03:12 2022

@author: USER
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

def main(argv=None):
    data_folder = "../../lid_driven_data/sensitivity/"
    data_set_first_name = "s150m100_l40_tForward1_nSamp20_morris_indices.npz"
    data_set_second_name = "s150m100_l40_tForward2_nSamp20_morris_indices.npz"
    
    #load data sets
    data_set_first = np.load(data_folder + data_set_first_name)
    data_set_second = np.load(data_folder + data_set_second_name)
    
    morris_mean_abs = np.array([data_set_first["morris_mean_abs"], data_set_second["morris_mean_abs"]])
    morris_mean = np.array([data_set_first["morris_mean"], data_set_second["morris_mean_abs"]])
    
    
    
    
    morris_abs_scaling = np.amax(morris_mean_abs, axis = 1,keepdims = True)
    morris_mean_abs_scaled = morris_mean_abs/ morris_abs_scaling
    
    morris_scaling = np.amax(np.abs(morris_mean), axis = 1,keepdims = True)
    morris_mean_scaled = morris_mean/ morris_scaling
    
    make_plot(morris_mean_abs_scaled[0], "morris_abs_1forward_scaled.png",\
              "Morris Absolute Mean at t = 150","Proportion of Max Sensitivity",\
              data_folder)
    
    make_plot(morris_mean_abs_scaled[1], "morris_abs_2forward_scaled.png",\
              "Morris Absolute Mean at t = 300","Proportion of Max Sensitivity",\
              data_folder)
        
    make_plot(morris_mean_abs[0], "morris_abs_1forward_unscaled.png",\
              "Morris Absolute Mean at t = 150","Sensitivity",\
              data_folder)     
        
    make_plot(morris_mean_abs[1], "morris_abs_2forward_unscaled.png",\
              "Morris Absolute Mean at t = 300","Sensitivity",\
              data_folder)
              
    make_plot(morris_mean_scaled[0], "morris_1forward_scaled.png",\
              "Morris Mean at t = 150","Proportion of Max Sensitivity",\
              data_folder)
    
    make_plot(morris_mean_scaled[1], "morris_2forward_scaled.png",\
              "Morris Mean at t = 300","Proportion of Max Sensitivity",\
              data_folder)
        
    make_plot(morris_mean[0], "morris_1forward_unscaled.png",\
              "Morris Mean at t = 150","Sensitivity",\
              data_folder)     
        
    make_plot(morris_mean[1], "morris_2forward_unscaled.png",\
              "Morris Mean at t = 300","Sensitivity",\
              data_folder)
    
    # fig, (ax1, ax2) = plt.subplots(figsize=(13, 3), ncols=2)
    # pos = ax1.imshow(morris_mean_abs_scaled[0])
    # ax1.set_xticks(x_ticks_breaks)
    # ax1.set_yticks(y_ticks_breaks)
    # ax1.grid(axis = 'both', which = 'major', linewidth = 1.5)
    # fig.colorbar(pos, ax=ax1)
    
    # pos = ax2.imshow(morris_mean_abs_scaled[1])
    # fig.colorbar(pos, ax=ax2)
    # plt.show()

def make_plot(morris_indices, save_name, title_label, color_bar_label, data_folder):
    fig, ax1 = plt.subplots(figsize=(20, 20), ncols=1)
    pos = ax1.imshow(morris_indices)
    y_ticks_labels_loc = np.array([0, 1, 2, 6, 13, 20, 27, 34])
    y_labels = np.array(["Re", "alpha", "penalty",  "Local \n Speed", "Basis \n Orientation", "x-location",\
                         "y-location", "axis \n extent"])
    x_ticks_labels_loc = np.array([0, 1, 2, 6])
    x_labels = np.array(["Energy", "Min Vorticity", "Max Vorticity", "Local \n Vorticity"])
    y_ticks_breaks = np.array([3, 10, 17, 24, 31])-.5
    x_ticks_breaks = np.array([2])+.5
    ax1.set_xticks(x_ticks_breaks, minor = True)
    ax1.set_yticks(y_ticks_breaks, minor = True)
    ax1.set_yticks(y_ticks_labels_loc)
    ax1.set_yticklabels(y_labels)
    ax1.set_xticks(x_ticks_labels_loc)
    ax1.set_xticklabels(x_labels, rotation = "90")
    ax1.set_title(title_label, pad =20)
    ax1.grid(axis = 'both', which = 'minor', linewidth = 4, color = 'r')
    fig.colorbar(pos, ax=ax1, label = color_bar_label)
    plt.tight_layout()
    plt.savefig(data_folder + "morris_1forward_scaled.png")

if __name__ == "__main__":
    sys.exit(main())
