# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 11:03:12 2022

@author: USER
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import os

def main(argv=None):
    
    use_energy = False
    mean_type = 'pert'
    Re = 17000
    
    #tForward = '1.0'
    snapshots=150
    modes=100
    energy =0.99
    delta = 1e-6
    nSamp = 40
    qoi_type = "intQOI"
    
    data_folder = "../../lid_driven_data/sensitivity/"
    data_set_prefix = "Re" + str(Re) +"_" + mean_type + "_s" + str(snapshots) 
    if use_energy:
        data_set_prefix += "e" + str(energy)
    else : 
        data_set_prefix += "m" + str(100)
    data_set_prefix += "_l" + str(int(1/delta))
    data_set_suffix = "_nSamp" + str(nSamp) + "_" + qoi_type + "_morris_indices.npz"
    
    data_set_first_name = data_set_prefix+ "_tForward1.0" + data_set_suffix
    data_set_second_name = data_set_prefix+ "_tForward1.0" + data_set_suffix
    
    plot_folder = "../../Figures/LidDriven/morris/" + data_set_prefix + "_nSamp" +str(nSamp) + "/"
    if not os.path.isdir(plot_folder):
        os.mkdir(plot_folder)
    plot_folder += qoi_type + "_"
        
    filetype = '.pdf'
    plt.rcParams['text.usetex'] = True
    
    #
    param_labels = np.array([r'$Re$', r'$\log(\alpha)$', r'$\log(\tau)$',\
                        r'$\bar{v}_1$', r'$\bar{v}_2$', r'$\bar{v}_3$', r'$\bar{v}_4$', r'$\bar{v}_5$', r'$\bar{v}_6$', r'$\bar{v}_7$',\
                        r'$\theta_1$',r'$\theta_2$',r'$\theta_3$',r'$\theta_4$',r'$\theta_5$',r'$\theta_6$',r'$\theta_7$',
                        r'$\xi_{x1}$',r'$\xi_{x2}$',r'$\xi_{x3}$',r'$\xi_{x4}$',r'$\xi_{x5}$',r'$\xi_{x6}$',r'$\xi_{x7}$',
                        r'$\xi_{y1}$',r'$\xi_{y2}$',r'$\xi_{y3}$',r'$\xi_{y4}$',r'$\xi_{y5}$',r'$\xi_{y6}$',r'$\xi_{y7}$',
                        '$l_1$','$l_2$','$l_3$','$l_4$','$l_5$','$l_6$','$l_7$'
                         ])
    qoi_labels_full = np.array([r'$V$', r'$E_K$', r'$V_1$',r'$V_2$',r'$V_3$',r'$V_4$',r'$V_5$',r'$V_6$',r'$V_7$'])
    qoi_labels_reduced = np.array([r'$V_1$',r'$V_2$',r'$V_3$',r'$V_4$',r'$V_5$',r'$V_6$',r'$V_7$'])
    
    #load data sets
    data_set_first = np.load(data_folder + data_set_first_name)
    data_set_second = np.load(data_folder + data_set_second_name)
    
    morris_mean_abs = np.array([data_set_first["morris_mean_abs"], data_set_second["morris_mean_abs"]])
    morris_mean = np.array([data_set_first["morris_mean"], data_set_second["morris_mean_abs"]])
    morris_std = np.array([data_set_first["morris_std"], data_set_second["morris_std"]])
    
    
    
    
    morris_abs_scaling = np.amax(morris_mean_abs, axis = 1,keepdims = True)
    morris_mean_abs_scaled = morris_mean_abs/ morris_abs_scaling
    
    morris_scaling = np.amax(np.abs(morris_mean), axis = 1,keepdims = True)
    morris_mean_scaled = morris_mean/ morris_scaling
    
    
    morris_std_scaling = np.amax(np.abs(morris_std), axis = 1,keepdims = True)
    morris_std_scaled = morris_std/ morris_std_scaling
    
    #-----------------------------Full data plots------------------------------    
    make_plot(morris_mean_abs_scaled[0], "morris_abs_1forward_scaled" + filetype,\
              r"$\mu^*$ at t = 150","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
    
    make_plot(morris_mean_abs_scaled[1], "morris_abs_2forward_scaled" + filetype,\
              r"$\mu^*$ at t = 300","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
        
    make_plot(morris_mean_abs[0], "morris_abs_1forward_unscaled" + filetype,\
              r"$\mu^*$ at t = 150","Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))     
        
    make_plot(morris_mean_abs[1], "morris_abs_2forward_unscaled" + filetype,\
              r"$\mu^*$ at t = 300","Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
              
    make_plot(morris_mean_scaled[0], "morris_1forward_scaled" + filetype,\
              r"$\mu$ at t = 150","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
    
    make_plot(morris_mean_scaled[1], "morris_2forward_scaled" + filetype,\
              r"$\mu$ at t = 300","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
        
    make_plot(morris_mean[0], "morris_1forward_unscaled" + filetype,\
              r"$\mu$ at t = 150","Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))     
        
    make_plot(morris_mean[1], "morris_2forward_unscaled" + filetype,\
              r"$\mu$ at t = 300","Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
        
    make_plot(morris_std_scaled[0], "morris_std_1forward_scaled" + filetype,\
              r"$\sigma$  at t = 150","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
    
    make_plot(morris_std_scaled[1], "morris_std_2forward_scaled" + filetype,\
              r"$\sigma$ at t = 300","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
        
    make_plot(morris_std[0], "morris_std_1forward_unscaled" + filetype,\
              r"$\sigma$ at t = 150","Standard Deviation",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))     
        
    make_plot(morris_std[1], "morris_std_2forward_unscaled" + filetype,\
              r"$\sigma$ at t = 300","Standard Deviation",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
    
    
    make_plot(morris_mean_abs_scaled[0], "morris_abs_1forward_scaled" + filetype,\
              r"$\mu^*$ at t = 150","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
    
    make_plot(morris_mean_abs_scaled[1], "morris_abs_2forward_scaled" + filetype,\
              r"$\mu^*$ at t = 300","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
        
    make_plot(morris_mean_abs[0], "morris_abs_1forward_unscaled" + filetype,\
              r"$\mu^*$ at t = 150","Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))     
    
    make_plot(morris_mean_abs[1], "morris_abs_2forward_unscaled" + filetype,\
              r"$\mu^*$ at t = 300","Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
              
    make_plot(morris_mean_scaled[0], "morris_1forward_scaled" + filetype,\
              r"$\mu$ at t = 150","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
    
    make_plot(morris_mean_scaled[1], "morris_2forward_scaled" + filetype,\
              r"$\mu$ at t = 300","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
        
    make_plot(morris_mean[0], "morris_1forward_unscaled" + filetype,\
              r"$\mu$ at t = 150","Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))     
        
    make_plot(morris_mean[1], "morris_2forward_unscaled" + filetype,\
              r"$\mu$ at t = 300","Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
        
    make_plot(morris_std_scaled[0], "morris_std_1forward_scaled" + filetype,\
              r"$\sigma$  at t = 150","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
    
    make_plot(morris_std_scaled[1], "morris_std_2forward_scaled" + filetype,\
              r"$\sigma$ at t = 300","Proportion of Max Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
        
    make_plot(morris_std[0], "morris_std_1forward_unscaled" + filetype,\
              r"$\sigma$ at t = 150","Standard Deviation",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))     
        
    make_plot(morris_std[1], "morris_std_2forward_unscaled" + filetype,\
              r"$\sigma$ at t = 300","Standard Deviation",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_full,\
              qoi_ticks_labels = np.arange(0,9))
        
    #-----------------------Local only data plots-----------------------------
    
    make_plot(morris_mean_abs[0,:,2:], "morris_abs_1forward_unscaled_localOnly" + filetype,\
              r"$\mu^*$ at t = 150","Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_reduced,\
              qoi_ticks_labels = np.arange(0,7),
              qoi_ticks_breaks = np.array([]))     
        
    make_plot(morris_mean_abs[1,:,2:], "morris_abs_2forward_unscaled_localOnly" + filetype,\
              r"$\mu^*$ at t = 300","Sensitivity",\
              plot_folder,
              poi_labels = param_labels,
              poi_ticks_labels = np.arange(0,38),
              qoi_labels = qoi_labels_reduced,\
              qoi_ticks_labels = np.arange(0,7),
              qoi_ticks_breaks = np.array([]))
    # fig, (ax1, ax2) = plt.subplots(figsize=(13, 3), ncols=2)
    # pos = ax1.imshow(morris_mean_abs_scaled[0])
    # ax1.set_xticks(poi_ticks_breaks)
    # ax1.set_yticks(qoi_ticks_breaks)
    # ax1.grid(axis = 'both', which = 'major', linewidth = 1.5)
    # fig.colorbar(pos, ax=ax1)
    
    # pos = ax2.imshow(morris_mean_abs_scaled[1])
    # fig.colorbar(pos, ax=ax2)
    # plt.show()

def make_plot(morris_indices, save_name, title_label, color_bar_label, plot_folder,
              poi_labels = np.array(["Re", "alpha", "penalty",  "Velocity \n Magnitude",\
                                    "Basis \n Orientation", "x-location",\
                                    "y-location", "axis \n extent"]),\
              qoi_labels = np.array(["Energy", "Vorticity", "Local \n Vorticity"]),\
              poi_ticks_labels = np.array([0, 1, 2, 6, 13, 20, 27, 34]),\
              qoi_ticks_labels = np.array([0, 1, 5]),\
              poi_ticks_breaks = np.array([3, 10, 17, 24, 31])-.5,\
              qoi_ticks_breaks = np.array([1])+.5, 
              rotate_y = True,
              orient = 'vertical'):
    if orient.lower() == "horizontal": 
        fig, ax1 = plt.subplots(figsize=(8,2.3), ncols=1)
        pos = ax1.imshow(morris_indices.transpose(), aspect = "auto")
        ax1.set_xticks(poi_ticks_breaks, minor = True)
        ax1.set_yticks(qoi_ticks_breaks, minor = True)
        ax1.set_yticks(qoi_ticks_labels, minor = False)
        ax1.set_yticklabels(qoi_labels, fontsize = 10)
        ax1.set_xticks(poi_ticks_labels, minor = False)
        if rotate_y:
            ax1.set_xticklabels(poi_labels, rotation = "-90", fontsize = 10)
        else :
            ax1.set_xticklabels(poi_labels, fontsize = 10)
    elif orient.lower() == "vertical":
        fig, ax1 = plt.subplots(figsize=(3,7.2), ncols=1)
        pos = ax1.imshow(morris_indices, aspect = "auto")
        ax1.set_xticks(qoi_ticks_breaks, minor = True)
        ax1.set_yticks(poi_ticks_breaks, minor = True)
        ax1.set_yticks(poi_ticks_labels, minor = False)
        ax1.set_yticklabels(poi_labels, fontsize = 10)
        ax1.set_xticks(qoi_ticks_labels, minor = False)
        if rotate_y:
            ax1.set_xticklabels(qoi_labels, rotation = "-90", fontsize = 10)
        else :
            ax1.set_xticklabels(qoi_labels, fontsize = 10)
        
    
    ax1.set_title(title_label, pad =15, fontsize = 12)
    ax1.grid(axis = 'both', which = 'minor', linewidth = 2, color = 'r')
    cbar = fig.colorbar(pos, ax=ax1)
    cbar.set_label(color_bar_label, rotation = 270, labelpad = 20, fontsize = 12)
    plt.tight_layout()
    plt.savefig(plot_folder + save_name, bbox_inches='tight', pad_inches = 0)

if __name__ == "__main__":
    sys.exit(main())
