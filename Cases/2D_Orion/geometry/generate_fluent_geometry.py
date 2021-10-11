# Copyright (c) 2021 by the aerofusion software authors. All rights reserved.

# -----------------------------------------------------------------------------
# \file generate_fluent_geometry.py
# \brief Generate the point data in Fluent format (txt file) with the geometry
# of the Orion crew module in 2D
#
# Requires a libconfig (.cfg) file with the geometric settings
# Output a series of points that can be imported in Fluent
# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------

import os
import sys
import libconf
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import logging
import re

# -----------------------------------------------------------------------------

def get_number_of_lines_in_file(filename):
  with open(filename) as file_handle:
    for idx, line in enumerate(file_handle):
      pass
  return idx + 1

# -----------------------------------------------------------------------------

# Handle environment variables for io open
def io_open_env_var(file_name_raw, rw_flags="r"):
  import io
  import os
  file_name_parsed = os.path.expandvars(file_name_raw)
  fh = io.open(file_name_parsed, rw_flags, encoding="utf-8")
  return fh

# -----------------------------------------------------------------------------
def find_center_circumference_from_two_points_and_radius(\
    point_P, point_Q, radius):
  x1 = point_P[0]
  y1 = point_P[1]
  x2 = point_Q[0]
  y2 = point_Q[1]
  radsq = radius * radius;
  q = np.sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)));
  x3 = (x1 + x2) / 2;
  y3 = (y1 + y2) / 2;
  return np.array(\
    [x3 + np.sqrt(radsq - ((q / 2) * (q / 2))) * ((y1 - y2) / q),
     y3 + np.sqrt(radsq - ((q / 2) * (q / 2))) * ((x2 - x1) / q),
     0.0])

# -----------------------------------------------------------------------------
def main(argv=None):

  if argv is None:
    argv = sys.argv[1:]

  if type(argv) == libconf.AttrDict:
    mesh_options = argv
  else:
    # Parse arguments and options -----------------------------------------
    usage_text = (
      "Usage: \n"
      "  python3 " + __file__ + "\n" +
      "Arguments:" + \
      "\n  input_filename[1]:" + \
      "\n  input_filename_override[2]:" + \
      "\n    File with libconfig options." \
      )
    parser = argparse.ArgumentParser(description = usage_text)
    parser.add_argument("input_filename",
                        help="Name of input file")
    parser.add_argument("--override",
                        help="overrides settings from input_filename",
                        default=None)
    parser.add_argument("-d", "--debug",
      action="store_true", dest="debug")
    parser.add_argument("-v", "--verbose",
      action="store_true", dest="verbose")
    parser.add_argument("-q", "--quiet",
      action="store_false", dest="verbose")

    args = parser.parse_args(argv)

    # Read options from input (libconfig) file
    with io_open_env_var(args.input_filename) as libconfig_file:
      options = libconf.load(libconfig_file)

    # override settings (if applicable)
    if args.override:
      with io_open_env_var(args.override) as libconfig_file:
        override_options = libconf.load(libconfig_file)
      # merge the overide settings into the main settings fail
      dict_merge(options, override_options)
      # now write back the merged result to the output_dir
      merged_cfg_filename = options["output_directory"] + \
                            "preprocess_merged.cfg"
      with io_open_env_var(merged_cfg_filename, "w+") as fh:
        libconf.dump(options, fh)

    # eval env variables
    options.output_directory = \
      str(os.path.expandvars(options["output_directory"]))

    if "logging_level" in options.keys():
      if options.logging_level == "DEBUG":
        logging.basicConfig(level=logging.DEBUG)

    if "mesh" in options:
      mesh_options = options.mesh
    else:
      mesh_options = options

  # Execute code --------------------------------------------------------------

  print("Options", mesh_options)
  radius_BC = mesh_options["geometry"]["radius_BC"]
  radius_CD = mesh_options["geometry"]["radius_CD"]
  angle_BD  = mesh_options["geometry"]["angle_BD"]
  #angle_CD  = mesh_options["geometry"]["angle_CD"]
  point_A = np.array(mesh_options["geometry"]["origin"])
  length_BC_h = mesh_options["geometry"]["length_BC_h"]
  length_BC_v = mesh_options["geometry"]["length_BC_v"]
  length_BD_h = mesh_options["geometry"]["length_BD_h"]
  length_BD_v = mesh_options["geometry"]["length_BD_v"]
  point_B_prime = point_A + [mesh_options["geometry"]["length_AB"], 0,0]
  point_C1_prime = point_B_prime + [-length_BC_h,  length_BC_v ,0]
  print("point_C1_prime", point_C1_prime)
  point_C2_prime = point_B_prime + [-length_BC_h, -length_BC_v, 0]
  point_D1_prime = point_B_prime + [-length_BD_h,  length_BD_v, 0]
  point_D2_prime = point_B_prime + [-length_BD_h, -length_BD_v, 0]
  point_E1_prime = point_A + [0, mesh_options["geometry"]["length_AE"],0]
  point_E2_prime = point_A + [0, -mesh_options["geometry"]["length_AE"],0]

  alpha = mesh_options["geometry"]["angle_of_attack"] * np.pi/180
  point_B = np.zeros([3])
  point_C1 = np.zeros([3])
  point_C2 = np.zeros([3])
  point_D1 = np.zeros([3])
  point_D2 = np.zeros([3])
  point_E1 = np.zeros([3])
  point_E2 = np.zeros([3])
  point_B[0]  = point_B_prime[0]*np.cos(alpha) - \
                point_B_prime[1]*np.sin(alpha)
  point_B[1]  = point_B_prime[0]*np.sin(alpha) + \
                point_B_prime[1]*np.cos(alpha)
  print("point_B", point_B, "vs", point_B_prime)
  point_C1[0] = point_C1_prime[0]*np.cos(alpha) - \
                point_C1_prime[1]*np.sin(alpha)
  point_C1[1] = point_C1_prime[0]*np.sin(alpha) + \
                point_C1_prime[1]*np.cos(alpha)
  point_C2[0] = point_C2_prime[0]*np.cos(alpha) - \
                point_C2_prime[1]*np.sin(alpha)
  point_C2[1] = point_C2_prime[0]*np.sin(alpha) + \
                point_C2_prime[1]*np.cos(alpha)
  point_D1[0] = point_D1_prime[0]*np.cos(alpha) - \
                point_D1_prime[1]*np.sin(alpha)
  point_D1[1] = point_D1_prime[0]*np.sin(alpha) + \
                point_D1_prime[1]*np.cos(alpha)
  point_D2[0] = point_D2_prime[0]*np.cos(alpha) - \
                point_D2_prime[1]*np.sin(alpha)
  point_D2[1] = point_D2_prime[0]*np.sin(alpha) + \
                point_D2_prime[1]*np.cos(alpha)
  point_E1[0] = point_E1_prime[0]*np.cos(alpha) - \
                point_E1_prime[1]*np.sin(alpha)
  point_E1[1] = point_E1_prime[0]*np.sin(alpha) + \
                point_E1_prime[1]*np.cos(alpha)
  point_E2[0] = point_E2_prime[0]*np.cos(alpha) - \
                point_E2_prime[1]*np.sin(alpha)
  point_E2[1] = point_E2_prime[0]*np.sin(alpha) + \
                point_E2_prime[1]*np.cos(alpha)

  # arc BC
  number_of_points_BC = mesh_options["geometry"]["number_of_points_BC"]
  angle_BC = np.arctan(point_C1[1]/(radius_BC - length_BC_h))
  print("DEBUG point_C1[1]", point_C1[1], "length_BC_h", length_BC_h)
  origin_BC = point_B - np.array([radius_BC,0.0,0.0])
  print("DEBUG point_C1_test_x",
        origin_BC + np.cos(angle_BC), "vs", point_C1[0])
  # Due to a small mismatch between:
  #  + point_C2 and the beginning of the arc C2-C1
  #  + point_C1 and the end of the arc C2-C1
  # we construct the arc without those beginning and end points and we append
  # the actual C2 and C1 points instead
  # C2-B
  points_C2_C1 = [point_C2]
  for idx in range(number_of_points_BC):
    beta_BQ = - angle_BC * (number_of_points_BC - idx - 1) / (number_of_points_BC)
    point_Q = origin_BC + \
              radius_BC * np.array([np.cos(beta_BQ), np.sin(beta_BQ), 0.0])
    print("beta_BQ", beta_BQ, "point_Q", point_Q, "radius_BC", radius_BC,
          "angle_BC", angle_BC)
    points_C2_C1.append(point_Q)
  # B-C1
  for idx in range(number_of_points_BC):
    beta_BQ = angle_BC * idx / number_of_points_BC
    point_Q = origin_BC + \
              radius_BC * np.array([np.cos(beta_BQ), np.sin(beta_BQ), 0.0])
    print("beta_BQ", beta_BQ, "point_Q", point_Q, "radius_BC", radius_BC,
          "angle_BC", angle_BC)
    points_C2_C1.append(point_Q)
  points_C2_C1.append(point_C1)

  # arc C1-D1
  center_CD = find_center_circumference_from_two_points_and_radius(\
    point_C1, point_D1, radius_CD)
  print("center_CD", center_CD)
  number_of_points_CD = mesh_options["geometry"]["number_of_points_CD"]
  angle_center_CD_C1 = \
    np.arctan2((point_C1[1]-center_CD[1]), (point_C1[0]-center_CD[0]))
  angle_center_CD_D1 = \
    np.arctan2((point_D1[1]-center_CD[1]), (point_D1[0]-center_CD[0]) )
  angle_CD = angle_center_CD_D1 - angle_center_CD_C1
  print("(point_C1[0]-center_CD[0])", (point_C1[0]-center_CD[0]))
  print("(point_C1[1]-center_CD[1])", (point_C1[1]-center_CD[1]))
  print("(point_D1[0]-center_CD[0])", (point_D1[0]-center_CD[0]))
  print("(point_D1[1]-center_CD[1])", (point_D1[1]-center_CD[1]))
  print("angle_center_CD_C1", np.rad2deg(angle_center_CD_C1))
  print("angle_center_CD_D1", np.rad2deg(angle_center_CD_D1))
  print("angle_CD", np.rad2deg(angle_CD))
  points_C1_D1 = []
  for idx in range(number_of_points_CD):
    beta_BQ = angle_center_CD_C1 + \
              angle_CD * idx / (number_of_points_CD-1)
    point_Q = center_CD + \
              radius_CD * np.array([np.cos(beta_BQ), np.sin(beta_BQ), 0.0])
    print("beta_BQ", np.rad2deg(beta_BQ))
    points_C1_D1.append(point_Q)
 # points_C1_D1.append(center_CD)

  # arc C2-D2
  center_CD[1] = -center_CD[1]
  print("center_CD", center_CD)
  number_of_points_CD = mesh_options["geometry"]["number_of_points_CD"]
  angle_center_CD_C2 = \
    np.arctan2((point_C2[1]-center_CD[1]), (point_C2[0]-center_CD[0]))
  angle_center_CD_D2 = \
    np.arctan2((point_D2[1]-center_CD[1]), (point_D2[0]-center_CD[0]) )
  angle_CD2 = angle_center_CD_D2 - angle_center_CD_C2
 # print("(point_C1[0]-center_CD[0])", (point_C1[0]-center_CD[0]))
 # print("(point_C1[1]-center_CD[1])", (point_C1[1]-center_CD[1]))
 # print("(point_D1[0]-center_CD[0])", (point_D1[0]-center_CD[0]))
 # print("(point_D1[1]-center_CD[1])", (point_D1[1]-center_CD[1]))
 # print("angle_center_CD_C1", np.rad2deg(angle_center_CD_C1))
 # print("angle_center_CD_D1", np.rad2deg(angle_center_CD_D1))
 # print("angle_CD", np.rad2deg(angle_CD))
  points_C2_D2 = []
  for idx in range(number_of_points_CD):
    beta_BQ = angle_center_CD_C2 - \
              angle_CD * idx / (number_of_points_CD-1)
    point_Q = center_CD + \
              radius_CD * np.array([np.cos(beta_BQ), np.sin(beta_BQ), 0.0])
    print("beta_BQ", np.rad2deg(beta_BQ))
    points_C2_D2.append(point_Q)
 
# line E-D
  number_of_points_DE = mesh_options["geometry"]["number_of_points_DE"]
  angel_ED = np.arctan((point_D1[1] - point_E1[1])/(point_D1[0] - point_E1[0]))
  length_E_D = np.sqrt((point_D1[1] - point_E1[1])**2 + \
                 (point_D1[0]-point_E1[0])**2)
  points_D1_E1 = []
  points_D2_E2 = []
  for idx in range(number_of_points_DE):    
      length_D_Q = idx * length_E_D /(number_of_points_DE - 1)
      point_Q =  np.array([point_D1[0], point_D1[1], 0.0]) - \
                   length_D_Q *\
                     np.array([np.cos(angel_ED), np.sin(angel_ED), 0.0]) 
      points_D1_E1.append(point_Q)
      points_D2_E2.append(np.array([point_Q[0], -point_Q[1], 0.0]))


# line EA
  number_of_points_EA = mesh_options["geometry"]["number_of_points_EA"]
  length_EA = mesh_options["geometry"]["length_AE"]
  points_E1_A = []
  points_E2_A = []
  for idx in range(number_of_points_EA):   
      length_EQ = idx * length_EA /(number_of_points_EA - 1)
      point_Q =  point_E1 - np.array([0.0, length_EQ, 0.0]) 
      points_E1_A.append(point_Q)
      points_E2_A.append(np.array([point_Q[0], -point_Q[1], 0.0]))
  import ipdb 
  ipdb.set_trace()

  probe_points = []
  probe_points.append(point_A)
  probe_points.append(point_E1)
  probe_points.append(point_D1)
  probe_points.append(point_C1)
  probe_points.append(point_B)
  probe_points.append(point_C2)
  probe_points.append(point_D2)
  probe_points.append(point_E2)
 
  # Write output file
  file = open(mesh_options["output"]["point_file"], "w")
  file.write("block"+ "  "+ "pointID"+ "  "+ "x" +"  "+ "y" + "  "+ "z"+ "\n")
  for pdx, point in enumerate(probe_points):
      file.write(str(1) + "  " + str(pdx+1) + "  " +
                 str(point[0])+ "  " + \
                 str(point[1])+ "  " + \
                 str(point[2])+ "\n")
  
  file.write(str(1) + "  " + str(len(probe_points)+1) + "  " +
                 str(point_A[0])+ "  " + \
                 str(point_A[1])+ "  " + \
                 str(point_A[2])+ "\n")
  file.close()

  print("DEBUG points_C2_C1", points_C2_C1)
  file = open(mesh_options["output"]["arc_C1C2_file"], "w")
  file.write("block"+ "  "+ "pointID"+ "  "+ "x" +"  "+ "y" + "  "+ "z"+ "\n")
  for pdx, point in enumerate(points_C2_C1):
      file.write(str(1) + "  " + str(pdx+1) + "  " +
                 str(point[0])+ "  " + \
                 str(point[1])+ "  " + \
                 str(point[2])+ "\n")
  file.close()

  file = open(mesh_options["output"]["arc_CD1_file"], "w")
  file.write("block"+ "  "+ "pointID"+ "  "+ "x" +"  "+ "y" + "  "+ "z"+ "\n")
  for pdx, point in enumerate(points_C1_D1):
      file.write(str(1) + "  " + str(pdx+1) + "  " +
                 str(point[0])+ "  " + \
                 str(point[1])+ "  " + \
                 str(point[2])+ "\n")
  file.close()
  
  file = open(mesh_options["output"]["arc_CD2_file"], "w")
  file.write("block"+ "  "+ "pointID"+ "  "+ "x" +"  "+ "y" + "  "+ "z"+ "\n")
  for pdx, point in enumerate(points_C2_D2):
      file.write(str(1) + "  " + str(pdx+1) + "  " +
                 str(point[0])+ "  " + \
                 str(point[1])+ "  " + \
                 str(point[2])+ "\n")
  file.close()
  
  # Data for plotting
  #output_filename_plot = mesh_options["output"]["plot_file"]
  #fig, ax = plt.subplots()
  #ax.plot(probe_points)
  #ax.set(xlabel='x', ylabel='y',
  #       title='Title')
  #ax.grid()
  #fig.savefig(output_filename_plot)
  #plt.show()
  

#------------------------------------------------------------------------------
if __name__ == "__main__":
  sys.exit(main())
