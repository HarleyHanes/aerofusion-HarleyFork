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
  point_A = np.array(mesh_options["geometry"]["origin"])
  point_B_prime = point_A + [mesh_options["geometry"]["length_AB"], 0,0]
  point_C1_prime = point_B_prime + [-mesh_options["geometry"]["length_BC_h"], \
                     mesh_options["geometry"]["length_BC_v"] ,0]
  point_C2_prime = point_B_prime + [-mesh_options["geometry"]["length_BC_h"], \
                     - mesh_options["geometry"]["length_BC_v"] ,0]
  point_D1_prime = point_B_prime + [-mesh_options["geometry"]["length_BD_h"], \
                     mesh_options["geometry"]["length_BD_v"] ,0]
  point_D2_prime = point_B_prime + [-mesh_options["geometry"]["length_BD_h"], \
                     - mesh_options["geometry"]["length_BD_v"] ,0]
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
  point_B[0] = point_B_prime[0]*np.cos(alpha) - \
                 point_B_prime[1]*np.sin(alpha)
  point_B[1] = point_B_prime[0]*np.sin(alpha) + \
                 point_B_prime[1]*np.cos(alpha)
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

 # creating dictionary of all the  8 points to store all the point in a sequence
#  probe_points = {"p1" : point_A, "p2": point_E1, "p3": point_D1, \
#                   "p4": point_C1, "p5": point_B, "p6": point_C2, \
#                     "p7": point_D2, "p8": point_E2} 

  num_control_point = 8
  num_dim = 3
  probe_points = np.zeros([num_control_point, num_dim])
  probe_points[0,:] = point_A
  probe_points[1,:] = point_E1
  probe_points[2,:] = point_D1
  probe_points[3,:] = point_C1
  probe_points[4,:] = point_B
  probe_points[5,:] = point_C2
  probe_points[6,:] = point_D2
  probe_points[7,:] = point_E2
 
  # Write output file
  file = open("orion_geometry.txt", "w")
  file.write("blobk"+ "  "+ "pintID"+ "  "+ "x" +"  "+ "y" + "  "+ "z"+ "\n")
  for i_point in range(num_control_point):
      file.write(str(1) + "  " + str(i_point+1) + "  " +
                  str(probe_points[i_point, 0])+ "  " + \
                   str(probe_points[i_point, 1])+ "  " + \
                     str(probe_points[i_point,2])+ "\n")


#------------------------------------------------------------------------------
if __name__ == "__main__":
  sys.exit(main())
