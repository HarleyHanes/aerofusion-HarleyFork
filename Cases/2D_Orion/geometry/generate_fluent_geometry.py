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
  print("point_A", point_A)

  # Define coordinates of points (A,B,C,...)
  


  # Write output file

#------------------------------------------------------------------------------
if __name__ == "__main__":
  sys.exit(main())
