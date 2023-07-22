#!/bin/sh

# Run the following commands to
# sudo find / -name 'nvcc'  # Path to binaries
# sudo find / -name 'libcublas.so.*'  # Path to libraries

# disable pre-allocation of memory for JAX
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# add the hsa_planar_control folder to the PYTHONPATH
export PYTHONPATH="${PYTHONPATH}://hsa_planar_control"
