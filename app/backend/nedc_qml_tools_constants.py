#!/usr/bin/env python

# file: $(NEDC_NFC)/class/python/nedc_qml_tools/nedc_qml_tools_constants.py
#
# revision history: 
#
# 20250203 (SP): initial version
#
# This file contains all the constants that are used in the nedc_qml_tools
# module.  
#  
#------------------------------------------------------------------------------

# import reqired system modules
#
import os

# import required NEDC modules
#
import nedc_debug_tools as ndt

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define variables to handle option names and values. For each of these,
# we list the parameter name, the allowed values, and the default values.
#


# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()

#------------------------------------------------------------------------------
#
# constants listed here
#
#------------------------------------------------------------------------------

# define the provider names 
#
PROVIDER_NAME_QISKIT = 'qiskit'
PROVIDER_NAME_DWAVE = 'dwave'

# define the encoder (a.k.a FeatureMaps) names 
#
ENCODER_NAME_ZZ = 'zz'
ENCODER_NAME_Z = 'z'
ENCODER_NAME_QRBM_FEATURE_ENCODER = 'bqm'

# define the ansatz names 
#
ANSATZ_NAME_REAL_AMPLITUDES = 'real_amplitudes'

# define the kernel names 
#
KERNEL_NAME_FIDELITY = 'fidelity'

# define the model names 
#
MODEL_NAME_QSVM = 'qsvm'
DEFAULT_MODEL_NAME = MODEL_NAME_QSVM
MODEL_NAME_QNN = 'qnn'
MODEL_NAME_QRBM = 'qrbm'

# define the optimizer names
#
OPTIM_NAME_COBYLA = 'cobyla'

# define measurement types
MEASUREMENT_TYPE_ESTIMATOR = 'estimator'
MEASUREMENT_TYPE_SAMPLER = 'sampler'


# define the hardware names
#
HARDWARE_NAME_CPU = 'cpu'
HARDWARE_NAME_GPU = 'gpu'
HARDWARE_NAME_QPU = 'qpu'

# define the default values for the parameters
#
DEFAULT_PROVIDER_NAME = PROVIDER_NAME_QISKIT
DEFAULT_ENCODER_NAME = ENCODER_NAME_ZZ
DEFAULT_MODEL_NAME = MODEL_NAME_QSVM
DEFAULT_HARDWARE_NAME = HARDWARE_NAME_CPU
DEFAULT_N_QUBITS = 4
DEFAULT_SHOTS = 1000
DEFAULT_NOISE_MODEL = None
DEFAULT_REPS = 2
DEFAULT_ENTANGLEMENT = 'full'
DEFAULT_NONE_VALUE = None
DEFAULT_OPTIM_MAX_STEPS = 50
DEFAULT_MESUREMENT_TYPE = MEASUREMENT_TYPE_SAMPLER
DEFAULT_NUM_CLASSES = 2
DEFAULT_EPOCHS = 10
DEFAULT_LEARNING_RATE = 0.01

# define the default values for the parameters for QRBM
#
DEFAULT_N_HIDDEN = 4
DEFAULT_N_VISIBLE = 2
DEFAULT_CHAIN_STRENGTH = 2
DEFAULT_KNN_N_NEIGHBORS = 2

# define miscellaneous constants
#
SVM_KERNEL_PRECOMPUTED = 'precomputed'
PERCENTAGE = 100
QUBO_VAR_TYPE = 'BINARY'

#
# end of constants module
#------------------------------------------------------------------------------