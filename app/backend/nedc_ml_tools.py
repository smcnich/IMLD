#!/usr/bin/env python
#
# file: $NEDC_NFC/class/python/nedc_ml_tools/nedc_ml_tools.py
#
# revision history:
#
# 20250322 (SP): fixed model_d format issue
# 20250322 (JP): fixed error messages
# 20250321 (PM): fixed RBM
# 20250318 (JP): more refactoring
# 20250304 (JP): reviewed and refactored
# 20250228 (AM): added GMM
# 20250226 (SP): added QSVM, QNN and QRBM classes
# 20250222 (PM): Moved MLToolsData class to its own file
# 20250107 (SP): added TRANSFORMER class
# 20240821 (DB): fixed an interface issue with scoring
# 20240120 (SM): added/fixed confusion matrix and accuracy scores
# 20240105 (PM): added new MLToolsData class and Euclidean Alg
# 20230623 (AB): code refactored to new comment style
# 20230515 (PM): reviewed and refactored
# 20230316 (JP): reviewed again
# 20230115 (JP): reviewed and refactored (again)
# 20230114 (PM): completed the implementation
# 20230110 (JP): reviewed and refactored
# 20221220 (PM): initial version
#
#
# This class contains a collection of classes and methods that consolidate
# some of our most common machine learning algorithms. Currently, we support:
#
#  Discriminant-Based Algorithms:
#   Euclidean Distance (EUCLIDEAN)
#   Principle Component Analysis (PCA)
#   Linear Discriminate Analysis (LDA)
#   Quadratic Discriminant Analysis (QDA)
#   Quadratic Linear Discriminate Analysis (QLDA)
#   Naive Bayes (NB)
#   Gaussian Mixture Models (GMM)
#
#  Nonparametric Models:
#   K-Nearest Neighbor (KNN)
#   KMEANS Clustering (KMEANS)
#   Random Forests (RNF)
#   Support Vector Machines (SVM)
#
#  Neural Network Models:
#   Multilayer Perceptron (MLP)
#   Restricted Boltzmann Machine (RBM)
#   Transformer (TRANSFORMER)
#
#  Quantum Computing-Based Models:
#   Quantum Support Vector Machines (QSVM)
#   Quantum Neural Network (QNN)
#   Quantum Restricted Boltzmann Machine (QRBM)
#
# in this class. These are accessed through a wrapper class called Algorithm.
#
# The implementations that are included here are a mixture of things found
# in the statistical package JMP, the Python machine learning library
# scikit-learn and the ISIP machine learning demo, IMLD.
#
# Example Usage:
#  This is an example of how to use this class to load and classify data
#   alg = Alg()
#   alg.set(LDA_NAME)
#   alg.load_parameters("params_v00.toml")
#   data = MLToolsData("data.csv")
#   score, model = alg.train(data)
#   labels, post = alg.predict(data)
#
#------------------------------------------------------------------------------

# import required system modules (basic)
#
from collections import defaultdict
import datetime as dt
import numpy as np
import pickle
import os
import sys

# import required system modules (machine learning)
#
from imblearn.metrics import sensitivity_score, specificity_score

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, silhouette_score)
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import (MLPClassifier, BernoulliRBM)
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import torch

# import required NEDC modules
#
import nedc_cov_tools as nct
import nedc_debug_tools as ndt
import nedc_file_tools as nft
from nedc_ml_tools_data import MLToolsData
import nedc_qml_tools as nqt
import nedc_trans_tools as ntt

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define names that appear in all algorithms (listed alphabetically)
#
ALG_NAME_ALG = "algorithm_name"
ALG_NAME_IMP = "implementation_name"
ALG_NAME_MDL = "model"

# define all the unique implementation names (listed alphabetically)
#
IMP_NAME_DISCRIMINANT = "discriminant"
IMP_NAME_DWAVE = "dwave"
IMP_NAME_EM = "em"
IMP_NAME_PYTORCH = "pytorch"
IMP_NAME_QISKIT = "qiskit"
IMP_NAME_SKLEARN = "sklearn"

# define common names that appear in many, but not all,
# algorithms (listed alphabetically)
#
ALG_NAME_ACT = "activation"
ALG_NAME_BSIZE = "batch_size"
ALG_NAME_C = "c"
ALG_NAME_CLABELS = "class_labels"
ALG_NAME_CLASSF = "classifier"
ALG_NAME_CMODELS = "class_models"
ALG_NAME_COV = "covariance"
ALG_NAME_CRITERION = "criterion"
ALG_NAME_DSCR = "discriminant"
ALG_NAME_EIGV = "eigenvalue"
ALG_NAME_ESTOP = "early_stopping"
ALG_NAME_GAMMA = "gamma"
ALG_NAME_HDW = "hardware"
ALG_NAME_HSIZE = "hidden_size"
ALG_NAME_KERNEL = "kernel"
ALG_NAME_LR = "learning_rate"
ALG_NAME_LRINIT = "learning_rate_init"
ALG_NAME_MAXDEPTH = 'max_depth'
ALG_NAME_MAXITER = "max_iters"
ALG_NAME_MEANS = "means"
ALG_NAME_MOMENTUM = "momentum"
ALG_NAME_NCLUSTERS = "n_clusters"
ALG_NAME_NCMP = "n_components"
ALG_NAME_NEST = "n_estimators"
ALG_NAME_NINIT = "n_init"
ALG_NAME_NLAYERS = "num_layers"
ALG_NAME_NNEARN = "k_nearest_neighbors"
ALG_NAME_PRIORS = "priors"
ALG_NAME_PRIORS_MAP = "map"
ALG_NAME_PRIORS_ML = "ml"
ALG_NAME_PROVIDER = "provider_name"
ALG_NAME_RANDOM = "random_state"
ALG_NAME_RANDSTATE = "random_state"
ALG_NAME_SHUFFLE = "shuffle"
ALG_NAME_SOLVER = "solver"
ALG_NAME_TRANS = "transform"
ALG_NAME_VALF = "validation_fraction"
ALG_NAME_WEIGHTS = "weights"

# define formats for generating a scoring report (listed alphabetically)
#
ALG_FMT_DEC = ".4f"
ALG_FMT_DTE = "Date: %s%s"
ALG_FMT_ERR = "%012s %10.2f%%"
ALG_FMT_LBL = "%06d"
ALG_FMT_PCT = float(100.0)
ALG_FMT_WCL = "%6d"
ALG_FMT_WLB = "%10s"
ALG_FMT_WPC = "%6.2f"
ALG_FMT_WST = "%6d (%6.2f%%)"

# define names common to all quantum algorithms (listed alphabetically)
#
QML_NAME_ANSATZ = "ansatz_name"
QML_NAME_ANSATZ_REPS = "ansatz_reps"
QML_NAME_CS = "chain_strength"
QML_NAME_DROPOUT = "dropout"
QML_NAME_EMBED_SIZE = "embed_size"
QML_NAME_ENCODER = "encoder_name"
QML_NAME_ENTANGLEMENT = "entanglement"
QML_NAME_EPOCH = "epoch"
QML_NAME_FEAT_REPS = "featuremap_reps"
QML_NAME_HARDWARE = "hardware"
QML_NAME_MAXSTEPS = "optim_max_steps"
QML_NAME_MEAS_TYPE = "meas_type"
QML_NAME_MLP_DIM = "mlp_dim"
QML_NAME_NHEADS = "nheads"
QML_NAME_NHIDDEN = "n_hidden"
QML_NAME_NQUBITS = "n_qubits"
QML_NAME_OPTIM = "optim_name"
QML_NAME_PROVIDER = "provider_name"
QML_NAME_SHOTS = "shots"

# declare global debug and verbosity objects so we can use them
# in both functions and classes
#
dbgl_g = ndt.Dbgl()
vrbl_g = ndt.Vrbl()

#******************************************************************************
#
# Algorithm-Specific Parameter Definitions
#
#******************************************************************************

#------------------------------------------------------------------------------
# Alg = EUCLIDEAN: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
EUCL_NAME = "EUCLIDEAN"
EUCL_IMPLS = [IMP_NAME_DISCRIMINANT]

# the parameter block (param_d) for EUCLIDEAN looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'discriminant',
#   'weights': list
#   }
#  })
#
# the model (model_d) for EUCLIDEAN contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'EUCLIDEAN',
#   'implementation_name': 'discriminant',
#   'model': {
#    'means': list
#    'weights': list
#   }
#  })
#

#------------------------------------------------------------------------------
# Alg = PCA: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name
#
PCA_NAME = "PCA"
PCA_IMPLS = [IMP_NAME_DISCRIMINANT]

# the parameter block (param_d) for PCA looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'discriminant',
#   'priors': 'ml',
#   'ctype': 'full',
#   'center': 'none',
#   'scale': 'biased'
#   'n_components': -1 (all)
#   }
#  })
#
# the model (model_d) for PCA contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'PCA',
#   'implementation_name': 'discriminant',
#   'model': {
#    'priors': numpy array,
#    'ctype': 'full',
#    'center': 'none',
#    'scale': 'biased',
#    'means': list,
#    'transform': matrix
#   }
#  })
#

#------------------------------------------------------------------------------
# Alg = LDA: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
LDA_NAME = "LDA"
LDA_IMPLS = [IMP_NAME_DISCRIMINANT]

# the parameter block (param_d) for LDA looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'discriminant',
#   'priors': 'ml',
#   'ctype': 'full',
#   'center': 'none',
#   'scale': 'none'
#   'n_components': -1 (all)
#   }
#  })
#
# the model (model_d) for LDA contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'LDA',
#   'implementation_name': 'discriminant',
#   'model': {
#    'priors': numpy array,
#    'ctype': 'full',
#    'center': 'none',
#    'scale': 'biased',
#    'means': list,
#    'transform': matrix
#   }
#  })
#

#------------------------------------------------------------------------------
# Alg = QDA: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
QDA_NAME = "QDA"
QDA_IMPLS = [IMP_NAME_DISCRIMINANT]

# the parameter block (param_d) for QDA looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'discriminant',
#   'priors': 'ml',
#   'ctype': 'full',
#   'center': 'none',
#   'scale': 'biased'
#   'n_components': -1 (all)
#   }
#  })
#
# the model (model_d) for QDA contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'QDA',
#   'implementation_name': 'discriminant',
#   'model': {
#    'priors': numpy array,
#    'ctype': 'full',
#    'center': 'none',
#    'scale': 'biased',
#    'means': list,
#    'transform': matrix
#   }
#  })
#

#------------------------------------------------------------------------------
# Alg = QLDA: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
QLDA_NAME = "QLDA"
QLDA_IMPLS = [IMP_NAME_DISCRIMINANT]

# the parameter block (param_d) for QLDA contains:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'discriminant',
#   'priors': 'ml',
#   'ctype': 'full',
#   'center': 'none',
#   'scale': 'none'
#   'n_components': -1 (all)
#   }
#  })
#
# the model (model_d) for QLDA contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'QLDA',
#   'implementation_name': 'discriminant',
#   'model': {
#    'priors': numpy array,
#    'ctype': 'full',
#    'center': 'none',
#    'scale': 'none'
#    'means': list,
#    'transform': matrix
#   }
#  })
#

#------------------------------------------------------------------------------
# Alg = NB: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
NB_NAME = "NB"
NB_IMPLS = [IMP_NAME_SKLEARN]

# the parameter block (param_d) for NB looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'sklearn',
#   'priors': 'ml',
#   'average': 'none',
#   'multi_class': 'ovr'
#   }
#  })
#
# the model (model_d) for NB contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'NB',
#   'implementation_name': 'sklearn',
#   'model': Sklearn.NB.Model
#  })
#

#------------------------------------------------------------------------------
# Alg = GMM: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
GMM_NAME = "GMM"
GMM_IMPLS = ["em"]

# the parameter block (param_d) for GMM looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'em',
#   'priors': 'ml',
#   'n_components': -1 (all),
#   'random_state': 27
#   }
#  })
#
# the model (model_d) for GMM contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'GMM',
#   'implementation_name': 'em',
#   'model': {
#    'priors': numpy array,
#    'class_models': dict,
#    'class_labels': numpy array
#   }
#  })
#

#------------------------------------------------------------------------------
# Alg = KNN: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
KNN_NAME = "KNN"
KNN_IMPLS = [IMP_NAME_SKLEARN]

# the parameter block (param_d) for KNN looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'sklearn',
#   'k_nearest_neighbors': 5
#   }
#  })
#
# the model (model_d) for KNN contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'KNN',
#   'implementation_name': 'sklearn',
#   'model': Sklearn KNN Model
#  })
#

#------------------------------------------------------------------------------
# Alg = KMEANS: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
KMEANS_NAME = "KMEANS"
KMEANS_IMPLS = [IMP_NAME_SKLEARN]

# the parameter block (param_d) for KMEANS looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'sklearn',
#   'n_clusters': 2,
#   'n_init': 3,
#   'random_state': 27,
#   'max_iters': 100
#   }
#  })
#
# the model (model_d) for KMEANS contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'KMEANS',
#   'implementation_name': 'sklearn',
#   'model': Sklearn KMEANS Model
#  })
#

#------------------------------------------------------------------------------
# Alg = RNF: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
RNF_NAME = "RNF"
RNF_IMPLS = [IMP_NAME_SKLEARN]

# the parameter block (param_d) for RNF looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'sklearn',
#   'n_estimators': 100,
#   'max_depth': 5,
#   'criterion': 'gini'
#   'random_state': 27
#   }
#  })
#
# the model (model_d) for RNF contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'RNF',
#   'implementation_name': 'sklearn',
#   'model': Sklearn RNF Model
#  })
#

#------------------------------------------------------------------------------
# Alg = SVM: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
SVM_NAME = "SVM"
SVM_IMPLS = [IMP_NAME_SKLEARN]

# the parameter block (param_d) for SVM looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'sklearn',
#   'c': 1,
#   'gamma': 0.1,
#   'kernel': 'linear'
#   }
#  })
#
# the model (model_d) for SVM contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'SVM',
#   'implementation_name': 'sklearn',
#   'model': Sklearn SVM Model
#  })
#

#------------------------------------------------------------------------------
# Alg = MLP: define dictionary keys for parameters
#------------------------------------------------------------------------------

# define the algorithm name and the available implementations
#
MLP_NAME = "MLP"
MLP_IMPLS = [IMP_NAME_SKLEARN]

# the parameter block (param_d) for MLP looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'sklearn',
#   'hidden_size': 3,
#   'activation': 'relu',
#   'solver': 'adam',
#   'batch_size: 'auto',
#   'learning_rate: 'constant',
#   'learning_rate_init: 0.001,
#   'random_state': 27,
#   'momentum': 0.9
#   'validation_fraction: 0.1,
#   'max_iters': 100,
#   'shuffle: true
#   'early_stopping: false,
#   }
#  })
#
# the model (model_d) for MLP contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'MLP',
#   'implementation_name': 'sklearn',
#   'model': Sklearn MLP Model
#  })
#

#------------------------------------------------------------------------------
# Alg = RBM: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
RBM_NAME = "RBM"
RBM_IMPLS = [IMP_NAME_SKLEARN]

RBM_NAME_NITER = "max_iters"
RBM_NAME_VERBOSE = "verbose"
RBM_NAME_CLASSIFIER = "classifier"

# the parameter block (param_d) for RBM looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'sklearn',
#   'classifier: 'KNN',
#   'k_nearest_neighbors = 2,
#   'learning_rate': 0.1,
#   'batch_size': 0,
#   'verbose': 0,
#   'random_state': none,
#   'n_components': 2,
#   'max_iters': 100,
#   }
#  })
#
# the model (model_d) for RBM contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'RBM',
#   'implementation_name': 'sklearn',
#   'model': Sklearn RBM Model
#  })
#

#------------------------------------------------------------------------------
# Alg = TRANSFORMER: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
TRANSF_NAME = "TRANSFORMER"
TRANSF_IMPLS = [IMP_NAME_PYTORCH]

# the parameter block (param_d) for TRANSFORMER looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'pytorch',
#   'epoch': 50,
#   'learning_rater': 0.001,
#   'batch_size': 32,
#   'embed_size': 32,
#   'nheads': 2,
#   'num_layers': 2,
#   'mlp_dim': 64,
#   'dropout': 0.1,
#   'random_state': 27
#   }
#  })

# the model (model_d) for TRANSFORMER contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'TRANSFORMER',
#   'implementation_name': 'pytorch',
#   'model': { Pytorch Transformer model's weights}
#  })
#

#------------------------------------------------------------------------------
# Alg = QSVM: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
QSVM_NAME = "QSVM"
QSVM_IMPLS = [IMP_NAME_QISKIT]

# the parameter block (param_d) for QSVM looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'qiskit',
#   'provider_name': 'qiskit',
#   'hardware': 'cpu',
#   'encoder_name': 'zz',
#   'n_qubits': 4,
#   'featuremap_reps': 2,
#   'entanglement': 'full',
#   'kernel': 'fidelity',
#   'shots': 1024
#   }
#  })
#
# the model (model_d) for QSVM contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'QSVM',
#   'implementation_name': 'qiskit',
#   'model': QSVM(provider=QiskitProvider)
#  })
#

#------------------------------------------------------------------------------
# Alg = QNN: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
QNN_NAME = "QNN"
QNN_IMPLS = [IMP_NAME_QISKIT]

# the parameter block (param_d) for QNN looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'qiskit',
#   'provider_name': 'qiskit',
#   'hardware': 'cpu',
#   'encoder_name': 'zz',
#   'n_qubits': 2,
#   'entanglement': 'full',
#   'featuremap_reps': 2,
#   'ansatz_reps': 2,
#   'ansatz_name': 'real_amplitudes',
#   'optim_name': 'cobyla',
#   'optim_max_steps': 50,
#   'meas_type': 'sampler'
#   }
#  })
#
# the model (model_d) for QNN contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'QNN',
#   'implementation_name': 'qiskit',
#   'model': QNN(provider=QiskitProvider)
#  })
#

#------------------------------------------------------------------------------
# Alg = QRBM: define dictionary keys for parameters
#------------------------------------------------------------------------------
#
# define the algorithm name and the available implementations
#
QRBM_NAME = "QRBM"
QRBM_IMPLS = [IMP_NAME_DWAVE]

# the parameter block (param_d) for QRBM looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'implementation_name': 'dwave',
#   'provider_name': 'dwave',
#   'encoder_name': 'bqm',
#   'n_hidden': 2,
#   'shots': 2,
#   'chain_strength': 2,
#   'k_nearest_neighbors': 2
#   }
#  })
#
# the model (model_d) for QRBM contains:
#
#  defaultdict(<class 'dict'>, {
#   'algorithm_name': 'QRBM',
#   'implementation_name': 'dwave',
#   'model': QRBM(provider=DWaveProvider)
#  })
#

#******************************************************************************
#
# ML Tools Parent Class: Alg
#
#******************************************************************************

class Alg:
    """
    class: Alg

    arguments:
     none

    description:
     This is a class that acts as a wrapper for all the algorithms supported
     in this library.
    """

    #--------------------------------------------------------------------------
    #
    # allocation methods: constructors/destructors/etc.
    #
    #--------------------------------------------------------------------------

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         none
        """

        # set the class name
        #
        Alg.__CLASS_NAME__ = self.__class__.__name__

        # set alg name to none
        #
        self.alg_d = None
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # assignment methods: set/get
    #
    #--------------------------------------------------------------------------

    def set(self, alg_name):
        """
        method: set

        arguments:
         alg_name: name of the algorithm as a string

        return:
         a boolean indicating status

        description:
         Note that this method does not descend into a specific alg class.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: set algorithm name (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, alg_name))

        # attempt to set the name only
        #
        if alg_name in ALGS:
            self.alg_d = ALGS[alg_name]()

        # if the algorithm is not set print an error message and exit
        #
        else:

            # print informational error message
            #
            print("Error: %s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "unknown algorithm name", alg_name))

            # exit ungracefully: algorithm setting failed
            #
            return False

        # exit gracefully
        #
        return True
    #
    # end of method

    def set_parameters(self, parameters):
        """
        method: set_parameters

        arguments:
         parameters: a dictionary object containing an algorithm's parameters

        return:
         a boolean value indicating status

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: setting parameters" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__))

        # check that the argument is a valid dictionary
        #
        if not isinstance(parameters, (dict, defaultdict)):

            # if not a valid dictionary print error message and exit
            #
            raise TypeError(
            f"{__FILE__} (line: {ndt.__LINE__} {ndt.__NAME__}: ",
            "invalid parameter structure",
            f"dict, defaultdict expected, got '{type(parameters).__name__}')")

        # set the parameters
        #
        self.alg_d.params_d = parameters

        # exit gracefully
        #
        return True
    #
    # end of method

    def get(self):
        """
        method: get

        arguments: none

        return:
         the current algorithm setting by name

        description:
         Note that this method does not descend into a specific alg class.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: getting the algorithm name" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__))

        # check if an algorithm is set
        #
        if self.alg_d is None:

            # print error message if the algorithm
            # was not set
            #
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "no algorithm has been set"))

            # exit ungracefully: algorithm is not set
            #
            return None

        # exit gracefully: return algorithm type
        #
        return self.alg_d.__class__.__name__
    #
    # end of method

    def get_ref_labels(self, data):
        """
        method: get_ref_labels

        arguments:
         data: the data including labels

        return:
         labels: a list of labels in a list (needed by numpy)

        description:
         We use this method to convert the data to a flat list of labels for
         the data. The reference labels are implied by the array location.
        """

        # get labels as the value of data dictionary
        #
        labels = data.labels

        # exit gracefully: return ref label info for numpy
        #
        return labels
    #
    # end of method

    def get_hyp_labels(self, hyp_labels):
        """
        method: get_hyp_labels

        arguments:
         data: the list of labels as a list of lists

        return:
         a list of labels in a list (needed by numpy)

        description:
         We use this method to convert the data to a flat list of labels.
        """

        # get labels as the value of data dictionary
        #
        labels = hyp_labels

        # exit gracefully: return the hyp_labels for numpy
        #
        return labels
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # i/o related methods: load/save
    #
    #--------------------------------------------------------------------------

    def load_model(self, fname):
        """
        method: load_model

        arguments:
         fname: a filename containing a model

        return:
         a dictionary containing the model

        description:
         This method loads a compatible picked model
        """

        # unpickle the model file
        #
        try:
            fp = open(fname, nft.MODE_READ_BINARY)
            model = pickle.load(fp)
        except:
            print("Error: %s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "error loading model file", fname))
            return None

        # check the type of data
        #
        if not isinstance(model, (dict, defaultdict)):
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "unknown model type"))
            return None

        # check model file key length: it should only contain 3 keys
        # ('algorithm_name', 'implementation_name', 'model')
        #
        if len(model) != int(3):
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "unknown model format"))
            return None

        if model[ALG_NAME_ALG] != self.alg_d.__class__.__name__:
            print("Error: %s (line: %s) %s::%s: %s, "
                  "Model Name: %s, Set Algorithm: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "the current set algorithm and model name does not match",
                   model[ALG_NAME_ALG],
                   self.alg_d.__class__.__name__))
            return None

        # set the parameters
        #
        self.alg_d.model_d = model

        # exit gracefully: return the model
        #
        return self.alg_d.model_d
    #
    # end of method

    def load_parameters(self, fname):
        """
        method: load_parameters

        arguments:
         fname: a filename containing a model parameters

        return:
         a dictionary containing the parameters

        description:
         This method loads a specific algorithm parameter block.
         the algorithm name must be set before it is called.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: loading parameters (%s)" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   fname))

        # check that an algorithm has been set
        #
        if self.alg_d is None:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "no algorithm has been set"))
            return None

        # expand the filename
        #
        ffname = nft.get_fullpath(fname)

        # make sure the file is a valid parameter file
        #
        if nft.get_version(ffname) is None:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "invalid parameter file (invalid version)"))
            return None

        # attempt to load the parameters
        #
        params = nft.load_parameters(ffname, self.alg_d.__class__.__name__)
        if params is None:
            print("Error: %s (line: %s) %s::%s: %s [%s]" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "unable to parse parameter file", fname))
            return None

        # set the internal parameters
        #
        self.alg_d.params_d = params

        # exit gracefully: return loaded parameters
        #
        return self.alg_d.params_d
    #
    # end of method

    def save_model(self, fname):
        """
        method: save_model

        arguments:
         fname: a filename to be written

        return:
         a boolean value indicating status

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: saving model (%s)" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   fname))

        # check that there is a valid model
        #
        if self.alg_d is None:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "invalid model"))
            return False

        # pickle it to a file and trap for errors
        #
        try:
            fp = open(fname, nft.MODE_WRITE_BINARY)
            pickle.dump(self.alg_d.model_d, fp)
            fp.close()
        except:
            print("Error: %s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "error writing model file", fname))
            return False

        # exit gracefully: model successfully saved
        #
        return True
    #
    # end of method

    def save_parameters(self, fname):
        """
        method: save_parameters

        arguments:
         fname: a filename to be written

        return:
         a boolean value indicating status

        description:
         This method writes the current self.alg.params_d to an
         NEDC parameter file
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: saving parameters" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__))

        # check if an algorithm has been set
        #
        if self.alg_d is None:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "no algorithm has been set"))
            return False

        # check if params_d is empty
        #
        if self.alg_d.params_d is None:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "parameter block is empty"))
            return False

        # open the file for writing
        #
        try:
            fp = open(fname, nft.MODE_WRITE_TEXT)
        except:
            print("Error: %s (line: %s) %s::%s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   fname))

        # write a bare bones parameter file:
        #  start with the version information
        #
        fp.write("%s %s %s" % (nft.DELIM_VERSION, nft.DELIM_EQUAL,
                               nft.PFILE_VERSION + nft.DELIM_NEWLINE))
        fp.write("%s %s %s" % (nft.DELIM_OPEN, self.alg_d.__class__.__name__,
                               nft.DELIM_CLOSE + nft.DELIM_NEWLINE))

        # add the parameter structure
        #
        for key, val in self.alg_d.params_d.items():
            fp.write(" %s %s %s" % (key, nft.DELIM_EQUAL,
                                    val + nft.DELIM_NEWLINE))

        # exit gracefully: parameters successfully written to parameter file
        #
        return True
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              *, # This means that after the "data" argument,
                 # you would need a keyword argument.
              write_train_labels = False,
              fname_train_labels = "train_labels.csv"):
        """
        method: train

        arguments:
         data: a list of numpy matrices
         write_train_labels: output the predicted labels from training (False)
         fname_train_labels: the fname to output it too ("train_labels.csv")
         excel: if true, write the file as excel (.xlsx) (False)

         Note: After the "data" argument, you would need a keyword argument.

         Ex: train(data, write_train_labesl = True, fname = "labels.csv")

        return:
         model: a dictionary containing the model (data dependent)
         score: a float value containing a measure of the goodness of fit

        description:
         Note that data is a list of matrices organized by class label,
         so the labels are implicit in the data. the index in the list is
         the class label, and the matrix are the feature vectors:

         class 0:
          data[0] = array[[ 1  2  3  4],
                          [ 5  6  7  8]]
         class 1:
          data[1] = array[[10 11 12 13],
                          [14 15 16 17],
                          [90 91 92 93]]

         note that these are numpy matrices not native Python lists.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__))

        # check if the algorithm has been configured
        #
        if self.alg_d is None:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "no algorithm has been set"))

            # exit ungracefully: algorithm not set
            #
            return None, None

        # check that the data variable is an MLToolsData
        # instance
        #
        if not isinstance(data, MLToolsData):
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "data is not type of MLToolsData"))

            # exit ungracefully: incompatible object type
            #
            return None, None

        # exit gracefully: return model and its score
        #
        return self.alg_d.train(data, write_train_labels, fname_train_labels)
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels
         posteriors: a float numpy vector with the posterior probability
         for each class assignment

        description:
         This is a wrapper method that calls the set algorithms predict method
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__))

        # check if the algorithm has been configured
        #
        if self.alg_d is None:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "no algorithm has been set"))

            # exit ungracefully: algorithm not set
            #
            return None, None

        # check if data variable is an MLToolsData instance
        #
        if not isinstance(data, MLToolsData):
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__,
                   "data is not type of MLToolsData"))

            # exit ungracefully: incompatible object type
            #
            return None, None

        # exit gracefully: return model prediction
        #
        return self.alg_d.predict(data, model)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # scoring methods: score/report
    #
    #--------------------------------------------------------------------------

    def confusion_matrix(self, num_classes, ref_labels, hyp_labels):
        """
        method: confusion_matrix

        arguments:
         num_classes: the number of classes
         ref_labels: a list of reference labels
         hyp_labels: a list of hypothesized labels

        return:
         a confusion matrix that is a numpy array

        description:
         Note: we pass the number of classes because all of the classes
         might not appear in the reference labels.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: generating a confusion matrix" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__))

        # build a labels matrix
        #
        lbls = list(range(num_classes))

        # exit gracefully: return confusion matrix
        #
        conf = sklearn_confusion_matrix(ref_labels, hyp_labels)
        return conf
    #
    # end of method

    def print_confusion_matrix(self, cnf, mapping_label, fp = sys.stdout):
        """
        method: print_confusion_matrix

        arguments:
         cnf: the confusion matrix
         mapping_label: the mapping labels from an algorithm
         fp: an open file pointer [stdout]

        return:
         a boolean value indicating status

        description:
         Prints the confusion matrix.
        """

        # get the number of rows and columns for the numeric data:
        #  we assume a square matrix in this case
        #
        nrows = len(cnf)
        ncols = len(cnf)

        # create the table headers
        #
        headers = ["Ref/Hyp:"]

        # iterate over the confusion matrix rows
        #
        for i in range(nrows):

            # append mapping label to headers list
            #
            if isinstance(mapping_label[i], int):
                headers.append(ALG_FMT_LBL % mapping_label[i])
            else:
                headers.append(mapping_label[i])

        # convert the confusion matrix to percentages
        #
        pct = np.empty_like(cnf, dtype = float)
        # sum over the confusion matrix rows
        #
        for i in range(nrows):

            # sum the rows values
            #
            sum = float(cnf[i].sum())

            # convert (row, column) of confusion matrix to percentages
            #
            for j in range(ncols):
                pct[i][j] = float(cnf[i][j]) / sum

        # get the width of each column and compute the total width:
        # the width of the percentage column includes "()" and two spaces
        #
        width_lab = int(float(ALG_FMT_WLB[1:-1]))
        width_cell = int(float(ALG_FMT_WCL[1:-1]))
        width_pct = int(float(ALG_FMT_WPC[1:-1]))
        width_paren = int(4)
        total_width_cell = width_cell + width_pct + width_paren
        total_width_table = width_lab + \
            ncols * (width_cell + width_pct + width_paren)

        # print the title
        #
        title = "confusion matrix"
        fp.write("%s".center(total_width_table - len(title)) % title)
        fp.write(nft.DELIM_NEWLINE)

        # print the first heading label right-aligned
        #
        fp.write("%*s" % (width_lab, "Ref/Hyp:"))

        # print the next ncols labels center-aligned:
        #  add a newline at the end
        #
        for i in range(1, ncols + 1):

            # compute the number of spaces needed to center-align
            #
            num_spaces = total_width_cell - len(headers[i])
            num_spaces_2 = int(num_spaces / 2)

            # write spaces, header, spaces
            #
            fp.write("%s" % nft.DELIM_SPACE * num_spaces_2)
            fp.write("%s" % headers[i])
            fp.write("%s" % nft.DELIM_SPACE * (num_spaces - num_spaces_2))

        fp.write(nft.DELIM_NEWLINE)

        # write the rows with numeric data:
        #  note that "%%" is needed to print a percent
        #
        for i in range(nrows):

            # write the row label
            #
            fp.write("%*s" % (width_lab, headers[i+1] + nft.DELIM_COLON))

            # write the numeric data and then add a new line
            #
            for j in range(ncols):
                fp.write(ALG_FMT_WST % (cnf[i][j], ALG_FMT_PCT * pct[i][j]))
            fp.write(nft.DELIM_NEWLINE)

        # exit gracefully: confusion matrix successfully printed
        #
        return True
    #
    # end of method

    def score(self, num_classes, data, hyp_labels, *,
              isPrint = False, fp = sys.stdout):
        """
        method: score

        arguments:
         num_classes: the number of classes
         data: the input data including reference labels
         hyp_labels: the hypothesis labels
         isPrint: a flag to print out the scoring output (False)

        return:
         conf_matrix, sens, spec, prec, acc, err, f1

         if print = True:
            return None

        description:
         Note that we must provide the number of classes because the data
         might not contain all the data.
        """

        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: scoring the results" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__))

        # get the reference and hypothesis field member data
        #
        r_labels = self.get_ref_labels(data)
        h_labels = self.get_hyp_labels(hyp_labels)


        # calculate confusion matrix
        #
        conf_matrix = self.confusion_matrix(num_classes, r_labels, h_labels)

        # calculate accuracy and error score
        #
        acc = accuracy_score(r_labels, h_labels)
        err = ALG_FMT_PCT * (float(1.0) - acc)
        sens = None
        spec = None
        prec = None
        f1 = None

        # print confusion matrix and return nothing if specified
        #
        if isPrint:

            # print the confusion matrix
            #
            self.print_confusion_matrix(conf_matrix, data.mapping_label,
                                        fp = fp)
            fp.write(nft.DELIM_NEWLINE)

            # generate and print the classification report
            #
            rpt = classification_report(r_labels, h_labels, zero_division = 1)
            fp.write(rpt)
            fp.write(nft.DELIM_NEWLINE)

            # print out the error rate
            #
            print(ALG_FMT_ERR % ("error rate", err))

        else:

            # set the averaging method accordingly
            #
            if num_classes > 2:
                average='macro'
            else:
                average='binary'

            # calculate necessary scores
            #
            sens = sensitivity_score(r_labels, h_labels, average=average)
            spec = specificity_score(r_labels, h_labels, average=average)
            prec = precision_score(r_labels, h_labels, average=average)
            f1 = f1_score(r_labels, h_labels, average=average)

        if isPrint:
            return None

        return conf_matrix, sens, spec, prec, acc, err, f1
    #
    # end of method

    def print_score(self, num_classes, data, hyp_labels, fp = sys.stdout):
        """
        method: print_score

        arguments:
         num_classes: the number of classes
         data: the input data including reference labels
         hyp_labels: the hypothesis labels
         fp: an open file pointer [stdout]

        return:
         a boolean value indicating status

        description:
         Note that we must provide the number of classes because the data
         might not contain all the data.
        """

        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: scoring the results" %
                  (__FILE__, ndt.__LINE__, Alg.__CLASS_NAME__, ndt.__NAME__))

        # print the date and time
        #
        fp.write(ALG_FMT_DTE % (dt.datetime.now(), nft.DELIM_NEWLINE))
        fp.write(nft.DELIM_NEWLINE)

        # use numpy to generate a confusion matrix
        #
        rlabels = self.get_ref_labels(data)
        hlabels = self.get_hyp_labels(hyp_labels)
        cnf = self.confusion_matrix(num_classes, rlabels, hlabels)

        # print the confusion matrix in ISIP format
        #
        self.print_confusion_matrix(cnf, fp)
        fp.write(nft.DELIM_NEWLINE)

        # generate and print the classification report
        #
        rpt = classification_report(rlabels, hlabels,
                                    zero_division = 1)
        fp.write(rpt)
        fp.write(nft.DELIM_NEWLINE)

        # compute the accuracy and the error rate
        #
        acc = accuracy_score(rlabels, hlabels)
        err = ALG_FMT_PCT * (float(1.0) - acc)
        print(ALG_FMT_ERR % ("error rate", err))

        # exit gracefully: score successfully printed
        #
        return True
    #
    # end of method

    def get_info(self):
        """
        method: get_info
  
        arguments:
         none
  
        return:
         a dictionary containing the algorithm information
  
        description:
         this method returns the information of the current algorithm.
        """
  
        # get the algorithm parameters
        #
        if hasattr(self.alg_d, 'get_info'):
            parameter_outcomes = self.alg_d.get_info()
        else:
            parameter_outcomes = None
 
        # exit gracefully
        # return the parameter outcomes
        #
        return parameter_outcomes
    #
    # end of method

#
# end of class (Alg)

#******************************************************************************
#
# Section 1: discriminant-based algorithms
#
#******************************************************************************

#------------------------------------------------------------------------------
# ML Tools Class: EUCLIDEAN
#------------------------------------------------------------------------------

class EUCLIDEAN:
    """
    class: Euclidean

    description:
     This is a class that implements Euclidean Distance.
    """

    def __init__(self) -> None:
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        EUCLIDEAN.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block (param_d) and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = NB_IMPLS[0]
        self.model_d[ALG_NAME_MDL] = defaultdict(dict)


    #
    # end of method

    def weightedDistance(self, p1, p2, w):
        """
        method: weightedDistance

        arguments:
         p1: point 1 (numpy array)
         p2: point 2 (numpy array)
          w: weight

        return:
         the weighted euclidean distance

        description:
         This method returns the weighted euclidean distance
        """

        q = p2 - p1

        # exit gracefully
        #
        return np.sqrt((w*q*q).sum())
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary containing the model (data dependent)
         score: a float value containing a measure of the goodness of fit

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, EUCLIDEAN.__CLASS_NAME__,
                   ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, EUCLIDEAN.__CLASS_NAME__,
                   ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # get sorted_labels and sorted_samples
        #
        data = data.sort()
        group_data = data.group_by_class()

        # a list of mean for each class
        #
        means = []
        for d in group_data.values():
            means.append(np.mean(d, axis = 0))

        self.model_d[ALG_NAME_MDL][ALG_NAME_MEANS] = means

        # get the weights
        #
        weights = self.params_d[ALG_NAME_WEIGHTS]

        # get algorithm implementation
        #
        impl = self.params_d[ALG_NAME_IMP]

        if ALG_NAME_IMP in self.params_d:
            impl = self.params_d[ALG_NAME_IMP]
        else:
            impl = self.model_d[ALG_NAME_IMP]

        # check if the implementation is valid
        #
        if impl not in EUCL_IMPLS:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, EUCLIDEAN.__CLASS_NAME__,
                   ndt.__NAME__, "unknown implementation"))
            return None

        if impl == IMP_NAME_DISCRIMINANT:

            # scoring
            #
            acc = 0
            for true_label, d in zip(data.labels, data.data):

                diff_means = []

                for ind, mean in enumerate(means):
                    diff_means.append(
                        self.weightedDistance(mean, d, float(weights[ind]))
                    )

                train_label_ind = np.argmin(diff_means)
                if data.mapping_label[train_label_ind] == true_label:
                    acc += 1

            self.model_d[ALG_NAME_IMP] = impl
            self.model_d[ALG_NAME_MDL][ALG_NAME_WEIGHTS] = weights

        score = acc / len(data.data)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a list of numpy float matrices of feature vectors
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels
         posteriors: a list of float numpy vectors with the posterior
         probabilities for each class

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, EUCLIDEAN.__CLASS_NAME__,
                   ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # check for model validity
        #
        if model[ALG_NAME_ALG] != EUCL_NAME:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, EUCLIDEAN.__CLASS_NAME__,
                   ndt.__NAME__, "incorrect model name"))
            return None, None

        if not model[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, EUCLIDEAN.__CLASS_NAME__,
                   ndt.__NAME__, "model parameter is empty"))
            return None, None

        means = self.model_d[ALG_NAME_MDL][ALG_NAME_MEANS]
        weights = self.model_d[ALG_NAME_MDL][ALG_NAME_WEIGHTS]
        mapping_label = data.mapping_label

        labels = []
        posteriors = []

        for d in data.data:

            diff_means = []
            cur_posteriors = []

            for ind, mean in enumerate(means):
                distance = self.weightedDistance(mean, d, float(weights[ind]))
                diff_means.append(distance)
                cur_posteriors.append(distance)
            predict_label_ind = np.argmin(diff_means)
            labels.append(predict_label_ind)
            posteriors.append(cur_posteriors)

        # exit gracefully
        #
        return labels, posteriors
    #
    # end of method
#
# end of class (EUCLIDEAN)

#------------------------------------------------------------------------------
# ML Tools Class: PCA
#------------------------------------------------------------------------------

class PCA:
    """
    class: PCA

    arguments:
     none

    description:
     This is a class that implements Principal Components Analysis (PCA).
    """

    #--------------------------------------------------------------------------
    #
    # constructors/destructors/etc.
    #
    #--------------------------------------------------------------------------

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        PCA.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict(dict)

        # initialize a parameter dictionary
        #
        self.params_d = defaultdict(dict)

        # set the names
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_DISCRIMINANT
        self.model_d[ALG_NAME_MDL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary containing the model (data dependent)
         score: a float value containing a measure of the goodness of fit

        description:
         PCA is implemented as what is known as "pooled covariance", meaning
         that a single covariance matrix is computed over all the data. See:
         https://www.askpython.com/python/examples/principal-component-analysis
         for more details.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, PCA.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, PCA.__CLASS_NAME__, ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # get sorted_labels and sorted_samples
        #
        data = data.sort()

        # fetch the unique labels
        #
        uni_label = np.unique(data.labels)

        # create list to hold new data
        #
        new_data = []

        # iterate over all unique labels
        #
        for i in range(len(uni_label)):

            # create a temporary list to hold class data
            #
            class_data = []

            # iterate over all labels stored in the MLToolsData instance
            #
            for j in range(len(data.labels)):

                # remap data
                #
                if uni_label[i] == data.labels[j]:
                    class_data.append(data.data[j])

            # convert new data into numpy array
            #
            new_data.append(np.array(class_data))

        # calculating number of classes
        #
        num_classes = len(new_data)

        # number of samples
        #
        npts = sum(len(element) for element in new_data)

        # initialize an empty array to save the priors
        #
        priors = np.empty((0,0))

        # case: (ml) equal priors
        #
        mode_prior = self.params_d[ALG_NAME_PRIORS]

        if mode_prior == ALG_NAME_PRIORS_ML:

            # compute the priors for each class
            #
            self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS] = \
                np.full(shape = num_classes,
                        fill_value = (float(1.0) / float(num_classes)))

        # if the priors are based on occurrences: nct.PRIORs_MAP
        #
        elif mode_prior == ALG_NAME_PRIORS_MAP:

            # create an array of priors by finding the the number of points
            # in each class and dividing by the total number of samples
            #
            for element in new_data:

                # appending the number of the samples in
                # each element (class) to empty array
                #
                priors = np.append(priors, len(element))

            # final calculation of priors
            #
            _sum = float(1.0) / float(npts)

            self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS] = priors * _sum

        else:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, PCA.__CLASS_NAME__, ndt.__NAME__,
                   "unknown value for priors"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # calculate the means of each class:
        #  note these are a list of numpy vectors
        #
        means = []
        for element in new_data:
            means.append(np.mean(element, axis = 0))

        # converting into numpy array
        #
        self.model_d[ALG_NAME_MDL][ALG_NAME_MEANS] = means

        # calculate the cov:
        #  note this is a single matrix
        #
        self.model_d[ALG_NAME_MDL][ALG_NAME_COV] = nct.compute(
            new_data,
            ctype = self.params_d[nct.PRM_CTYPE],
            center = self.params_d[nct.PRM_CENTER],
            scale = self.params_d[nct.PRM_SCALE])
        cov = self.model_d[ALG_NAME_MDL][ALG_NAME_COV]

        # number of components
        #
        n_comp= int(self.params_d[ALG_NAME_NCMP])

        if n_comp == -1:
            n_comp = new_data[0].shape[1]

        #if not( 0 < n_comp <= len(data.data[0])):
        if not( 0 < n_comp <= new_data[0].shape[1]):
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, PCA.__CLASS_NAME__, ndt.__NAME__,
                   "features out of range"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # eigenvalue and eigen vector decomposition
        #
        eigvals, eigvecs = np.linalg.eig(cov)

        # sorting based on eigenvalues
        #
        sorted_indexes = eigvals.argsort()[::-1]

        # get eigenvalue and eigenvectors
        #
        eigvals = eigvals[sorted_indexes[0:n_comp]]
        eigvecs = eigvecs[:,sorted_indexes[0:n_comp]]
        if any(eigvals < 0):
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, PCA.__CLASS_NAME__, ndt.__NAME__,
                   "negative eigenvalues"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # This steps (below) normalize the eigenvectors by scaling
        # them with the inverse square root of eigenvalues.
        # This "whitening" transformation ensures that the transformed
        # features have unit variance.
        #
        try:
            eigval_in = np.linalg.inv(np.diag(eigvals ** (1/2)))

        except np.linalg.LinAlgError:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, PCA.__CLASS_NAME__, ndt.__NAME__,
                   "singular matrix model is none"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # transformation matrix that will transform the data and
        # means into a subspace
        #
        transformation_mat = np.dot(eigvecs, eigval_in)
        self.model_d[ALG_NAME_MDL][ALG_NAME_TRANS] = transformation_mat

        # compute a goodness of fit measure: use the average weighted
        # mean-square-error of the reconstructed data computed across the
        # entire data set using the transformation matrix we computed.
        #
        gsum = float(0.0)
        for i, d in enumerate(new_data):

            # iterate over the data in class i and calculate the norms
            #
            _sum = float(0.0)
            for v in d:
                diff = np.dot(transformation_mat, (v - means[i]).T)

                # mean-squared-error of the reconstructed data
                # equivalent to np.linalg.norm(diff)**2
                #
                _sum += np.dot(diff.T, diff)

            # weight by the prior
            #
            gsum += self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS][i] * _sum

        # extracting the scalar from 2d matrix
        #
        score = (gsum / float(npts))

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a list of numpy float matrices of feature vectors
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels
         posteriors: a list of float numpy vectors with the posterior
         probabilities for each class

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, PCA.__CLASS_NAME__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # check for model validity
        #
        if model[ALG_NAME_ALG] != PCA_NAME:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, PCA.__CLASS_NAME__, ndt.__NAME__,
                   "incorrect model name"))
            return None, None

        if not model[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, PCA.__CLASS_NAME__, ndt.__NAME__,
                   "model parameter is empty"))
            return None, None

        # get the number of dimensions
        #
        ndim = data.data.shape[1]

        # transformation matrix that will transform the data and
        # means into a subspace
        #
        transformation_mat =  model[ALG_NAME_MDL][ALG_NAME_TRANS]
        means = model[ALG_NAME_MDL][ALG_NAME_MEANS]

        # calculate number of classes
        #
        num_classes = len(means)

        transformed_means = []
        for i in range(len(means)):
            m_new = np.dot(means[i], transformation_mat)
            transformed_means.append(m_new)

        # converting into numpy array for computational safety
        # transformed_means = np.array(transformed_means)
        #
        # pre-compute the scaling term
        #
        scale = np.power(2 * np.pi, -ndim / 2)

        # loop over data
        #
        labels = []
        posteriors = []

        for j in range(len(data.data)):

            # loop over number of classes
            #
            sample = data.data[j]
            transformed_sample = np.dot(sample, transformation_mat)
            evidence = 0
            post = []

            for k in range(num_classes):

                # manually compute the log likelihood
                # as a weighted Euclidean distance
                #
                # mean of class k
                #
                prior =  model[ALG_NAME_MDL][ALG_NAME_PRIORS][k]

                # mean of class k
                #
                trans_mean = transformed_means[k]

                # posterior calculation for sample j
                #
                sq_distance = np.dot((transformed_sample - trans_mean).T,
                                     (transformed_sample - trans_mean))
                gauss_likelihood = np.exp(-1/2 * sq_distance)
                posterior_contrib = gauss_likelihood * scale * prior
                evidence = evidence + posterior_contrib
                post.append(posterior_contrib)

            # convert the likelihoods into posteriors
            #
            post = post/evidence


            # choose the class label with the highest posterior
            #
            labels.append(np.argmax(post))

            # save the posteriors
            #
            posteriors.append(post)

        # exit gracefully
        #
        return labels, posteriors
    #
    # end of method

    def get_info(self):
        """
        method: get_info
  
        arguments:
         none
  
        return:
         a dictionary containing the algorithm information
  
        description:
         this method returns the means and covariances of the current algorithm.
        """
  
        # get the algorithm means and covariances
        #
        data = {
            "means": self.model_d[ALG_NAME_MDL][ALG_NAME_MEANS],
            "covariances": self.model_d[ALG_NAME_MDL][ALG_NAME_COV]
        }
 
        log_output = ""
        class_index = 0
 
        # Handle means (list of numpy arrays)
        if 'means' in data:
            for mean in data['means']:
                log_output += f"Class {class_index} Means: {mean.tolist()}\n"
                class_index += 1
 
        # Handle covariances (numpy 2D array)
        if 'covariances' in data:
            matrix = data['covariances']
            log_output += f"Covariances: {matrix[0]}\n"
 
            # Blank space for padding as per your requirement
            for row in matrix[1:]:
                log_output += f"             {row}\n"
 
        # exit gracefully
        #
        return log_output
    #
    # end of method
#
# end of PCA

#------------------------------------------------------------------------------
# ML Tools Class: LDA
#------------------------------------------------------------------------------

class LDA:
    """
    class: LDA

    description:
     This is a class that implements class_independent Linear Discriminant
     Analysis (LDA).
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        LDA.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict(dict)

        # initialize a parameter dictionary
        #
        self.params_d = defaultdict(dict)

        # set the names
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_DISCRIMINANT
        self.model_d[ALG_NAME_MDL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: as a dictionary of priors, means and covariance
         score: a float value containing a measure of the goodness of fit

        description:
         LDA is implemented using this link :
         https://usir.salford.ac.uk/id/eprint/52074/1/AI_Com_LDA_Tarek.pdf
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, LDA.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, LDA.__CLASS_NAME__, ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # calculate number of classes
        #
        data = data.sort()
        uni_label = np.unique(data.labels)
        new_data = []

        for i in range(len(uni_label)):

            # loop over the labels
            #
            class_data =[]
            for j in range(len(data.labels)):
                if uni_label[i]==data.labels[j]:
                    class_data.append(data.data[j])
            new_data.append(np.array(class_data))

        # set the number of classes
        #
        num_classes = len(new_data)

        # calculate number of data points
        #
        npts = 0
        for element in new_data:
            npts += len(element)

        # initialize an empty array to save the priors
        #
        priors = np.empty((0,0))

        # case: (ml) equal priors
        #
        mode_prior = self.params_d[ALG_NAME_PRIORS]
        if mode_prior == ALG_NAME_PRIORS_ML:

            # compute the priors for each class
            #
            priors = np.full(shape = num_classes,
                            fill_value = (float(1.0) / float(num_classes)))
            self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS] = priors

        # if the priors are based on occurrences
        #
        elif mode_prior == ALG_NAME_PRIORS_MAP:

            # create an array of priors by finding the the number of points
            # in each class and dividing by the total number of samples
            #
            for element in new_data:

                # appending the number of samples in
                # each element(class) to empty array
                #
                priors = np.append(priors, len(element))

            # final calculation of priors
            #
            sum = float(1.0) / float(npts)
            priors = priors * sum
            self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS] = priors

        else:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, LDA.__CLASS_NAME__, ndt.__NAME__,
                   "unknown value for priors"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # calculate the means of each class:
        # note these are a list of numpy vectors
        #
        means = []
        for elem in new_data:
            means.append([np.mean(elem, axis = 0)])

        means = np.array(means)

        self.model_d[ALG_NAME_MDL][ALG_NAME_MEANS] = means

        # calculate the global mean
        #
        mean_glob = np.mean(np.vstack(new_data), axis = 0)

        # calculate s_b and s_w.
        # we need to calculate them for the transformation matrix
        #
        n_features = new_data[0].shape[1]

        # initialize within class scatter
        #
        sw = np.zeros((n_features, n_features))

        # initialize between class scatter
        #
        sb = np.zeros((n_features, n_features))

        # within class scatter calculation
        #
        for i, d in enumerate(new_data):

            # calculation of within class scatter
            #
            sw += nct.compute(d,
                              ctype = self.params_d[nct.PRM_CTYPE],
                              center = self.params_d[nct.PRM_CENTER],
                              scale= self.params_d[nct.PRM_SCALE])

            # compute the between class scatter calculation
            #
            mean_diff = means[i] - mean_glob

            # number of data point in class d
            #
            n_c = d.shape[0]
            sb += n_c * np.dot(mean_diff.T, mean_diff)

        try:
            sw_in = np.linalg.inv(sw)

        except np.linalg.LinAlgError:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, LDA.__CLASS_NAME__, ndt.__NAME__,
                   "singular matrix"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # calculation of sw^-1*sb
        #
        transformation_base = sw_in.dot(sb)

        # number of the eigenvectors need to be chosen
        # it is equal to the num_class minus 1
        #
        reduced_dimension = min((len(new_data))-1, n_features)

        # eigen vector and eigen value decomposition
        #
        eigvals, eigvecs = np.linalg.eig(transformation_base)

        # sorted eigenvalues and eigvecs and choose
        # the first l-1 columns from eigenvalues
        # and eigenvectors
        #
        sorted_indexes = eigvals.argsort()[::-1]
        eigvals = eigvals[sorted_indexes[0:reduced_dimension]]
        eigvecs = eigvecs[:,sorted_indexes[0:reduced_dimension]]
        if any(eigvals < 0):
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, LDA.__CLASS_NAME__, ndt.__NAME__,
                   "negative eigenvalues"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # This steps(below) normalizes the eigenvectors by scaling them with
        # the inverse square root of eigenvalues and "whitening" transformation
        # ensures that the transformed features have unit variance.
        #
        try:
            eigval_in = np.linalg.inv(np.diag(eigvals**(1/2)))

        except np.linalg.LinAlgError:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, LDA.__CLASS_NAME__, ndt.__NAME__,
                   "singular matrix model is none"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # calculation of the transformation matrix that will transform
        # the data and means into a subspace
        #
        transformation_mat = np.dot(eigvecs, eigval_in)
        self.model_d[ALG_NAME_MDL][ALG_NAME_TRANS] = transformation_mat


        # compute a goodness of fit measure: use the average weighted
        # mean-square-error of the reconstructed data computed across the
        # entire data set using the transformation matrix we computed.
        #
        gsum = float(0.0)
        for i, d in enumerate(new_data):

            # iterate over the data in class i and calculate the norms
            #
            _sum = float(0.0)
            for v in d:

                # mean-squared-error of the reconstructed data
                # equivalent to np.linalg.norm(diff)**2
                #
                diff = np.dot((v - means[i]), transformation_mat)
                _sum += np.dot(diff.T, diff)

            # weight by the prior
            #
            gsum += self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS][i] * _sum

        # extracting the scalar from 2d matrix
        #
        score = (gsum / float(npts))[0][0]

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels
         posteriors: a list of float numpy vectors with the posterior
         probabilities for each class

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, LDA.__CLASS_NAME__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # check for model validity
        #
        if model[ALG_NAME_ALG] != LDA_NAME:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, LDA.__CLASS_NAME__, ndt.__NAME__,
                   "incorrect model name"))
            return None, None

        if not model[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, LDA.__CLASS_NAME__, ndt.__NAME__,
                   "model parameter is empty"))
            return None, None

        # get the number of dimensions
        #
        ndim = data.data.shape[1]

        # transformation matrix that will transform the data and
        # means into a subspace
        #
        transformation_mat = model[ALG_NAME_MDL][ALG_NAME_TRANS]
        means = model[ALG_NAME_MDL][ALG_NAME_MEANS]

        # calculate number of classes and transform means
        #
        num_classes = len(means)
        transformed_means = []
        for i in range(len(means)):
            m_new = np.dot(means[i], transformation_mat)
            transformed_means.append(m_new)

        # pre-compute the scaling term
        #
        scale = np.power(2 * np.pi, -ndim / 2)

        # loop over data
        #
        labels = []
        posteriors = []
        for j in range(len(data.data)):

            # loop over number of classes
            #
            sample = data.data[j]
            transformed_sample = np.dot(sample, transformation_mat)
            evidence = 0
            post =[]

            # loop over the number of classes
            #
            for k in range(num_classes):

                # manually compute the log likelihood
                # as a weighted Euclidean distance
                #
                # mean of class k
                #
                prior =  model[ALG_NAME_MDL][ALG_NAME_PRIORS][k]

                # mean of class k
                # taking the [0] is for reducing one wrapper dimension
                #
                trans_mean = transformed_means[k][0]

                # posterior calculation for sample j
                #
                sq_distance = np.dot((transformed_sample - trans_mean).T,
                                     (transformed_sample - trans_mean))
                gauss_likelihood = np.exp(-1/2 * sq_distance)
                posterior_contrib = gauss_likelihood * scale * prior
                evidence = evidence + posterior_contrib
                post.append(posterior_contrib)

            # convert likelihoods to posteriors
            #
            post = post/evidence

            # choose the class label with the highest posterior
            #
            labels.append(np.argmax(post))

            # save the posteriors
            #
            posteriors.append(post)

        # exit gracefully
        #
        return labels, posteriors
    #
    # end of method

    def get_info(self):
        """
        method: get_info
  
        arguments:
         none
  
        return:
         a dictionary containing the algorithm information
  
        description:
         this method returns the means and covariances of the current algorithm.
        """
  
        # get the algorithm means and covariances
        #
        data = {
            "means": self.model_d[ALG_NAME_MDL][ALG_NAME_MEANS],
        }
         
        log_output = ""
        class_index = 0
 
        # Handle means (list of numpy arrays)
        if 'means' in data:
            for mean in data['means']:
                log_output += f"Class {class_index} Means: {mean.tolist()}\n"
                class_index += 1
 
        # exit gracefully
        #
        return log_output
    #
    # end of method
#
# end of LDA

#------------------------------------------------------------------------------
# ML Tools Class: QDA
#------------------------------------------------------------------------------

class QDA:
    """
    class: QDA

    arguments:
     none

    description:
     This is a class that implements Quadratic Components Analysis (QDA). This
     is also known as class-dependent PCA.
    """

    #--------------------------------------------------------------------------
    #
    # constructors/destructors/etc.
    #
    #--------------------------------------------------------------------------

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        QDA.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict(dict)

        # initialize a parameter dictionary
        #
        self.params_d = defaultdict(dict)

        # set the names
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_DISCRIMINANT
        self.model_d[ALG_NAME_MDL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean whether to write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary containing the model (data dependent)
         score: a float value containing a measure of the goodness of fit

        description:
         QDA is implemented as class dependent PCA.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, QDA.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, QDA.__CLASS_NAME__, ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # getting the number of classes/labels
        #
        data = data.sort()
        uni_label = np.unique(data.labels)
        new_data =[]
        for i in range(len(uni_label)):

            # loop over the labels to store same label data into a list
            #
            class_data =[]
            for j in range(len(data.labels)):
                if uni_label[i] == data.labels[j]:
                    class_data.append(data.data[j])

            # adding smae label data list into another list
            #
            new_data.append(np.array(class_data))

        # set the number of classes
        #
        num_classes = len(new_data)

        # number of components
        #
        n_comp = int(self.params_d[ALG_NAME_NCMP])
        if n_comp == -1:
            n_comp = new_data[0].shape[1]
        if not(0 < n_comp <= new_data[0].shape[1]):
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, QDA.__CLASS_NAME__, ndt.__NAME__,
                   "features out of range"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # calculate number of data points
        #
        npts = 0
        for element in new_data:
            npts += len(element)

        # initialize an empty array to save the priors
        #
        priors = np.empty((0,0))

        # case: (ml) equal priors
        #
        mode_prior = self.params_d[ALG_NAME_PRIORS]
        if mode_prior == ALG_NAME_PRIORS_ML:

            # save priors for each class
            #
            self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS] = \
                np.full(shape = num_classes,
                        fill_value = (float(1.0) / float(num_classes)))

        # if the priors are based on occurrences : nct.PRIORs_MAP
        #
        elif mode_prior == ALG_NAME_PRIORS_MAP:

            # create an array of priors by finding the the number of points
            # in each class and dividing by the total number of samples
            #
            for element in new_data:

                # appending the number of samples in
                # each element(class) to empty array
                #
                priors = np.append(priors, len(element))

            # final calculation of priors
            #
            sum = float(1.0) / float(npts)
            self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS] = priors * sum

        # returning None due to unknown prior modes
        #
        else:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, QDA.__CLASS_NAME__, ndt.__NAME__,
                   "unknown value for priors"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # calculate the means of each class:
        #  note these are a list of numpy vectors
        #
        means = []
        for elem in new_data:
            means.append(np.mean(elem, axis = 0))
        self.model_d[ALG_NAME_MDL][ALG_NAME_MEANS] = means

        # loop over classes to calculate covariance of each class
        #
        transformation_mats = []
        for i, element in enumerate(new_data):

            # getting the covariance matrix for each class data
            #
            covar = nct.compute(element,
                ctype= self.params_d[nct.PRM_CTYPE],
                center = self.params_d[nct.PRM_CENTER],
                scale = self.params_d[nct.PRM_SCALE])

            # eigen vector and eigen value decomposition
            # for each class
            #
            eigvals, eigvecs = np.linalg.eig(covar)

            # sorted eigenvalues and eigvecs and choose
            # the first l-1 columns from eigenvalues
            # and eigenvectors
            #
            sorted_indexes = eigvals.argsort () [::-1]
            eigvals = eigvals[sorted_indexes[0:n_comp]]
            eigvecs = eigvecs[:,sorted_indexes[0:n_comp]]
            if any(eigvals < 0):
                print("Error: %s (line: %s) %s::%s: %s" %
                      (__FILE__, ndt.__LINE__, QDA.__CLASS_NAME__,
                       ndt.__NAME__, "negative eigenvalues"))
                self.model_d[ALG_NAME_MDL].clear()
                return None, None

            # This steps(below) normalizes the eigenvectors by scaling them
            # with the inverse square root of eigenvalues and this "whitening"
            # transformation ensures that the transformed features
            # have unit variance.
            #
            try:
                eigvals_in = np.linalg.inv(np.diag(eigvals**(1/2)))

            except np.linalg.LinAlgError as e:
                print("Error: %s (line: %s) %s::%s: %s" %
                      (__FILE__, ndt.__LINE__, QDA.__CLASS_NAME__,
                       ndt.__NAME__, "singular matrix model is none"))
                self.model_d[ALG_NAME_MDL].clear()
                return None, None

            # calculation of transformation matrix
            #
            transf_mat = np.dot(eigvecs, eigvals_in)
            transformation_mats.append(transf_mat)

        # Storing the transformation matrices into the model dictionary
        #
        self.model_d[ALG_NAME_MDL][ALG_NAME_TRANS] = transformation_mats

        # compute a goodness of fit measure: use the average weighted
        # mean-square-error of the reconstructed data computed across the
        # entire data set using the transformation matrix we computed.
        #
        gsum = float(0.0)
        for i, d in enumerate(new_data):

            # iterate over the data in class i and calculate the norms
            #
            _sum = float(0.0)
            for v in d:

                # mean-squared-error of the reconstructed data
                # equivalent to np.linalg.norm(diff)**2
                #
                diff = np.dot((v-means[i]), transformation_mats[i])
                _sum += np.dot(diff.T, diff)

            # weight by the prior
            #
            gsum += self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS][i] * _sum

        # normalize
        #
        score = gsum / float(npts)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a list of numpy float matrices of feature vectors
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels
         posteriors: a list of float numpy vectors with the posterior
         probabilities for each class

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, QDA.__CLASS_NAME__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # check for model validity
        #
        if model[ALG_NAME_ALG] != QDA_NAME:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, QDA.__CLASS_NAME__, ndt.__NAME__,
                   "incorrect model name"))
            return None, None

        if not model[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, QDA.__CLASS_NAME__, ndt.__NAME__,
                   "model parameter is empty"))
            return None, None

        # get the number of features
        #
        ndim = data.data.shape[1]

        # transformation matrix that will transform the data and
        # means into a subspace
        #
        transformation_mats =  model[ALG_NAME_MDL][ALG_NAME_TRANS]
        means = model[ALG_NAME_MDL][ALG_NAME_MEANS]

        # calculate number of classes and transform means
        #
        num_classes = len(means)
        transformed_means = []
        for i in range(len(means)):
            m_new = np.dot(means[i], transformation_mats[i])
            transformed_means.append(m_new)

        # pre-compute the scaling term
        #
        scale = np.power(2 * np.pi, -ndim / 2)

        # loop over data
        #
        labels = []
        posteriors = []

        # loop over each matrix in data
        #
        for j in range(len(data.data)):

            # create temporary helper variables
            #
            evidence = 0
            post = []
            sample = data.data[j]

            # loop over number of classes
            #
            for k in range(num_classes):

                # manually compute the log likelihood
                # as a weighted Euclidean distance
                #
                transformed_sample = np.dot(sample, transformation_mats[k])

                # priors of class k
                #
                prior =  model[ALG_NAME_MDL][ALG_NAME_PRIORS][k]

                # mean of class k
                #
                trans_mean = transformed_means[k]

                # posterior calculation for sample j
                #
                sq_distance = np.dot((transformed_sample - trans_mean).T,
                                     (transformed_sample - trans_mean))
                gauss_likelihood = np.exp(-1/2 * sq_distance)
                posterior_contrib = gauss_likelihood * scale * prior
                evidence = evidence + posterior_contrib
                post.append(posterior_contrib)

            post = post/evidence

            # choose the class label with the highest posterior
            #
            labels.append(np.argmax(post))

            # save the posteriors
            #
            posteriors.append(post)

        # exit gracefully
        #
        return labels, posteriors
    #
    # end of method
#
# end of QDA

#------------------------------------------------------------------------------
# ML Tools Class: QLDA
#------------------------------------------------------------------------------

class QLDA:
    """
    class: QLDA

    description:
     This is a class that implements class_dependent
     Linear Discriminant Analysis (QLDA).
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        QLDA.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict(dict)

        # initialize a parameter dictionary
        #
        self.params_d = defaultdict(dict)

        # set the names
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_DISCRIMINANT
        self.model_d[ALG_NAME_MDL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary of covariance, means and priors
         score: f1 score

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, QLDA.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, QLDA.__CLASS_NAME__, ndt.__NAME__,
                   "Doesn't support training on pre-trained model"))
            return None, None

        # calculate number of classes
        #
        data = data.sort()
        uni_label = np.unique(data.labels)
        new_data =[]
        for i in range(len(uni_label)):

            # loop over all the labels
            #
            class_data = []
            for j in range(len(data.labels)):
                if uni_label[i] == data.labels[j]:
                    class_data.append(data.data[j])
            new_data.append(np.array(class_data))

        # set the number of classes
        #
        num_classes = len(new_data)

        # calculate number of data points
        #
        npts = 0
        for element in new_data:
            npts += len(element)

        # initialize an empty array to save the priors
        #
        priors = np.empty((0,0))

        # case: (ml) equal priors
        #
        mode_prior = self.params_d[ALG_NAME_PRIORS]
        if mode_prior == ALG_NAME_PRIORS_ML:

            # compute the priors for each class
            #
            priors = np.full(shape = num_classes,
                             fill_value = (float(1.0) / float(num_classes)))
            self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS] = priors

        # case: if the priors are based on occurrences
        #
        elif mode_prior == ALG_NAME_PRIORS_MAP:

            # create an array of priors by finding the the number of points
            # in each class and dividing by the total number of samples
            #
            for element in new_data:

                # appending the number of samples in
                # each element(class) to empty array
                #
                priors = np.append(priors, len(element))

            # final calculation of priors
            #
            sum = float(1.0) / float(npts)
            priors = priors * sum
            self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS] = priors

        # case: unknown parameter setting
        #
        else:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, QLDA.__CLASS_NAME__, ndt.__NAME__,
                   "unknown value for priors"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # calculate the means of each class:
        # note these are a list of numpy vectors
        #
        means = []
        for elem in new_data:
            means.append(np.mean(elem, axis = 0))
        self.model_d[ALG_NAME_MDL][ALG_NAME_MEANS] = means

        # calculate the global mean
        #
        mean_glob = np.mean(np.vstack(new_data), axis = 0)

        # calculate s_b and s_w.
        # we need to calculate them for the transformation matrix
        #
        n_features = new_data[0].shape[1]

        # number of the eigenvectors need to be chosen
        # it is equal to the num_class minus 1
        #
        l = (len(new_data)) - 1
        transformation_mats = []

        # initialize between class scatter
        #
        sb = np.zeros((n_features, n_features))
        for i in range(len(new_data)):

            # between class scatter calculation
            #
            mean_diff = (means[i] - mean_glob).reshape(n_features, 1)
            sb += len(new_data)*(mean_diff).dot(mean_diff.T)

        # within class scatter and final covariance for each class
        #
        for i, d in enumerate(new_data):

            # calculation of within class scatter for each class
            #
            sw = nct.compute(d, \
            ctype = self.params_d[nct.PRM_CTYPE],
            center = self.params_d[nct.PRM_CENTER],
            scale = self.params_d[nct.PRM_SCALE])

            # check singularity
            #
            try:
                sw_in = np.linalg.inv(sw)

            except np.linalg.LinAlgError:
                print("Error: %s (line: %s) %s::%s: %s" %
                      (__FILE__, ndt.__LINE__, QLDA.__CLASS_NAME__,
                       ndt.__NAME__, "singular matrix"))
                self.model_d[ALG_NAME_MDL].clear()
                return None, None

            # calculation of sw^-1*sb for each class
            #
            transformation_base = sw_in.dot(sb)

            # eigen vector and eigen value decomposition for each class
            #
            eigvals, eigvecs = np.linalg.eig(transformation_base)

            # sorted eigenvalues and eigvecs and choose
            # the first l-1 columns from eigenvalues
            # and eigenvectors
            #
            sorted_indexes = eigvals.argsort () [::-1]
            eigvals = eigvals[sorted_indexes[0:l]]
            eigvecs = eigvecs[:,sorted_indexes[0:l]]
            if any(eigvals < 0):
                print("Error: %s (line: %s) %s::%s: %s" %
                      (__FILE__, ndt.__LINE__, QLDA.__CLASS_NAME__,
                       ndt.__NAME__, "negative eigenvalues"))
                self.model_d[ALG_NAME_MDL].clear()
                return None, None

            # This steps(below) normalizes the eigenvectors by scaling
            # them with the inverse square root of eigenvalues.
            # This "whitening" transformation ensures that the transformed
            # features have unit variance.
            #
            try:
                eigvals_in = np.linalg.inv(np.diag(eigvals**(1/2)))

            except np.linalg.LinAlgError:
                print("Error: %s (line: %s) %s::%s: %s" %
                      (__FILE__, ndt.__LINE__, QLDA.__CLASS_NAME__,
                       ndt.__NAME__, "singular matrix model"))
                self.model_d[ALG_NAME_MDL].clear()
                return None, None

            # calculation of transformation matrix
            #
            transf_mat = np.dot(eigvecs, eigvals_in)
            transformation_mats.append(transf_mat)

        self.model_d[ALG_NAME_MDL][ALG_NAME_TRANS] = transformation_mats

        # compute a goodness of fit measure: use the average weighted
        # mean-square-error of the reconstructed data computed across the
        # entire data set using the transformation matrix we computed.
        #
        gsum = float(0.0)
        for i, d in enumerate(new_data):

            # iterate over the data in class i and calculate the norms
            #
            _sum = float(0.0)
            for v in d:

                # mean-squared-error of the reconstructed data
                # equivalent to np.linalg.norm(diff)**2
                #
                diff = np.dot((v-means[i]), transformation_mats[i])
                _sum += np.dot(diff.T, diff)

            # weight by the prior
            #
            gsum += self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS][i] * _sum
        score = gsum / float(npts)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a list of numpy float matrices of feature vectors
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels
         posteriors: a list of float numpy vectors with the posterior
         probabilities for each class

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, QLDA.__CLASS_NAME__,
                   ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # check for model validity
        #
        if model[ALG_NAME_ALG] != QLDA_NAME:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, QLDA.__CLASS_NAME__,
                   ndt.__NAME__, "incorrect model name"))
            return None, None

        if not model[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, QLDA.__CLASS_NAME__, ndt.__NAME__,
                   "model parameter is empty"))
            return None, None

        # get the number of features
        #
        ndim = data.data.shape[1]

        # transformation matrix that will transform the data and
        # means into a subspace
        #
        transformation_mats =  model[ALG_NAME_MDL][ALG_NAME_TRANS]
        means = model[ALG_NAME_MDL][ALG_NAME_MEANS]

        # calculate number of classes and transform means
        #
        num_classes = len(means)
        transformed_means = []
        for i in range(len(means)):
            m_new = np.dot(means[i], transformation_mats[i])
            transformed_means.append(m_new)

        # pre-compute the scaling term
        #
        scale = np.power(2 * np.pi, -ndim / 2)

        # loop over data
        #
        labels = []
        posteriors = []

        # loop over each matrix in data
        #
        for j in range(len(data.data)):

            # create temporary helper variables
            #
            evidence = 0
            post = []
            sample = data.data[j]

            # loop over number of classes
            #
            for k in range(num_classes):


                # manually compute the log likelihood
                # as a weighted Euclidean distance
                #
                transformed_sample = np.dot(sample, transformation_mats[k])

                # priors of class k
                #
                prior =  model[ALG_NAME_MDL][ALG_NAME_PRIORS][k]

                # mean of class k
                #
                trans_mean = transformed_means[k]

                # posterior calculation for sample j
                #
                sq_distance = np.dot((transformed_sample - trans_mean).T,
                                     (transformed_sample - trans_mean))
                gauss_likelihood = np.exp(-1/2 * sq_distance)
                posterior_contrib = gauss_likelihood * scale * prior
                evidence = evidence + posterior_contrib
                post.append(posterior_contrib)

            # convert the likelihoods to posteriors
            #
            post = post/evidence

            # choose the class label with the highest posterior
            #
            labels.append(np.argmax(post))

            # save the posteriors
            #
            posteriors.append(post)

        # exit gracefully
        #
        return labels, posteriors
    #
    # end of method
#
# end of QLDA

#------------------------------------------------------------------------------
# ML Tools Class: NB
#------------------------------------------------------------------------------

class NB:
    """
    class: NB

    description:
     This is a class that implements Naive Bayes (NB).
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        NB.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = NB_IMPLS[0]
        self.model_d[ALG_NAME_MDL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary of covariance, means and priors
         score: f1 score

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, NB.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, NB.__CLASS_NAME__, ndt.__NAME__,
                   "Doesn't support training on pre-trained model"))
            return None, None

        # calculate number of classes
        #
        data = data.sort()
        uni_label = np.unique(data.labels)
        new_data = []

        for i in range(len(uni_label)):
            class_data = []
            for j in range(len(data.labels)):
                if uni_label[i] == data.labels[j]:
                    class_data.append(data.data[j])
            new_data.append(np.array(class_data))

        # set the number of classes
        #
        num_classes = len(new_data)

        # calculate number of data points
        #
        npts = 0
        for element in new_data:
            npts += len(element)

        # case: (ml) equal priors
        #
        mode_prior = self.params_d[ALG_NAME_PRIORS]
        if mode_prior == ALG_NAME_PRIORS_ML:

            # compute the priors for each class
            #
            priors = np.full(shape = num_classes,
                             fill_value = (float(1.0) / float(num_classes)))

        # if the priors are based on occurrences
        #
        elif mode_prior == ALG_NAME_PRIORS_MAP:

            # create an array of priors by finding the the number of points
            # in each class and dividing by the total number of samples
            #
            for element in new_data:

                # appending the number of samples in
                # each element(class) to empty array
                #
                priors = np.append(priors, len(element))

            # final calculation of priors
            #
            sum = float(1.0) / float(npts)
            priors = priors * sum

        else:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, NB.__CLASS_NAME__, ndt.__NAME__,
                   "unknown value for priors"))
            return None

        # make the final data
        #
        f_data = np.vstack((new_data))

        # getting the labels
        #
        labels = []
        for i in range(len(new_data)):

            # loop over all the data
            #
            for j in range(len(new_data[i])):
                labels.append(i)

        # save the labels
        #
        labels = np.array(labels)

        # use the implementation from the parameters
        #
        # NOTE: this is for backward compatability
        if ALG_NAME_IMP in self.params_d:
            impl = self.params_d[ALG_NAME_IMP]
        else:
            impl = self.model_d[ALG_NAME_IMP]

        if impl not in NB_IMPLS:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, NB.__CLASS_NAME__, ndt.__NAME__,
                   "unknown implementation"))
            return None

        # check for the implementation name
        #
        if impl == IMP_NAME_SKLEARN:

            # fit the model
            #
            self.model_d[ALG_NAME_MDL] = \
                GaussianNB(priors = priors).fit(f_data, labels)
            self.model_d[ALG_NAME_IMP] = impl

        # prediction
        #
        ypred = self.model_d[ALG_NAME_MDL].predict(f_data)

        # score calculation using auc
        #
        score = f1_score(labels, ypred, average = "macro")

        # write to file
        #
        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, NB.__CLASS_NAME__, ndt.__NAME__))

        # check if model is none
        #
        data = np.array(data.data)
        if model is None:
            model = self.model_d

        # compute the labels
        #
        p_labels = model[ALG_NAME_MDL].predict(data)

        # posterior calculation
        #
        posteriors = model[ALG_NAME_MDL].predict_proba(data)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of class (NB)

#------------------------------------------------------------------------------
# ML Tools Class: GMM
#------------------------------------------------------------------------------

class GMM:
    """
    class: GMM

    description:
     This is a class that implements classification using Gaussian Mixture
     Models (GMM). A separate GMM is trained for each class and the final
     decision is based on the posterior probability computed via Bayes rule.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the GMMClassifier class.
        """

        # Set the class name (for informational/debug purposes)
        #
        GMM.__CLASS_NAME__ = self.__class__.__name__

        # Initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # Set the model; we will store a dictionary of trained GMMs per class
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_DISCRIMINANT
        self.model_d[ALG_NAME_MDL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    # Computational methods: train / predict
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: an MLToolsData object containing:
               - data.data: a list/numpy array of feature vectors
               - data.labels: a list/numpy array of labels
         write_train_labels: a boolean indicating whether to write the train
                             predictions to file
         fname_train_labels: the filename for the train labels output

        return:
         model: a dictionary of GMM models (and additional parameters)
         score: the macro-averaged f1 score on the training data

        description:
         This method trains a separate GMM for each class using the training
         data. It computes the class priors, fits the models, predicts on the
         training data, calculates the f1 score, and (optionally) writes out
         the predicted labels.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, GMM.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, GMM.__CLASS_NAME__, ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # identify unique labels and group the data by class
        #
        data = data.sort()
        uni_label = np.unique(data.labels)
        new_data = []  # list to hold numpy arrays for each class's data

        for ul in uni_label:
            class_data = []
            for i in range(len(data.labels)):
                if data.labels[i] == ul:
                    class_data.append(data.data[i])
            new_data.append(np.array(class_data))

        num_classes = len(new_data)

        # get the number of mixture components from the parameters
        #
        n_components= int(self.params_d[ALG_NAME_NCMP])

        if n_components == -1:
            n_components = new_data[0].shape[1]

        if not(0 < n_components <= new_data[0].shape[1]):
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, GMM.__CLASS_NAME__, ndt.__NAME__,
                   "features out of range"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # calculate number of data points
        #
        npts = 0
        for element in new_data:
            npts += len(element)

        # initialize an empty array to save the priors
        #
        priors = np.empty((0,0))

        # determine the prior mode from parameters
        #
        mode_prior = self.params_d[ALG_NAME_PRIORS]

        if mode_prior == ALG_NAME_PRIORS_ML:
            # Equal priors for each class
            priors = np.full(shape = num_classes,
                             fill_value = (1.0 / num_classes))

        elif mode_prior == ALG_NAME_PRIORS_MAP:
            # Priors based on the relative frequency of each class
            counts = np.array([len(class_data) for class_data in new_data])
            priors = counts / float(npts)

        else:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, GMM.__CLASS_NAME__, ndt.__NAME__,
                   "unknown value for priors"))
            self.model_d[ALG_NAME_MDL].clear()
            return None, None

        # save the priors in the model dictionary
        #
        self.model_d[ALG_NAME_MDL][ALG_NAME_PRIORS] = priors
        random_state = int(self.params_d[ALG_NAME_RANDOM])

        # train a GMM for each class
        #
        class_models = {}
        for idx, class_data in enumerate(new_data):
            gm = GaussianMixture(n_components=n_components,
                                 random_state=random_state)
            gm.fit(class_data)
            class_models[idx] = gm

        # save the trained class models and class labels
        #
        self.model_d[ALG_NAME_MDL][ALG_NAME_CMODELS] = class_models
        self.model_d[ALG_NAME_MDL][ALG_NAME_CLABELS] = uni_label

        # prepare a full data matrix and labels for computing
        # the training score
        #
        f_data = np.vstack(new_data)
        labels = []
        for idx, class_data in enumerate(new_data):
            labels.extend([idx] * len(class_data))
        labels = np.array(labels)

        # prediction on the training data via a
        # posterior probability calculation:
        #  for each sample, compute for each class:
        #   likelihood = exp(log-likelihood) * prior
        # then choose the class with the highest resulting value.
        #
        ypred = []
        posteriors = []
        for x in f_data:
            likelihoods = []

            # loop over the number of classes
            #
            for idx in range(num_classes):

                # compute log likelihood for sample x under the class's GMM
                #
                log_likelihood = \
                    class_models[idx].score_samples(x.reshape(1, -1))[0]
                likelihood = np.exp(log_likelihood) * priors[idx]
                likelihoods.append(likelihood)

            # compute the ouptut labels
            #
            likelihoods = np.array(likelihoods)
            pred_label = np.argmax(likelihoods)
            ypred.append(pred_label)

            # compute normalized posterior probabilities
            #
            post = likelihoods / np.sum(likelihoods)
            posteriors.append(post)

        # get the predicted labels
        #
        ypred = np.array(ypred)

        # compute the macro-averaged f1 score on the training data
        #
        score = f1_score(labels, ypred, average = "macro")

        # optionally, write the predicted training labels to file using the
        # MLToolsData.write() method
        #
        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: an MLToolsData object containing feature vectors
               (each row is a vector)
         model: a GMM model dictionary (if None, the internal model is used)

        return:
         p_labels: a numpy array of predicted labels
         posteriors: a numpy array of posterior probabilities (one row/sample)

        description:
         This method predicts the class labels for the given data using the
         trained GMM models. It computes the posterior probabilities for
         each sample and selects the class with the highest probability.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, GMM.__CLASS_NAME__, ndt.__NAME__))

        # check if model is none
        #
        data = np.array(data.data)
        if model is None:
            model = self.model_d

        # retrieving the priors, class models and class labels
        # from the trained model
        #
        priors = model[ALG_NAME_MDL][ALG_NAME_PRIORS]
        class_models = model[ALG_NAME_MDL][ALG_NAME_CMODELS]
        class_labels = model[ALG_NAME_MDL][ALG_NAME_CLABELS]
        num_classes = len(class_models)
        # Posterior calculation
        #
        p_labels = []
        posteriors = []
        for single_data in data:

            # likelihoods calculation
            #
            likelihoods = []
            for idx in range(num_classes):

                # log Likelihood calculation for each sample
                #
                log_likelihood = class_models[idx].score_samples(
                    single_data.reshape(1, -1))[0]
                likelihood = np.exp(log_likelihood) * priors[idx]
                likelihoods.append(likelihood)

            # capture the likelihoods
            #
            likelihoods = np.array(likelihoods)

            # predicting labels based on maximum likelihood
            #
            pred_label = np.argmax(likelihoods)
            p_labels.append(class_labels[pred_label])
            print(likelihoods, np.sum(likelihoods))
            post = likelihoods / np.sum(likelihoods)
            posteriors.append(post)

        # capture the labels
        #
        p_labels = np.array(p_labels)
        posteriors = np.array(posteriors)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of class (GMM)

#******************************************************************************
#
# Section 2: nonparametric models
#
#******************************************************************************

#------------------------------------------------------------------------------
# ML Tools Class: KNN
#------------------------------------------------------------------------------

class KNN:
    """
    class: KNN

    description:
     This is a class that implements KNN.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        KNN.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d[ALG_NAME_MDL] = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_SKLEARN
        self.model_d[ALG_NAME_MDL] = {}
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary of covariance, means and priors
         score: f1 score

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, KNN.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, KNN.__CLASS_NAME__, ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # fit the model
        #
        n = int(self.params_d[ALG_NAME_NNEARN])
        self.model_d[ALG_NAME_MDL] = \
            KNeighborsClassifier(n_neighbors = n).fit(samples, labels)

        # prediction
        #
        ypred = self.model_d[ALG_NAME_MDL].predict(samples)

        # score calculation using auc ( f1 score )
        #
        score = f1_score(labels, ypred, average = "macro")

        # write to file
        #
        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, KNN.__CLASS_NAME__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        samples = np.array(data.data)

        p_labels = model[ALG_NAME_MDL].predict(samples)

        # posterior calculation
        #
        posteriors = model[ALG_NAME_MDL].predict_proba(samples)

        return p_labels, posteriors
    #
    # end of method
#
# end of class (KNN)

#------------------------------------------------------------------------------
# ML Tools Class: KMEANS
#------------------------------------------------------------------------------

class KMEANS:
    """
    class: KMEANS

    description:
     This is a class that implements KMEANS.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        KMEANS.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_SKLEARN
        self.model_d[ALG_NAME_MDL] = {}
        
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary of covariance, means and priors
         score: silhouette score

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, KMEANS.__CLASS_NAME__,
                   ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, KMEANS.__CLASS_NAME__, ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # fit the model
        #
        imp = self.params_d[ALG_NAME_IMP]
        n_cluster = int(self.params_d[ALG_NAME_NCLUSTERS])
        random_state =int(self.params_d[ALG_NAME_RANDOM])
        n_init = int(self.params_d[ALG_NAME_NINIT])
        m_iter = int(self.params_d[ALG_NAME_MAXITER])

        self.model_d[ALG_NAME_MDL] = KMeans(n_clusters=n_cluster,
        random_state = random_state,
        n_init = n_init,
        max_iter = m_iter).fit(samples)

        predicted_labels = self.model_d[ALG_NAME_MDL].labels_

        score = silhouette_score(samples, predicted_labels)

        if write_train_labels:
            data.write(oname = fname_train_labels, label = predicted_labels)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, KMEANS.__CLASS_NAME__,
                   ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d
        posteriors = []
        data = np.array(data.data)
        p_labels = model[ALG_NAME_MDL].predict(data)

        # cluster centers
        #
        centers = model[ALG_NAME_MDL].cluster_centers_

        # posteriors calculation
        #
        for d in data:
            dis_c = []
            count = 0
            for c in centers:
                dis = np.linalg.norm(d - c)
                dis_c.append(dis)
                count = count + dis
            posteriors.append(dis_c/count)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of class (KMEANS)

#------------------------------------------------------------------------------
# ML Tools Class: RNF
#------------------------------------------------------------------------------

class RNF:
    """
    class: RNF

    description:
     This is a class that implements RNF.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        RNF.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_SKLEARN
        self.model_d[ALG_NAME_MDL] = {}
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary of covariance, means and priors
         score: f1 score

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, RNF.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, RNF.__CLASS_NAME__, ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # fit the model
        #
        n_estimators = int(self.params_d[ALG_NAME_NEST])
        max_depth = int(self.params_d[ALG_NAME_MAXDEPTH])
        criterion = self.params_d[ALG_NAME_CRITERION]
        random_state = int(self.params_d[ALG_NAME_RANDOM])
        self.model_d[ALG_NAME_MDL] = \
            RandomForestClassifier(n_estimators = n_estimators,
                                   max_depth = max_depth,
                                   criterion = criterion,
                                   random_state= random_state).fit(samples,
                                                                   labels)

        # prediction
        #
        ypred = self.model_d[ALG_NAME_MDL].predict(samples)

        # score calculation using auc
        #
        score = f1_score(labels, ypred, average = "macro")

        # write to file
        #
        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, RNF.__CLASS_NAME__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        samples = np.array(data.data)

        p_labels = model[ALG_NAME_MDL].predict(samples)

        # posterior calculation
        #
        posteriors = model[ALG_NAME_MDL].predict_proba(samples)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of class (RNF)

#------------------------------------------------------------------------------
# ML Tools Class: SVM
#------------------------------------------------------------------------------

class SVM:
    """
    class: SVM

    description:
     This is a class that implements SVM.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        SVM.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_SKLEARN
        self.model_d[ALG_NAME_MDL] = {}
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary of covariance, means and priors
         score: f1 score

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, SVM.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, SVM.__CLASS_NAME__, ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # fit the model
        #
        c = float(self.params_d[ALG_NAME_C])
        gamma =float(self.params_d[ALG_NAME_GAMMA])
        kernel = self.params_d[ALG_NAME_KERNEL]

        self.model_d[ALG_NAME_MDL] = SVC(C = c,
                        gamma = gamma,
                        kernel = kernel,
                        probability=True).fit(samples, labels)

        # prediction
        #
        ypred = self.model_d[ALG_NAME_MDL].predict(samples)

        # score calculation using auc ( f1 score)
        #
        score = f1_score(labels, ypred, average="macro")

        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, SVM.__CLASS_NAME__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        samples = np.array(data.data)

        p_labels = model[ALG_NAME_MDL].predict(samples)

        # posterior calculation
        #
        posteriors = model[ALG_NAME_MDL].predict_proba(samples)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of class (SVM)

#******************************************************************************
#
# Section 3: neural network-based models
#
#******************************************************************************

#------------------------------------------------------------------------------
# ML Tools Class: MLP
#------------------------------------------------------------------------------

class MLP:
    """
    class: MLP

    description:
     This is a class that implements MLP.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        MLP.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_SKLEARN
        self.model_d[ALG_NAME_MDL] = {}
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary of covariance, means and priors
         score: f1 score

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, MLP.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, MLP.__CLASS_NAME__, ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # fit the model
        #
        imp = self.params_d[ALG_NAME_IMP]
        h_s = int(self.params_d[ALG_NAME_HSIZE])
        act = self.params_d[ALG_NAME_ACT]
        b_s = self.params_d[ALG_NAME_BSIZE]
        sol = self.params_d[ALG_NAME_SOLVER]
        lr = self.params_d[ALG_NAME_LR]
        lr_init = float(self.params_d[ALG_NAME_LRINIT])
        e_stop = bool(self.params_d[ALG_NAME_ESTOP])
        sh = bool(self.params_d[ALG_NAME_SHUFFLE])
        val= float(self.params_d[ALG_NAME_VALF])
        m = float(self.params_d[ALG_NAME_MOMENTUM])
        r_state = int(self.params_d[ALG_NAME_RANDOM])
        m_iter = int(self.params_d[ALG_NAME_MAXITER])

        self.model_d[ALG_NAME_MDL] = \
            MLPClassifier(hidden_layer_sizes = (h_s,),
                          activation = act,
                          solver = sol,
                          batch_size = b_s,
                          learning_rate = lr,
                          learning_rate_init = lr_init,
                          shuffle = sh,
                          random_state = r_state,
                          momentum = m,
                          early_stopping = e_stop,
                          validation_fraction=val,
                          max_iter=m_iter).fit(samples, labels)

        # prediction
        #
        ypred = self.model_d[ALG_NAME_MDL].predict(samples)

        # score calculation using auc ( f1 score )
        #
        score = f1_score(labels, ypred, average="macro")

        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, MLP.__CLASS_NAME__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        samples = np.array(data.data)

        p_labels = model[ALG_NAME_MDL].predict(samples)

        # posterior calculation
        #
        posteriors = model[ALG_NAME_MDL].predict_proba(samples)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method

#
# end of class (MLP)

#------------------------------------------------------------------------------
# ML Tools Class: RBM
#------------------------------------------------------------------------------

class RBM:
    """
    class: RBM

    description:
     This is a class that implements RBM.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        RBM.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_SKLEARN
        self.model_d[ALG_NAME_MDL] = []
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a dictionary of covariance, means and priors
         score: f1 score

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, RBM.__CLASS_NAME__, ndt.__NAME__))

        if self.model_d[ALG_NAME_MDL]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__, RBM.__CLASS_NAME__, ndt.__NAME__,
                   "doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # fit the model
        #
        n_comp = int(self.params_d[ALG_NAME_NCMP])
        n_iter = int(self.params_d[RBM_NAME_NITER])
        random_state = int(self.params_d[ALG_NAME_RANDOM])
        lr = float(self.params_d[ALG_NAME_LR])
        b_size = int(self.params_d[ALG_NAME_BSIZE])
        verbose = bool(self.params_d[RBM_NAME_VERBOSE])
        classifier = str(self.params_d[RBM_NAME_CLASSIFIER])
        rbm= BernoulliRBM(n_components=n_comp, learning_rate=lr,
                          batch_size=b_size, n_iter=n_iter,
                          verbose=verbose, random_state=random_state)
        
        # set classifier's param_d
        #
        ALGS[classifier].params_d = self.params_d
        
        # train the classifier model
        #
        ALGS[classifier].train(data = data, write_train_labels = False,
              fname_train_labels = "train_labels.csv")
        
        self.model_d[ALG_NAME_MDL] = \
            Pipeline(steps = [
                ('rbm', rbm),
                ('classifier', ALGS[classifier].model_d[ALG_NAME_MDL])
            ]).fit(samples, labels)

        # prediction
        #
        ypred = self.model_d[ALG_NAME_MDL].predict(samples)

        # score calculation using auc ( f1 score )
        #
        score = f1_score(labels, ypred, average="macro")

        # write to file
        #
        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, RBM.__CLASS_NAME__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # compute the labels
        #
        samples = np.array(data.data)
        p_labels = model[ALG_NAME_MDL].predict(samples)

        # posterior calculation
        #
        posteriors = model[ALG_NAME_MDL].predict_proba(samples)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of class (RBM)

#------------------------------------------------------------------------------
# ML Tools Class: TRANSFORMER
#------------------------------------------------------------------------------

class TRANSFORMER:
    """
    class: TRANSFORMER

    description:
     This is a class that implements TRANSFORMER.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        TRANSFORMER.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_PYTORCH
        self.model_d[ALG_NAME_MDL] = {}

    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a PyTorch state_dict containing the model
         error: training error rate

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, TRANSFORMER.__CLASS_NAME__,
                   ndt.__NAME__))

        # get the samples
        #
        samples =  np.array(data.data)

        # getting the labels
        #
        labels = data.labels

        # get the parameters
        #
        self.lr = float(self.params_d[ALG_NAME_LR])
        self.batch_size = int(self.params_d[ALG_NAME_BSIZE])
        self.embed_size = int(self.params_d[QML_NAME_EMBED_SIZE])
        self.nheads = int(self.params_d[QML_NAME_NHEADS])
        self.num_layers = int(self.params_d[ALG_NAME_NLAYERS])
        self.mlp_dim = int(self.params_d[QML_NAME_MLP_DIM])
        self.dropout = float(self.params_d[QML_NAME_DROPOUT])
        self.random_state = int(self.params_d[ALG_NAME_RANDSTATE])

        # set the random seed
        #
        dbgl_g.set_seed(value=self.random_state)

        # create the model
        #
        trans_model = ntt.NEDCTransformer(
            input_dim = samples.shape[1],
            num_classes = len(np.unique(labels)),
            d_model = self.embed_size, nhead=self.nheads,
            num_layers = self.num_layers, dim_feedforward = self.mlp_dim,
            dropout = self.dropout)

        # check if the model is pre-trained
        #
        if self.model_d[ALG_NAME_MDL]:
            if dbgl_g > ndt.NONE:
                print("Info: %s (line: %s) %s::%s: %s" %
                      (__FILE__, ndt.__LINE__, TRANSFORMER.__CLASS_NAME__,
                       ndt.__NAME__, "loading a pre-trained model"))

            # load the model's weights
            #
            trans_model.load_state_dict(self.model_d[ALG_NAME_MDL])


        else:
            if dbgl_g > ndt.NONE:
                print("Info: %s (line: %s) %s::%s: %s" %
                      (__FILE__, ndt.__LINE__, TRANSFORMER.__CLASS_NAME__,
                       ndt.__NAME__, "training a new model"))

        # move the model to the default device
        # CPU/GPU based on their availability
        #
        trans_model = trans_model.to_device(trans_model)

        # assign an initial NEDCTransformer object
        #
        self.model_d[ALG_NAME_MDL] = trans_model

        # get the epochs
        #
        self.epochs = int(self.params_d[QML_NAME_EPOCH])

        # initialize the accuracy measures
        #
        accuracies = []

        # get the corss entropy loss function and ADAM optimizer
        #
        criterion = self.model_d[ALG_NAME_MDL]\
        .get_cross_entropy_loss_function()
        optimizer = self.model_d[ALG_NAME_MDL]\
        .get_adam_optimizer(lr=self.lr)

        # train the model
        #
        for epoch in range(self.epochs):

            # set the model to train mode
            #
            self.model_d[ALG_NAME_MDL].train()

            # initialize the correct and total
            #
            train_correct = 0
            train_total = 0

            # get random indices for batches
            #
            rnd_indices = np.arange(len(samples))
            np.random.shuffle(rnd_indices)

            # initialize the training loss list
            #
            train_losses = []

            # iterate over the batches
            #
            for batch_start in range(0, len(samples), self.batch_size):

                # get the current batch
                #
                batch_indices = rnd_indices[batch_start:
                                    batch_start + self.batch_size]
                current_samples = samples[batch_indices]
                current_labels = labels[batch_indices]

                # zero the gradients
                #
                optimizer.zero_grad()

                # get output from the model's forward pass
                #
                output = self.model_d[ALG_NAME_MDL](current_samples)

                # calculate the loss
                #
                loss = criterion(output,
                                 self.model_d[ALG_NAME_MDL]\
                                 .to_tensor(current_labels))

                # append the loss to the list
                #
                train_losses.append(loss.item())

                # run backward propagation algorithm
                #
                loss.backward()

                # update the weights
                #
                optimizer.step()

                # detach the output and convert it to numpy array
                #
                output = output.cpu().detach().numpy()

                # get the predicted labels
                #
                predicted = np.argmax(output, 1)

                # calculate the accuracy
                #
                train_total += len(current_labels)
                train_correct += np.sum(predicted == current_labels)

            # print the epoch and training loss
            #
            print(f"Epoch: {epoch + 1}/{self.epochs}, "
                  f"Training Loss: {np.mean(train_losses):{ALG_FMT_DEC}}")

            # calculate the accuracy for one batch
            #
            batch_accuracy =  train_correct / train_total

            # append the accuracy to the list
            #
            accuracies.append(batch_accuracy)

        # get the average of training accuracies
        #
        accuracy = np.mean(accuracies)

        # calculate the error rate
        #
        error_rate = 1 - accuracy
        print(f"Training error rate:  {error_rate:{ALG_FMT_DEC}}")

        # convert the model to state_dict, so that it can be saved by using
        # Alg.save_model()
        #
        state_dict = self.model_d[ALG_NAME_MDL].state_dict()

        # assign the model to the model_d['model']
        #
        self.model_d[ALG_NAME_MDL] = state_dict

        # get the predicted labels
        #
        ypred, _ = self.predict(data=data)

        # if write_train_labels is True, write the labels to the file
        #
        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, error_rate
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, TRANSFORMER.__CLASS_NAME__,
                   ndt.__NAME__))
        # get the samples
        #
        samples = np.array(data.data)

        # get the parameters
        #
        self.lr = float(self.params_d[ALG_NAME_LR])
        self.batch_size = int(self.params_d[ALG_NAME_BSIZE])
        self.embed_size = int(self.params_d[QML_NAME_EMBED_SIZE])
        self.nheads = int(self.params_d[QML_NAME_NHEADS])
        self.num_layers = int(self.params_d[ALG_NAME_NLAYERS])
        self.mlp_dim = int(self.params_d[QML_NAME_MLP_DIM])
        self.dropout = float(self.params_d[QML_NAME_DROPOUT])

        # create the model
        #
        trans_model = ntt.NEDCTransformer(input_dim=samples.shape[1],
                                          num_classes=data.num_of_classes,
                                          d_model=self.embed_size,
                                          nhead=self.nheads,
                                          num_layers=self.num_layers,
                                          dim_feedforward=self.mlp_dim,
                                          dropout=self.dropout)
        # move the model to the default device
        # CPU/GPU based on their availability
        #
        trans_model = trans_model.to_device(trans_model)

        # load the model's weights
        #
        trans_model.load_state_dict(self.model_d[ALG_NAME_MDL])

        # create an empty list to store the predicted labels
        #
        p_labels = []

        # set the model to evaluation mode
        #
        trans_model.eval()

        # torch.no_grad() is used to disable the gradient calculation
        # because we are only predicting the labels
        #
        with torch.no_grad():

            # iterate over the batches
            #
            for batch in range(0, len(samples), self.batch_size):

                # get the current batch
                #
                current_samples = samples[batch:batch + self.batch_size]

                # get the output from the model
                #
                output = trans_model(current_samples)

                # detach the output and convert it to numpy array
                #
                output = output.cpu().detach().numpy()

                # get the predicted labels
                #
                predicted = np.argmax(output, 1)

                # append the predicted labels to the list, using extend
                # method because predicted is a list, so we need to
                # extend it to p_labels
                #
                p_labels.extend(predicted)

        # posterior do not apply to TRANSFORMER
        #
        posteriors = None

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of class (TRANSFORMER)

#******************************************************************************
#
# Section 4: quantum computing-based models
#
#******************************************************************************

#------------------------------------------------------------------------------
# ML Tools Class: QSVM
#------------------------------------------------------------------------------

class QSVM:
    """
    class: QSVM

    description:
     This is a class that implements QSVM.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        QSVM.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_QISKIT
        self.model_d[ALG_NAME_MDL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a PyTorch state_dict containing the model
         error: training error rate

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, QSVM.__CLASS_NAME__, ndt.__NAME__))

        # get the samples
        #
        samples =  np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # get the parameters
        #
        self.mdl_name = str(self.model_d[ALG_NAME_ALG])
        self.provider_name = str(self.params_d[QML_NAME_PROVIDER])
        self.hardware = str(self.params_d[QML_NAME_HARDWARE])
        self.encoder = str(self.params_d[QML_NAME_ENCODER])
        self.entanglement = str(self.params_d[QML_NAME_ENTANGLEMENT])
        self.n_qubits = int(self.params_d[QML_NAME_NQUBITS])
        self.feat_reps = int(self.params_d[QML_NAME_FEAT_REPS])
        self.shots = int(self.params_d[QML_NAME_SHOTS])
        self.kernel_name = str(self.params_d[ALG_NAME_KERNEL])

        # create the model
        #
        qsvm_model = nqt.QML(model_name=self.mdl_name,
                             provider_name=
                             self.provider_name,
                             hardware=self.hardware,
                             encoder_name=self.encoder,
                             entanglement=
                             self.entanglement,
                             n_qubits=self.n_qubits,
                             featuremap_reps=self.feat_reps,
                             shots=self.shots,
                             kernel_name=self.kernel_name)

        # get the trained model
        #
        self.model_d[ALG_NAME_MDL] = qsvm_model.fit(samples, labels)

        # get the error rate for the training samples
        #
        error_rate, ypred = qsvm_model.score(samples, labels)

        # if write_train_labels is True, write the labels to the file
        #
        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, error_rate
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, QSVM.__CLASS_NAME__, ndt.__NAME__))

        # get the samples
        #
        samples = np.array(data.data)

        # get the trained model
        #
        model = model[ALG_NAME_MDL]

        # get the predicted labels
        #
        p_labels = model.predict(samples)

        # currently QSVM does not support posterior calculation
        #
        posteriors = None

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of class (QSVM)

#------------------------------------------------------------------------------
# ML Tools Class: QNN
#------------------------------------------------------------------------------

class QNN:
    """
    class: QNN

    description:
     This is a class that implements QNN.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        QNN.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_QISKIT
        self.model_d[ALG_NAME_MDL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a PyTorch state_dict containing the model
         error: training error rate

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, QNN.__CLASS_NAME__, ndt.__NAME__))

        # get the samples
        #
        samples =  np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # get the parameters
        #
        self.mdl_name = str(self.model_d[ALG_NAME_ALG])
        self.provider_name = str(self.params_d[QML_NAME_PROVIDER])
        self.hardware = str(self.params_d[QML_NAME_HARDWARE])
        self.encoder = str(self.params_d[QML_NAME_ENCODER])
        self.entanglement = str(self.params_d[QML_NAME_ENTANGLEMENT])
        self.n_qubits = int(self.params_d[QML_NAME_NQUBITS])
        self.feat_reps = int(self.params_d[QML_NAME_FEAT_REPS])

        self.ansatz = str(self.params_d[QML_NAME_ANSATZ])
        self.ansatz_reps = int(self.params_d[QML_NAME_ANSATZ_REPS])
        self.optim_name = str(self.params_d[QML_NAME_OPTIM])
        self.optim_max_steps = int(self.params_d[QML_NAME_MAXSTEPS])
        self.meas_type = str(self.params_d[QML_NAME_MEAS_TYPE])

        # create the model
        #
        qnn_model = nqt.QML(model_name=self.mdl_name,
                            provider_name=
                            self.provider_name,
                            hardware=self.hardware,
                            encoder_name=self.encoder,
                            entanglement=
                            self.entanglement,
                            n_qubits=self.n_qubits,
                            featuremap_reps=self.feat_reps,
                            ansatz=self.ansatz,
                            ansatz_reps=self.ansatz_reps,
                            optim_name=self.optim_name,
                            optim_max_steps=self.optim_max_steps,
                            measurement_type=self.meas_type,
                            n_classes = data.num_of_classes
                            )

        # get the trained model
        #
        self.model_d[ALG_NAME_MDL] = qnn_model.fit(samples, labels)

        # get the error rate for the training samples
        #
        error_rate, ypred = qnn_model.score(samples, labels)

        # if write_train_labels is True, write the labels to the file
        #
        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, error_rate
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, QNN.__CLASS_NAME__, ndt.__NAME__))

        # get the samples
        #
        samples = np.array(data.data)

        # get the trained model
        #
        model = model[ALG_NAME_MDL]

        # get the predicted labels
        #
        p_labels = model.predict(samples)

        # currently QNN does not support posterior calculation
        #
        posteriors = None

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of class (QNN)

#------------------------------------------------------------------------------
# ML Tools Class: QRBM
#------------------------------------------------------------------------------

class QRBM:
    """
    class: QRBM

    description:
     This is a class that implements QRBM.
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         This is the default constructor for the class.
        """

        # set the class name
        #
        QRBM.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict(dict)

        # set the model
        #
        self.model_d[ALG_NAME_ALG] = self.__class__.__name__
        self.model_d[ALG_NAME_IMP] = IMP_NAME_DWAVE
        self.model_d[ALG_NAME_MDL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self, data: MLToolsData,
              write_train_labels: bool, fname_train_labels: str):
        """
        method: train

        arguments:
         data: a list of numpy float matrices of feature vectors
         write_train_labels: a boolean to whether write the train data
         fname_train_labels: the filename of the train file

        return:
         model: a PyTorch state_dict containing the model
         error: training error rate

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: training a model" %
                  (__FILE__, ndt.__LINE__, QRBM.__CLASS_NAME__, ndt.__NAME__))

        # get the samples
        #
        samples =  np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # get the parameters
        #
        self.mdl_name = str(self.model_d[ALG_NAME_ALG])
        self.provider_name = str(self.params_d[QML_NAME_PROVIDER])
        self.n_hidden = int(self.params_d[QML_NAME_NHIDDEN])
        self.shots = int(self.params_d[QML_NAME_SHOTS])
        self.cs = int(self.params_d[QML_NAME_CS])
        self.n_neighbors = int(self.params_d[ALG_NAME_NNEARN])
        self.encoder = str(self.params_d[QML_NAME_ENCODER])

        # get the number of visible node which is the number of features
        # in the dataset
        #
        self.n_visible = samples[0].shape[0]

        # create the model
        #
        qrbm_knn_model = nqt.QML(model_name=self.mdl_name,
                                 provider_name=
                                 self.provider_name,
                                 encoder_name=self.encoder,
                                 n_hidden=self.n_hidden,
                                 n_visible=self.n_visible,
                                 shots=self.shots,
                                 chain_strength=self.cs,
                                 n_neighbors=self.n_neighbors,
                                )

        # get the trained model
        #
        self.model_d[ALG_NAME_MDL] = qrbm_knn_model.fit(samples, labels)

        # get the error rate for the training samples
        #
        error_rate, ypred = qrbm_knn_model.score(samples, labels)

        # if write_train_labels is True, write the labels to the file
        #
        if write_train_labels:
            data.write(oname = fname_train_labels, label = ypred)

        # exit gracefully
        #
        return self.model_d, error_rate
    #
    # end of method

    def predict(self, data: MLToolsData, model = None):
        """
        method: predict

        arguments:
         data: a numpy float matrix of feature vectors (each row is a vector)
         model: an algorithm model (None = use the internal model)

        return:
         labels: a list of predicted labels

        description:
         none
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s::%s: entering predict" %
                  (__FILE__, ndt.__LINE__, QRBM.__CLASS_NAME__, ndt.__NAME__))

        # get the samples
        #
        samples = np.array(data.data)

        # get the trained model
        #
        model = model[ALG_NAME_MDL]

        # get the predicted labels
        #
        p_labels = model.predict(samples)

        # currently QRBM does not support posterior calculation
        #
        posteriors = None

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of class (QRBM)

#******************************************************************************
#
# Section 5: definitions dependent on the above classes go here
#
#******************************************************************************

# define variables to configure the machine learning algorithms
#
ALGS = {EUCL_NAME:EUCLIDEAN,
        PCA_NAME:PCA,
        LDA_NAME:LDA,
        QDA_NAME:QDA,
        QLDA_NAME:QLDA,
        NB_NAME:NB,
        GMM_NAME:GMM,

        KNN_NAME:KNN,
        KMEANS_NAME:KMEANS,
        RNF_NAME:RNF,
        SVM_NAME:SVM,

        MLP_NAME:MLP,
        RBM_NAME:RBM,
        TRANSF_NAME:TRANSFORMER,

        QSVM_NAME:QSVM,
        QNN_NAME:QNN,
        QRBM_NAME:QRBM}

#
# end of file

