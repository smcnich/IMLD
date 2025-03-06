#!/usr/bin/env python
#
# file: $NEDC_NFC/class/python/nedc_ml_tools/nedc_ml_tools.py
#
# revision history:
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
#  Principle Component Analysis (PCA)
#  Quadratic Discriminant Analysis (QDA)
#  Linear Discriminate Analysis class independent (LDA)
#  Linear Discriminate Analysis class dependent (QLDA)
#
# in this class. These are accessed through a wrapper class called Algorithm.
#
# The implementations that are included here are a mixture of things found
# in the statistical package JMP, the Python machine learning library
# scikit-learn and the ISIP machine learning demo, IMLD.
#------------------------------------------------------------------------------

# import required system modules
#
from collections import defaultdict
import datetime as dt
import numpy as np
import pickle
import os
import sys
from imblearn.metrics import sensitivity_score, specificity_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (silhouette_score,
                            classification_report,
                            precision_score,
                            accuracy_score, f1_score)
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.metrics import f1_score
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn import linear_model
import torch

# import required NEDC modules
#
import nedc_debug_tools as ndt
import nedc_file_tools as nft
import nedc_cov_tools as nct
import nedc_trans_tools as ntt
import nedc_qml_tools as nqt
from nedc_ml_tools_data import MLToolsData

#---------------------------- Example -----------------------------------------
# alg = Alg()
# alg.set(LDA_NAME)
# alg.load_parameters("params_00.txt")
#
# data = MLToolsData("data.csv")
#
# score, model = alg.train(data)
# labels, post = alg.predict(data)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

DATA_LABELS = "labels"
DATA_DATA = "data"

DEF_TRAIN_LABELS_FNAME = "train_labels.csv"

# define the names of keys in dictionaries that are used to access parameters:
#  these are common across all algorithms. a parameter dictionary has
#  two keys (name and param). a model dictionary has two keys (name and model).
#
ALG_PRIORS_ML = "ml"
ALG_PRIORS_MAP = "map"

ALG_PRM_KEY_NAME = "name"
ALG_PRM_KEY_PARAM = "params"
ALG_PRM_KEY_PRIOR = "prior"
ALG_PRM_KEY_WEIGHTS = "weights"


ALG_MDL_KEY_NAME = ALG_PRM_KEY_NAME
ALG_MDL_KEY_MODEL = "model"
ALG_MDL_KEY_MEANS = "means"
ALG_MDL_KEY_COV = "covariance"
ALG_MDL_KEY_PRIOR = "prior"
ALG_MDL_KEY_TRANS = "transform"
ALG_MDL_KEY_EVAL = "eigen_value"
ALG_MDL_KEY_MAPPING_LABEL = "mapping_label"

ALG_MDL_KEY_CLASS_MODELS = "class_models"
ALG_MDL_KEY_CLASS_LABELS = "class_labels"


# define formats for generating a scoring report
#
ALG_SCL_PCT = float(100.0)

ALG_FMT_DTE = "Date: %s%s"
ALG_FMT_LBL = "%06d"
ALG_FMT_ERR = "%012s %10.2f%%"
ALG_FMT_WLB = "%10s"
ALG_FMT_WCL = "%6d"
ALG_FMT_WPC = "%6.2f"
ALG_FMT_WST = "%6d (%6.2f%%)"
ALG_FMT_DEC = ".4f"

#------------------------------------------------------------------------------
# Alg = PCA: define dictionary keys for parameters
#
# define the algorithm name
#
PCA_NAME = "PCA"

# the parameter block for PCA looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'PCA',
#   'params': {
#                'name': 'PCA',
#               'prior': 'ml',
#               'ctype': 'full',
#              'center': 'None',
#               'scale': 'biased'
#   }
#  })
#
# Define the keys for these parameters. The keys ("name" and "params") are
# the same across all algorithms. The actual parameters are contained
# in a sub-dictionary that has keys specific to that algorithm.
#
PCA_PRM_KEY_NAME = ALG_PRM_KEY_NAME
PCA_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
PCA_PRM_KEY_PRIOR = ALG_PRM_KEY_PRIOR
PCA_PRM_KEY_CTYPE = nct.PRM_CTYPE
PCA_PRM_KEY_CENTER = nct.PRM_CENTER
PCA_PRM_KEY_SCALE = nct.PRM_SCALE
PCA_PRM_KEY_COMP ='n_components'

# The model for PCA contains:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'PCA',
#   'model': {
#               'prior': numpy array,
#         '      means': list,
#          'covariance': matrix
#   }
#  })
#
# Define the keys for these model parameters. The first key, name, is the same
# across all algorithms. The actual parameters are contained in
# a sub-dictionary that has keys specific to that algorithm.
#
PCA_MDL_KEY_NAME = ALG_MDL_KEY_NAME
PCA_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
PCA_MDL_KEY_PRIOR = ALG_MDL_KEY_PRIOR
PCA_MDL_KEY_MEANS = ALG_MDL_KEY_MEANS
PCA_MDL_KEY_TRANS = ALG_MDL_KEY_TRANS
PCA_MDL_KEY_COV = ALG_MDL_KEY_COV
PCA_MDL_KEY_EVAL = ALG_MDL_KEY_EVAL

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = QDA: define dictionary keys for parameters
#

# define the algorithm name
#
QDA_NAME = "QDA"

# the parameter block for QDA looks like this:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'QDA',
#    'params': {
#                'name': 'QDA',
#               'prior': 'ml',
#               'ctype': 'full',
#              'center': 'None',
#               'scale': 'biased'
#    }
#   })
#
# Define the keys for these parameters.
#
QDA_PRM_KEY_NAME = ALG_PRM_KEY_NAME
QDA_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
QDA_PRM_KEY_PRIOR = ALG_PRM_KEY_PRIOR
QDA_PRM_KEY_CTYPE = nct.PRM_CTYPE
QDA_PRM_KEY_CENTER = nct.PRM_CENTER
QDA_PRM_KEY_SCALE = nct.PRM_SCALE
QDA_PRM_KEY_COMP ='n_components'

# The model for QDA contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'QDA',
#    'model': {
#               'prior': numpy array,
#               'means': list,
#          'covariance': matrix
#   }
#  })
#
# Define the keys for these model parameters.
#
QDA_MDL_KEY_NAME = ALG_MDL_KEY_NAME
QDA_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
QDA_MDL_KEY_PRIOR = ALG_MDL_KEY_PRIOR
QDA_MDL_KEY_MEANS = ALG_MDL_KEY_MEANS
QDA_MDL_KEY_COV = ALG_MDL_KEY_COV
QDA_MDL_KEY_TRANS = ALG_MDL_KEY_TRANS
QDA_MDL_KEY_EVAL = ALG_MDL_KEY_EVAL
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = LDA: define dictionary keys for parameters
#

# define the algorithm name
#
LDA_NAME = "LDA"

# the parameter block for LDA looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'LDA',
#   'params': {
#                'name': 'LDA',
#               'prior': 'ml',
#               'ctype': 'full',
#              'center': 'None',
#               'scale': 'None'
#    }
#  })
#
#
# Define the keys for these parameters.
#
LDA_PRM_KEY_NAME = ALG_PRM_KEY_NAME
LDA_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
LDA_PRM_KEY_PRIOR = ALG_PRM_KEY_PRIOR
LDA_PRM_KEY_CTYPE = nct.PRM_CTYPE
LDA_PRM_KEY_CENTER = nct.PRM_CENTER
LDA_PRM_KEY_SCALE = nct.PRM_SCALE

# The model for LDA contains:
#
#   defaultdict(<class 'dict'>, {
#   'name': 'LDA',
#   'model': {
#               'prior': numpy array,
#               'means': list,
#           'transform': matrix
#   }
#  })
#
# Define the keys for these model parameters.
#
LDA_MDL_KEY_NAME = ALG_MDL_KEY_NAME
LDA_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
LDA_MDL_KEY_PRIOR = ALG_MDL_KEY_PRIOR
LDA_MDL_KEY_MEANS = ALG_MDL_KEY_MEANS
LDA_MDL_KEY_TRANS = ALG_MDL_KEY_TRANS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = QLDA: define dictionary keys for parameters
#

# define the algorithm name
#
QLDA_NAME = "QLDA"

# the parameter block for QLDA contains:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'QLDA',
#   'params': {
#                'name': 'QLDA',
#               'prior': 'ml',
#               'ctype': 'full',
#              'center': 'None',
#               'scale': 'None'
#   }
#  })
#
# define the keys for these parameters. the keys ("name" and "params"), are
# the same across all algorithms. the actual parameters are contained
# in a sub-dictionary that has keys specific to that algorithm.
#
QLDA_PRM_KEY_NAME = ALG_PRM_KEY_NAME
QLDA_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
QLDA_PRM_KEY_PRIOR = ALG_PRM_KEY_PRIOR
QLDA_PRM_KEY_CTYPE = nct.PRM_CTYPE
QLDA_PRM_KEY_CENTER = nct.PRM_CENTER
QLDA_PRM_KEY_SCALE = nct.PRM_SCALE

# The model for QLDA contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'QLDA',
#    'model': {
#               'prior': numpy array,
#               'means': list,
#           'transform': matrix
#   }
#  })
#
# Define the keys for these model parameters.
#
QLDA_MDL_KEY_NAME = ALG_MDL_KEY_NAME
QLDA_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
QLDA_MDL_KEY_PRIOR = ALG_MDL_KEY_PRIOR
QLDA_MDL_KEY_MEANS = ALG_MDL_KEY_MEANS
QLDA_MDL_KEY_TRANS = ALG_MDL_KEY_TRANS
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = NB: define dictionary keys for parameters
#

# define the algorithm name
#
NB_NAME = "NB"

# the parameter block for NB looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'NB',
#   'params': {
#                'name': 'NB',
#               'prior': 'ml'
#             'average': 'None'
#         'multi_class': 'ovr'
#    }
#  })
NB_PRM_KEY_NAME = ALG_PRM_KEY_NAME
NB_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
NB_PRM_KEY_PRIOR = ALG_PRM_KEY_PRIOR

# The model for NB contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'NB',
#    'model': Sklearn.NB.Model
#  })
#
# Define the keys for these parameters.
#
NB_MDL_KEY_NAME = ALG_MDL_KEY_NAME
NB_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = EUCLIDEAN: define dictionary keys for parameters
#
# define the algorithm name
#
EUCLIDEAN_NAME = "EUCLIDEAN"

# the parameter block for EUCLIDEAN looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'EUCLIDEAN',
#   'params': {
#                'name': 'EUCLIDEAN',
#             'weights': list
#    }
#  })
EUCLIDEAN_PRM_KEY_NAME = ALG_PRM_KEY_NAME
EUCLIDEAN_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
EUCLIDEAN_PRM_KEY_WEIGHTS = ALG_PRM_KEY_WEIGHTS

# The model for EUCLIDEAN contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'EUCLIDEAN',
#    'model' {
#                'means': list,
#    }
#  })
#
# Define the keys for these parameters.
#
EUCLIDEAN_MDL_KEY_NAME = ALG_MDL_KEY_NAME
EUCLIDEAN_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
EUCLIDEAN_MDL_KEY_MEANS = ALG_MDL_KEY_MEANS

#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = KNN: define dictionary keys for parameters
#

# define the algorithm name
#
KNN_NAME = "KNN"

# the parameter block for NB looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'KNN',
#   'params': {
#                 'name': 'KNN',
#             'neighbor': 1
#    }
#  })

KNN_PRM_KEY_NAME = ALG_PRM_KEY_NAME
KNN_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
KNN_PRM_KEY_NEIGHB = "neighbor"

# The model for KNN contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'KNN',
#    'model': Sklearn KNN Model
#  })
#
KNN_MDL_KEY_NAME = ALG_MDL_KEY_NAME
KNN_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = RNF: define dictionary keys for parameters
#

# define the algorithm name
#
RNF_NAME = "RNF"

# the parameter block for NB looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'RNF',
#   'params': {
#                 'name': 'RNF',
#          'n_estimator': 1,
#            'max_depth': 5,
#            'criterion': 'gini'
#    }
#  })

RNF_PRM_KEY_NAME = ALG_PRM_KEY_NAME
RNF_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
RNF_PRM_KEY_ESTIMATOR = "estimator"
RNF_PRM_KEY_CRITERION = "criterion"
RNF_PRM_KEY_MAXDEPTH  = 'max_depth'
RNF_PRM_KEY_RANDOM = 'random_state'

# The model for RNF contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'RNF',
#    'model': Sklearn RNF Model
#  })
#
RNF_MDL_KEY_NAME = ALG_MDL_KEY_NAME
RNF_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = SVM: define dictionary keys for parameters
#

# define the algorithm name
#
SVM_NAME = "SVM"

# the parameter block for SVMlooks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'SVM ,
#   'params': {
#                 'name': 'SVM',
#                    'C': 1,
#                'gamma': 0.1,
#               'kernel': 'linear'
#    }
#  })

SVM_PRM_KEY_NAME = ALG_PRM_KEY_NAME
SVM_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
SVM_PRM_KEY_C = "c"
SVM_PRM_KEY_GAMMA = "gamma"
SVM_PRM_KEY_KERNEL = 'kernel'

# The model for SVM contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'SVM',
#    'model': Sklearn SVM Model
#  })
#
SVM_MDL_KEY_NAME = ALG_MDL_KEY_NAME
SVM_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = SVM: define dictionary keys for parameters
#

# define the algorithm name
#
MLP_NAME = "MLP"

# the parameter block for SVMlooks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'MLP ,
#   'params': {
#                 'name': 'MLP',
#          'hidden_size': 3,
#           'activation': 'relu',
#               'solver': 'adam',
#            'batch_size: 'auto',
#         'learning_rate: 'constant',
#         'random_state': 0,
#    'learning_rate_init: 0.001,
#        'early_stopping: False,
#               'shuffle: True,
#   'validation_fraction: 0.1,
#             'momentum': 0.9
#             'max_iter': 200
#    }
#  })

MLP_PRM_KEY_NAME = ALG_PRM_KEY_NAME
MLP_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
MLP_PRM_KEY_HSIZE = "hidden_size"
MLP_PRM_KEY_ACT = "activation"
MLP_PRM_KEY_SOLVER = "solver"
MLP_PRM_KEY_BSIZE = "batch_size"
MLP_PRM_KEY_LR = "learning_rate"
MLP_PRM_KEY_RANDOM = "random_state"
MLP_PRM_KEY_LRINIT = "learning_rate_init"
MLP_PRM_KEY_STOP = "early_stopping"
MLP_PRM_KEY_SHUFFLE = "shuffle"
MLP_PRM_KEY_VAL = "validation_fraction"
MLP_PRM_KEY_MOMENTUM = "momentum"
MLP_PRM_KEY_MITER = "max_iter"

# The model for MLP contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'MLP',
#    'model': Sklearn MLP Model
#  })
#
MLP_MDL_KEY_NAME = ALG_MDL_KEY_NAME
MLP_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = KMEANS: define dictionary keys for parameters
#

# define the algorithm name
#
KMEANS_NAME = "KMEANS"

# the parameter block for KMEANS looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'KMEANS"
#   'params': {
#               'name': 'KMEANS',
#          'n_cluster': 2,
#             'n_init': 3,
#       'random_state': 0,
#           'max_iter': 100
#    }
#  })

KMEANS_PRM_KEY_NAME = ALG_PRM_KEY_NAME
KMEANS_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
KMEANS_PRM_KEY_NCLUSTER = "n_cluster"
KMEANS_PRM_KEY_NINIT = "n_init"
KMEANS_PRM_KEY_RANDOM = "random_state"
KMEANS_PRM_KEY_MITER = "max_iter"

# The model for KMEANS contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'KMEANS',
#    'model': Sklearn KMEANS Model
#  })
#
KMEANS_MDL_KEY_NAME = ALG_MDL_KEY_NAME
KMEANS_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Alg = RBM: define dictionary keys for parameters
#
# define the algorithm name
#
RBM_NAME = "RBM"

# the parameter block for RBM looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'RBM"
#   'params': {
#               'name': 'RBM',
#       'n_components': 2,
#      'learning_rate': 3,
#         'batch_size': 0,
#             'n_iter': 100,
#            'verbose': 0,
#       'random_state': None
#    }
#  })

RBM_PRM_KEY_NAME = ALG_PRM_KEY_NAME
RBM_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
RBM_PRM_KEY_COMP = "n_components"
RBM_PRM_KEY_NITER = "n_iter"
RBM_PRM_KEY_RANDOM = "random_state"
RBM_PRM_KEY_LR = "learning_rate"
RBM_PRM_KEY_BSIZE = "batch_size"
RBM_PRM_KEY_VERBOSE = "verbose"
RBM_PRM_KEY_CLASSIF ="classifier"

# The model for RBM contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'RBM',
#    'model': Sklearn RBM Model
#  })
#
RBM_MDL_KEY_NAME = ALG_MDL_KEY_NAME
RBM_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL

#------------------------------------------------------------------------------
# Alg = TRANSFORMER: define dictionary keys for parameters
#
# define the algorithm name
#
TRANSFORMER_NAME = "TRANSFORMER"

# the parameter block for TRANSFORMER looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'TRANSFORMER"
#   'params': {
#                name : TRANSFORMER
#               epoch : 50
#                  lr : 0.001
#          batch_size : 32
#          embed_size : 32
#              nheads : 2
#          num_layers : 2
#             MLP_dim : 64
#             dropout : 0.1
#    }
#  })

TRANS_PRM_KEY_NAME = ALG_PRM_KEY_NAME
TRANS_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
TRANS_PRM_KEY_EPOCH = "epoch"
TRANS_PRM_KEY_LR = "lr"
TRANS_PRM_KEY_BSIZE = "batch_size"
TRANS_PRM_KEY_EMBED_SIZE = "embed_size"
TRANS_PRM_KEY_NHEADS = "nheads"
TRANS_PRM_KEY_NLAYERS = "num_layers"
TRANS_PRM_KEY_MLP_DIM = "MLP_dim"
TRANS_PRM_KEY_DROPOUT = "dropout"

# The model for TRANSFORMER contains:
# {'name': 'TRANSFORMER', 'model':
# defaultdict(None, {'name': 'TRANSFORMER, 'model': OrderedDict([
#     ('input_embedding.weight', tensor([32, 2])),
#     ('input_embedding.bias', tensor(32)),
#     ('encoder.layers.0.self_attention_block.w_q.weight', tensor([32, 32])),
#     ('encoder.layers.0.self_attention_block.w_q.bias', tensor(32)),
#     ('encoder.layers.0.self_attention_block.w_k.weight', tensor([32, 32])),
#     ('encoder.layers.0.self_attention_block.w_k.bias', tensor(32)),
#     ...
#     ('encoder.norm.alpha', tensor([0.9810])),
#     ('encoder.norm.beta', tensor([0.9940])),
#     ('classifier.weight', tensor([2, 32])),
#     ('classifier.bias', tensor([ 0.0642, -0.1198]))
# ])})
TRANS_MDL_KEY_NAME = ALG_MDL_KEY_NAME
TRANS_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL

#------------------------------------------------------------------------------
# Alg = QSVM: define dictionary keys for parameters
#
# define the algorithm name
#
QSVM_NAME = "QSVM"

# the parameter block for QSVM looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'QSVM"
#   'params': {
#               name          : QSVM
#               model_name    : qsvm
#               provider_name : qiskit
#               hardware      : cpu
#               encoder_name  : zz
#               kernel_name   : fidelity
#               entanglement  : full
#               reps          : 2
#               n_qubits      : 4
#               shots         : 1024
#    }
#  })

# define quantum ML algorithms related common parameters keys
#
QML_PRM_KEY_MDL_NAME = "model_name"
QML_PRM_KEY_PROVIDER = "provider_name"
QML_PRM_KEY_HARDWARE = "hardware"
QML_PRM_KEY_ENCODER = "encoder_name"
QML_PRM_KEY_ENTANGLEMENT = "entanglement"
QML_PRM_KEY_FEAT_REPS = "featuremap_reps"
QML_PRM_KEY_ANSATZ_REPS = "ansatz_reps"
QML_PRM_KEY_NQUBITS = "n_qubits"
QML_PRM_KEY_SHOTS = "shots"

# define QSVM algorithms related common model keys
#
QSVM_PRM_KEY_NAME = ALG_PRM_KEY_NAME
QSVM_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
QSVM_PRM_KEY_PROVIDER = QML_PRM_KEY_PROVIDER
QSVM_PRM_KEY_HARDWAR = QML_PRM_KEY_HARDWARE
QSVM_PRM_KEY_ENCODER = QML_PRM_KEY_ENCODER
QSVM_PRM_KEY_ENTANGLEMENT = QML_PRM_KEY_ENTANGLEMENT
QSVM_PRM_KEY_NQUBITS = QML_PRM_KEY_NQUBITS
QSVM_PRM_KEY_FEAT_REPS = QML_PRM_KEY_FEAT_REPS
QSVM_PRM_KEY_ANSATZ_REPS = QML_PRM_KEY_ANSATZ_REPS
QSVM_PRM_KEY_MDL_NAME = QML_PRM_KEY_MDL_NAME
QSVM_PRM_KEY_SHOTS = QML_PRM_KEY_SHOTS
QSVM_PRM_KEY_KERNEL = "kernel_name"

# The model for QSVM contains:
# {'name': 'QSVM', 'model':
# defaultdict(None, {'name': 'QSVM, 'model': <nedc_qml_tools.QSVM object at
# 0x7f887741a1d0>})
#
QSVM_MDL_KEY_NAME = ALG_MDL_KEY_NAME
QSVM_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL

#------------------------------------------------------------------------------
# Alg = QNN: define dictionary keys for parameters
#
# define the algorithm name
#
QNN_NAME = "QNN"

# the parameter block for QNN looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'QNN"
#   'params': {
#               name             : QNN
#               model_name       : qnn
#               provider_name    : qiskit
#               encoder_name     : zz
#               ansatz_name      : real_amplitudes
#               hardware         : COBYLA
#               optim_name       : COBYLA
#               optim_max_steps  : 50
#               entanglement     : full
#               reps             : 2
#               n_qubits         : 2
#               meas_type        : sampler
#    }
#  })


# define QSVM algorithms related common model keys
#
QNN_PRM_KEY_NAME = ALG_PRM_KEY_NAME
QNN_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
QNN_PRM_KEY_PROVIDER = QML_PRM_KEY_PROVIDER
QNN_PRM_KEY_HARDWAR = QML_PRM_KEY_HARDWARE
QNN_PRM_KEY_ENCODER = QML_PRM_KEY_ENCODER
QNN_PRM_KEY_ENTANGLEMENT = QML_PRM_KEY_ENTANGLEMENT
QNN_PRM_KEY_NQUBITS = QML_PRM_KEY_NQUBITS
QNN_PRM_KEY_REPS = QML_PRM_KEY_FEAT_REPS
QNN_PRM_KEY_ANSATZ_REPS = QML_PRM_KEY_ANSATZ_REPS
QNN_PRM_KEY_MDL_NAME = QML_PRM_KEY_MDL_NAME
QNN_PRM_KEY_MEAS_TYPE = "meas_type"
QNN_PRM_KEY_ANSATZ = "ansatz_name"
QNN_PRM_KEY_MAXSTEPS = "optim_max_steps"
QNN_PRM_KEY_OPTIM = "optim_name"

# The model for QNN contains:
# {'name': 'QNN', 'model':
# defaultdict(None, {'name': 'QNN, 'model': <nedc_qml_tools.QNN object at
# 0x7fd6626e2b10>})
#
QNN_MDL_KEY_NAME = ALG_MDL_KEY_NAME
QNN_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL

#------------------------------------------------------------------------------
# Alg = QRBM: define dictionary keys for parameters
#
# define the algorithm name
#
QRBM_NAME = "QRBM"

# the parameter block for QRBM looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'QRBM"
#   'params': {
#               name             : QRBM
#               model_name       : qrbm
#               n_hiddem         : 10
#               shots            : 2
#               chain_strength   : 2
#    }
#  })


# define QRBM algorithms related common model keys
#
QRBM_PRM_KEY_NAME = ALG_PRM_KEY_NAME
QRBM_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
QRBM_PRM_KEY_MDL_NAME = QML_PRM_KEY_MDL_NAME
QRBM_PRM_KEY_SHOTS = QML_PRM_KEY_SHOTS
QRBM_PRM_KEY_PROVIDER = QML_PRM_KEY_PROVIDER
QRBM_PRM_KEY_ENCODER = QML_PRM_KEY_ENCODER
QRBM_PRM_KEY_NHIDDEN = "n_hidden"
QRBM_PRM_KEY_CS = "chain_strength"
QRBM_PRM_KEY_N_NEIGHBORS = "knn_n_neighbors"


# The model for QRBM contains:
# {'name': 'QRBM', 'model':
# defaultdict(None, {'name': 'QRBM, 'model': QRBMClassifier
# (provider=DWaveProvider))
#
QRBM_MDL_KEY_NAME = ALG_MDL_KEY_NAME
QRBM_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL

#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Alg = GMM: define dictionary keys for parameters
#

# define the algorithm name
#
GMM_NAME = "GMM"

# the parameter block for NB looks like this:
#
#  defaultdict(<class 'dict'>, {
#   'name': 'GMM',
#   'params': {
#               'name': 'GMM',
#               'prior': 'ml',
#               'n_components': 1,
#    }
#  })
GMM_PRM_KEY_NAME = ALG_PRM_KEY_NAME
GMM_PRM_KEY_PARAM = ALG_PRM_KEY_PARAM
GMM_PRM_KEY_PRIOR = ALG_PRM_KEY_PRIOR
GMM_PRM_KEY_COMP ='n_components'
GMM_PRM_KEY_RANDOM = 'random_state'


GMM_MDL_KEY_NAME = ALG_MDL_KEY_NAME
GMM_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
GMM_MDL_KEY_PRIOR = ALG_MDL_KEY_PRIOR
GMM_MDL_KEY_CLASS_MODELS = ALG_MDL_KEY_CLASS_MODELS
GMM_MDL_KEY_CLASS_LABELS = ALG_MDL_KEY_CLASS_LABELS

# The model for NB contains:
#
#   defaultdict(<class 'dict'>, {
#    'name': 'NB',
#    'model': Sklearn.NB.Model
#  })
#
# Define the keys for these parameters.
#
NB_MDL_KEY_NAME = ALG_MDL_KEY_NAME
NB_MDL_KEY_MODEL = ALG_MDL_KEY_MODEL
#------------------------------------------------------------------------------

# declare global debug and verbosity objects so we can use them
# in both functions and classes
#
dbgl_g = ndt.Dbgl()
vrbl_g = ndt.Vrbl()

#******************************************************************************
#
# Parent Class: Alg
#
#******************************************************************************

class Alg:
    """
    Class: Alg

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
         note that this method does not descend into a specific alg class.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: set algorithm name (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, alg_name))

        # attempt to set the name only
        #
        if alg_name in ALGS:
            self.alg_d = ALGS[alg_name]

        # if the algorithm is not set print an error message and exit
        #
        else:

            # print informational error message
            #
            print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "unknown algorithm name", alg_name))

            # exit ungracefully
            #  algorithm setting failed
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
            print("%s (line: %s) %s: setting parameters" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check that the argument is a valid dictionary
        #
        if not isinstance(parameters, (dict, defaultdict)):

            # if not a valid dictionary print error message and exit
            #
            raise TypeError(
            f"{__FILE__} (line: {ndt.__LINE__} {ndt.__NAME__}: ",
            "invalid parameter structure",
            f"dict, defaultdict expected, got '{type(parameters).__name__}')")

        # check the algorithm name of the parameter file
        #
        if self.set(parameters[ALG_PRM_KEY_NAME]) is False:

            # if the algorithm specified in the parameter file
            # is not supprted print error message
            #
            print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "unsupported algorithm name", parameters[ALG_PRM_KEY_NAME]))

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
         note that this method does not descend into a specific alg class.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: getting the algorithm name" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if an algorithm is set
        #
        if self.alg_d is None:

            # print error message if the algorithm
            # was not set
            #
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "no algorithm has been set"))

            # exit ungracefully
            #  algoritm is not set
            #
            return None

        # exit gracefully
        #  return algorithm type
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

        # exit gracefully
        #  return ref label info for numpy
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

        # exit gracefully
        #  return the hyp_labels for numpy
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
         this method loads a compatible picked model
        """

        # unpickle the model file
        #
        try:
            fp = open(fname, nft.MODE_READ_BINARY)
            model = pickle.load(fp)
        except:
            print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "error loading model file", fname))
            return None

        # check the type of data
        #
        if not isinstance(model, (dict, defaultdict)):
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "unknown model type"))
            return None

        # check model file key length: it should only contain 2 keys
        #
        if len(model) != int(2):
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "unknown model format"))
            return None

        if model[ALG_MDL_KEY_NAME] != self.alg_d.__class__.__name__:
            print("Error: %s (line: %s) %s: the current set algorithm and \
                model name does not match, "
                "Model Name: %s, Set Algorithm: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__, model[ALG_MDL_KEY_NAME],
                self.alg_d.__class__.__name__))
            return None

        # set the parameters
        #
        self.alg_d.model_d = model

        # exit gracefully
        #  return the model
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
         this method loads a specific algorithm parameter block.
         the algorithm name must be set before it is called.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: loading parameters (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

        # check that an algorithm has been set
        #
        if self.alg_d is None:
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "no algorithm has been set"))
            return None

        # make sure the file is a valid parameter file
        #
        if nft.get_version(fname) is None:
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "invalid parameter file (version not specified or \
                   invalid version)"))
            return None

        # attempt to load the parameters
        #
        params = nft.load_parameters(fname, self.alg_d.__class__.__name__)
        if params is None:
            print("Error: %s (line: %s) %s: %s [%s]" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "unable to parse parameter file", fname))
            return None

        # set the internal parameters
        #
        self.alg_d.params_d[ALG_PRM_KEY_NAME] = self.alg_d.__class__.__name__
        self.alg_d.params_d[ALG_PRM_KEY_PARAM] = params

        # exit gracefully
        #  return loaded parameters
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
            print("%s (line: %s) %s: saving model (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

        # check that there is a valid model
        #
        if self.alg_d is None:
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "invalid model"))
            return False

        # pickle it to a file and trap for errors
        #
        try:
            fp = open(fname, nft.MODE_WRITE_BINARY)
            pickle.dump(self.alg_d.model_d, fp)
            fp.close()
        except:
            print("Error: %s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "error writing model file", fname))
            return False

        # exit gracefully
        #  model successfully saved
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
         this method writes the current self.alg.params_d to an
         NEDC parameter file
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: saving parameters" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if an algorithm has been set
        #
        if self.alg_d is None:
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "no algorithm has been set"))
            return False

        # check if params_d is empty
        #
        if self.alg_d.params_d is None:
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "parameter block is empty"))
            return False

        # open the file for writing
        #
        try:
            fp = open(fname, nft.MODE_WRITE_TEXT)
        except:
            print("Error: %s (line: %s) %s: error opening file (%s)" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, fname))

        # write a bare bones parameter file:
        #  start with the version information
        #
        fp.write("%s %s %s" % (nft.DELIM_VERSION, nft.DELIM_EQUAL,
                               nft.PFILE_VERSION + nft.DELIM_NEWLINE))
        fp.write("%s %s" % (self.alg_d.__class__.__name__,
                            nft.DELIM_BOPEN + nft.DELIM_NEWLINE))

        # add the parameter structure
        #
        for key, val in self.alg_d.params_d[ALG_PRM_KEY_PARAM].items():
            fp.write(" %s %s %s" % (key, nft.DELIM_EQUAL,
                                    val + nft.DELIM_NEWLINE))

        fp.write("%s" % (nft.DELIM_BCLOSE + nft.DELIM_NEWLINE))

        # exit gracefully
        #  parameters successfully written to parameter file
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
              fname_train_labels = DEF_TRAIN_LABELS_FNAME,
              ):
        """
        method: train

        arguments:
         data: a list of numpy matrices where the index corresponds to the class
         write_train_labels: option to output the predicted labels from the after
                             after training (False)
         fname_train_labels: the fname to output it too ("train_labels.csv")
         excel: if true, write the file as excel (.xlsx) (False)

         Note: After the "data" argument, you would need a keyword argument.

         Ex: train(data, write_train_labesl = True, fname = "labels.csv")

        return:
         model: a dictionary containing the model (data dependent)
         score: a float value containing a measure of the goodness of fit

        description:
         note that data is a list of matrices organized by class label,
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if the algorithm has been configured
        #
        if self.alg_d is None:
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "no algorithm has been set"))

            # exit ungracefully
            #  algorithm not set
            #
            return None, None

        # check that the data variable is an MLToolsData
        # instance
        #
        if not isinstance(data, MLToolsData):
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "data is not type of MLToolsData"))

            # exit ungracefully
            #  incompatible object type
            #
            return None, None

        # exit gracefully
        #  return model and its score
        #
        return self.alg_d.train(data, write_train_labels, fname_train_labels)

    #
    # end of method

    def predict(self,
                data: MLToolsData,
                model = None):
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
         this is a wrapper method that calls the set algorithms predict method
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if the algorithm has been configured
        #
        if self.alg_d is None:
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "no algorithm has been set"))

            # exit ungracefully
            #  algorithm not set
            #
            return None, None

        # check if data variable is an MLToolsData instance
        #
        if not isinstance(data, MLToolsData):
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "data is not type of MLToolsData"))

            # exit ungracefully
            #  incompatible object type
            #
            return None, None

        # exit gracefully
        #  reutn model prediction
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
         note: we pass the number of classes because all of the classes
         might not appear in the reference labels.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: generating a confusion matrix" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # build a labels matrix
        #
        lbls = list(range(num_classes))

        # exit gracefully
        #  return confusion matrix
        #
        return sklearn_confusion_matrix(ref_labels, hyp_labels, labels = lbls)

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
         prints the confusion matirx
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

            # convert (row, column) of confusion matrix to
            # precentages
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
                fp.write(ALG_FMT_WST % (cnf[i][j], ALG_SCL_PCT * pct[i][j]))
            fp.write(nft.DELIM_NEWLINE)

        # exit gracefully
        #  confusion matrix successfully printed
        #
        return True

    #
    # end of method

    def score(self, num_classes, data, hyp_labels, *,
              isPrint = False,
              fp = sys.stdout):
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
            print("%s (line: %s) %s: scoring the results" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__))

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
        err = ALG_SCL_PCT * (float(1.0) - acc)
        sens = None
        spec = None
        prec = None
        f1 = None

        # print confusion matrix and return nothing if specified
        #
        if isPrint:

            # print the confusion matrix
            #
            self.print_confusion_matrix(conf_matrix, data.mapping_label, fp = fp)
            fp.write(nft.DELIM_NEWLINE)

            # generate a master list of labels:
            # we have to do this because some of the labels might not appear
            # in the data
            #
            lbls = []
            for i in range(num_classes):
                lbls.append(ALG_FMT_LBL % i)

            # generate and print the classification report
            #
            rpt = classification_report(r_labels, h_labels,
                                        labels = lbls, zero_division = 1)
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

        # exit gracefully
        #  operation successfully completed
        #
        return None if isPrint else conf_matrix, sens, spec, prec, acc, err, f1

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
            print("%s (line: %s) %s: scoring the results" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

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

        # generate a master list of labels:
        #  we have to do this because some of the labels might not appear
        #  in the data
        #
        lbls = []
        for i in range(num_classes):
            lbls.append(ALG_FMT_LBL % i)

        # generate and print the classification report
        #
        rpt = classification_report(rlabels, hlabels,
                                    labels = lbls, zero_division = 1)
        fp.write(rpt)
        fp.write(nft.DELIM_NEWLINE)

        # compute the accuracy and the error rate
        #
        acc = accuracy_score(rlabels, hlabels)
        err = ALG_SCL_PCT * (float(1.0) - acc)
        print(ALG_FMT_ERR % ("error rate", err))

        # exit gracefully
        #  score successfully printed
        #
        return True

    #
    # end of method

#
# end of Alg

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
    Class: Euclidean

    description:
     this is a class that implements Euclidean Distance
    """

    def __init__(self) -> None:
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
        """
        # set the class name
        #
        EUCLIDEAN.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d[EUCLIDEAN_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[EUCLIDEAN_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[EUCLIDEAN_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[EUCLIDEAN_MDL_KEY_MODEL] = defaultdict(dict)
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
            this method returns the weighted euclidean distance
        """
        q = p2 - p1
        return np.sqrt((w*q*q).sum())
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
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

        self.model_d[EUCLIDEAN_MDL_KEY_MODEL][EUCLIDEAN_MDL_KEY_MEANS] = means

        # get the weights
        #
        weights = self.params_d[EUCLIDEAN_PRM_KEY_PARAM][EUCLIDEAN_PRM_KEY_WEIGHTS]

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

        score = acc / len(data.data)

        return self.model_d, score
    #
    # end of method

    def predict(self,
                data: MLToolsData,
                model = None):

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: entering predict" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # check for model validity
        #
        if model[EUCLIDEAN_MDL_KEY_NAME] != EUCLIDEAN_NAME:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "incorrect model name"))
            return None, None

        if not model[EUCLIDEAN_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "model parameter is empty"))
            return None, None

        means = self.model_d[EUCLIDEAN_MDL_KEY_MODEL][EUCLIDEAN_MDL_KEY_MEANS]
        weights = self.params_d[EUCLIDEAN_PRM_KEY_PARAM][EUCLIDEAN_PRM_KEY_WEIGHTS]
        mapping_label = data.mapping_label

        labels = []
        posteriors = []

        for d in data.data:

            diff_means = []
            cur_posterios = []

            for ind, mean in enumerate(means):
                distance = self.weightedDistance(mean, d, float(weights[ind]))
                diff_means.append(distance)
                cur_posterios.append(distance)
            predict_label_ind = np.argmin(diff_means)
            labels.append(predict_label_ind)
            posteriors.append(cur_posterios)

        return labels, posteriors
    #
    # end of method
#
# end of EUCLIDEAN

#------------------------------------------------------------------------------
# ML Tools Class: PCA
#------------------------------------------------------------------------------

class PCA:
    """
    Class: PCA

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
         this is the default constructor for the class.
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
        self.params_d[PCA_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[PCA_PRM_KEY_PARAM] = defaultdict(dict)

        # set the names
        #
        self.model_d[PCA_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[PCA_MDL_KEY_MODEL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data : MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
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

            # iterate over all labels stored in
            # MLToolsData instance
            #
            for j in range(len(data.labels)):

                # remap data
                #
                if uni_label[i]==data.labels[j]:
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
        mode_prior = self.params_d[PCA_PRM_KEY_PARAM][PCA_PRM_KEY_PRIOR]

        if mode_prior == ALG_PRIORS_ML:

            # compute the priors for each class
            #
            self.model_d[PCA_MDL_KEY_MODEL][PCA_MDL_KEY_PRIOR] = \
                np.full(shape = num_classes,
                        fill_value = (float(1.0) / float(num_classes)))

        # if the priors are based on occurrences : nct.PRIORs_MAP
        #
        elif mode_prior == ALG_PRIORS_MAP:

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

            self.model_d[PCA_MDL_KEY_MODEL][PCA_MDL_KEY_PRIOR] = priors * _sum

        else:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "unknown value for priors"))
            self.model_d[PCA_MDL_KEY_MODEL].clear()
            return None, None

        # calculate the means of each class:
        #  note these are a list of numpy vectors
        #
        means = []
        for element in new_data:
            means.append(np.mean(element, axis = 0))

        self.model_d[PCA_MDL_KEY_MODEL][PCA_MDL_KEY_MEANS] = means

        # calculate the cov:
        #  note this is a single matrix
        #
        self.model_d[PCA_MDL_KEY_MODEL][PCA_MDL_KEY_COV] = nct.compute(
            new_data,
            ctype = self.params_d[PCA_PRM_KEY_PARAM][PCA_PRM_KEY_CTYPE],
            center= self.params_d[PCA_PRM_KEY_PARAM][PCA_PRM_KEY_CENTER],
            scale= self.params_d[PCA_PRM_KEY_PARAM][PCA_PRM_KEY_SCALE])
        cov = self.model_d[PCA_MDL_KEY_MODEL][PCA_MDL_KEY_COV]
        print(cov)

        # number of components
        #
        n_comp= int(self.params_d[PCA_PRM_KEY_PARAM][PCA_PRM_KEY_COMP])

        #if not( 0 < n_comp <= len(data.data[0])):
        if not( 0 < n_comp <= new_data[0].shape[1]):
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "features out of range"))

            self.model_d[PCA_MDL_KEY_MODEL].clear()

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
                print("Error: %s (line: %s) %s: %s" %
                    (__FILE__, ndt.__LINE__, ndt.__NAME__,
                    "negative eigenvalues"))
                self.model_d[PCA_MDL_KEY_MODEL].clear()
                return None, None

        try:
            eigval_in = np.linalg.inv(np.diag(eigvals ** (1/2)))

        except np.linalg.LinAlgError:

            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "singular matrix model is none"))

            self.model_d[PCA_MDL_KEY_MODEL].clear()

            return None, None

        t = eigvecs @ eigval_in
        self.model_d[PCA_MDL_KEY_MODEL][PCA_MDL_KEY_TRANS] = t

        # compute a transformation matrix for class-independent PCA
        #
        transf = np.identity(new_data[0].shape[1])

        # compute a goodness of fit measure: use the average weighted
        # mean-square-error computed across the entire data set
        #
        gsum = float(0.0)
        for i, d in enumerate(new_data):

           # iterate over the data in class i and calculate the norms

            _sum = float(0.0)
            for v in d:
                _sum += np.linalg.norm(transf * (v - means[i]))

            # weight by the prior
            #
            gsum += self.model_d[PCA_MDL_KEY_MODEL][PCA_MDL_KEY_PRIOR][i] * \
                _sum

        score = gsum / float(npts)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # check for model validity
        #
        if model[PCA_MDL_KEY_NAME] != PCA_NAME:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "incorrect model name"))
            return None, None

        if not model[PCA_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "model parameter is empty"))
            return None, None

        # get the number of dimensions
        #
        ndim = data.data.shape[1]

        # transform data to new environment
        #
        t =  model[PCA_MDL_KEY_MODEL][PCA_MDL_KEY_TRANS]
        mu = model[PCA_MDL_KEY_MODEL][PCA_MDL_KEY_MEANS]

        # calculate number of classes
        #
        num_classes = len(mu)
        mt = []
        for i in range(len(mu)):
            m_new = mu[i] @ t
            mt.append(m_new)

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
            sample = data.data[j] @ t
            count =0
            post =[]
            for k in range(num_classes):

                # manually compute the log likelihood
                # as a weighted Euclidean distance
                #
                # mean of class k
                #
                prior =  model[PCA_MDL_KEY_MODEL][PCA_MDL_KEY_PRIOR][k]

                # mean of class k
                #
                m = mt[k]

                # posterior calculation for sample j
                #
                # @: short-hand notation for matrix multiplication
                #
                g1 = (sample - m).T @ (sample - m)
                g2 = np.exp(-1/2 * g1)
                g = g2 * scale * prior
                count = count + g
                post.append(g)

            post = post/count


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
# end of PCA

#------------------------------------------------------------------------------
# ML Tools Class: LDA
#------------------------------------------------------------------------------

class LDA:
    """
    Class: LDA

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
         this is the default constructor for the class.
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
        self.params_d[LDA_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[LDA_PRM_KEY_PARAM] = defaultdict(dict)

        # set the names
        #
        self.model_d[LDA_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[LDA_MDL_KEY_MODEL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None

        # calculate number of classes
        #
        data = data.sort()
        uni_label = np.unique(data.labels)
        new_data =[]

        for i in range(len(uni_label)):

            class_data =[]

            for j in range(len(data.labels)):

                if uni_label[i]==data.labels[j]:
                    class_data.append(data.data[j])

            new_data.append(np.array(class_data))

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
        mode_prior = self.params_d[LDA_PRM_KEY_PARAM][LDA_PRM_KEY_PRIOR]
        if mode_prior == ALG_PRIORS_ML:

            # compute the priors for each class
            #
            priors = np.full(shape = num_classes,
                            fill_value = (float(1.0) / float(num_classes)))
            self.model_d[LDA_MDL_KEY_MODEL][LDA_MDL_KEY_PRIOR] = priors

        # if the priors are based on occurrences
        #
        elif mode_prior == ALG_PRIORS_MAP:

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
            self.model_d[LDA_MDL_KEY_MODEL][LDA_MDL_KEY_PRIOR] = priors

        else:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "unknown value for priors"))
            self.model_d[LDA_MDL_KEY_MODEL].clear()
            return None, None

        # calculate the means of each class:
        # note these are a list of numpy vectors
        #
        means = []
        for elem in new_data:
            means.append(np.mean(elem, axis = 0))

        self.model_d[LDA_MDL_KEY_MODEL][LDA_MDL_KEY_MEANS] = means

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
            sw += priors[i]*nct.compute(d, \
            ctype = self.params_d[LDA_PRM_KEY_PARAM][LDA_PRM_KEY_CTYPE],
            center= self.params_d[LDA_PRM_KEY_PARAM][LDA_PRM_KEY_CENTER],
            scale= self.params_d[LDA_PRM_KEY_PARAM][LDA_PRM_KEY_SCALE])

            # between class scatter calculation
            #
            mean_diff = (means[i] - mean_glob)
            sb += priors[i] * (mean_diff).dot(mean_diff.T)

        try:
            sw_in = np.linalg.inv(sw)

        except np.linalg.LinAlgError:

            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "singular matrix"))
            self.model_d[LDA_MDL_KEY_MODEL].clear()

            return None, None

        # calculation of sw^-1*sb
        #
        j = sw_in.dot(sb)

        # number of the eigenvectors need to be chosen
        # it is equal to the num_class minus 1
        #
        l = (len(new_data))-1

        # eigen vector and eigen value decomposition
        #
        eigvals, eigvecs = np.linalg.eig(j)

        # sorted eigenvalues and eigvecs and choose
        # the first l-1 columns from eigenvalues
        # and eigenvectors
        #
        sorted_indexes = eigvals.argsort()[::-1]
        eigvals = eigvals[sorted_indexes[0:l]]
        eigvecs = eigvecs[:,sorted_indexes[0:l]]
        if any(eigvals < 0):
                print("Error: %s (line: %s) %s: %s" %
                    (__FILE__, ndt.__LINE__, ndt.__NAME__,
                    "negative eigenvalues"))
                self.model_d[LDA_MDL_KEY_MODEL].clear()
                return None, None

        try:
            eigval_in = np.linalg.inv(np.diag(eigvals**(1/2)))

        except np.linalg.LinAlgError:

            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "singular matrix model is none"))

            self.model_d[LDA_MDL_KEY_MODEL].clear()

            return None, None

        # calculation of the transformation matrix
        #
        t = eigvecs @ eigval_in
        self.model_d[LDA_MDL_KEY_MODEL][LDA_MDL_KEY_TRANS] = t

        # compute a transformation matrix for LDA
        #
        transf = np.identity(new_data[0].shape[1])

        gsum = float(0.0)
        for i, d in enumerate(new_data):

            # iterate over the data in class i and calculate the norms
            #
            _sum = float(0.0)
            for v in d:
                _sum += np.linalg.norm(transf * (v - means[i]))

            # weight by the prior
            #
            gsum += self.model_d[LDA_MDL_KEY_MODEL][LDA_MDL_KEY_PRIOR][i] * \
                _sum

        score = gsum / float(npts)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # check for model validity
        #
        if model[LDA_MDL_KEY_NAME] != LDA_NAME:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "incorrect model name"))
            return None, None

        if not model[LDA_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "model parameter is empty"))
            return None, None

        # get the number of classes and number of features
        #
        ndim = data.data.shape[1]

        # transform data to new environment
        #
        t =  model[LDA_MDL_KEY_MODEL][LDA_MDL_KEY_TRANS]
        mu = model[LDA_MDL_KEY_MODEL][LDA_MDL_KEY_MEANS]
        num_classes = len(mu)
        mt = []
        for i in range(len(mu)):
            m_new = mu[i] @ t
            mt.append(m_new)

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

            d = data.data[j]
            d = d @ t
            count = 0
            post = np.zeros((1, num_classes))

            # loop over number of classes
            #
            for k in range(num_classes):

                # manually compute the log likelihood
                # as a weighted Euclidean distance
                #
                # mean of class k
                #
                prior =  model[LDA_MDL_KEY_MODEL][LDA_MDL_KEY_PRIOR][k]

                # mean of class k
                #
                m = mt[k]

                # posterior calculation for sample j
                #
                # @: short-hand notation for matrix multiplication
                #
                #
                g1 = (d - m).T @ (d - m)
                g2 = np.exp(-1/2 * g1)
                g = g2 * scale * prior
                count = count + g
                post[0,k] +=g
            post = post/count

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
# end of LDA

#------------------------------------------------------------------------------
# ML Tools Class: QDA
#------------------------------------------------------------------------------

class QDA:
    """
    Class: QDA

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
         this is the default constructor for the class.
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
        self.params_d[QDA_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[QDA_PRM_KEY_PARAM] = defaultdict(dict)

        # set the names
        #
        self.model_d[QDA_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[QDA_MDL_KEY_MODEL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
         QDA is implemented as class dependent PCA.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None

        data = data.sort()
        uni_label = np.unique(data.labels)
        new_data =[]
        for i in range(len(uni_label)):
            class_data =[]

            for j in range(len(data.labels)):
                if uni_label[i]==data.labels[j]:
                    class_data.append(data.data[j])

            new_data.append(np.array(class_data))

        ndim = new_data[0].shape[1]

        num_classes = len(new_data)

        # number of components
        #
        n_comp= int(self.params_d[QDA_PRM_KEY_PARAM][QDA_PRM_KEY_COMP])
        if not(0 < n_comp <= new_data[0].shape[1]):
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "features out of range"))
            self.model_d[QDA_MDL_KEY_MODEL].clear()
            return None, None

        # calculate number of classes
        #

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
        mode_prior = self.params_d[QDA_PRM_KEY_PARAM][QDA_PRM_KEY_PRIOR]
        if mode_prior == ALG_PRIORS_ML:

            # save priors for each class
            #
            self.model_d[QDA_MDL_KEY_MODEL][QDA_MDL_KEY_PRIOR] = \
                np.full(shape = num_classes,
                        fill_value = (float(1.0) / float(num_classes)))

        # if the priors are based on occurrences : nct.PRIORs_MAP
        #
        elif mode_prior == ALG_PRIORS_MAP:

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
            self.model_d[QDA_MDL_KEY_MODEL][QDA_MDL_KEY_PRIOR] = priors * sum

        else:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "unknown value for priors"))
            self.model_d[QDA_MDL_KEY_MODEL].clear()
            return None, None

        # calculate the means of each class:
        #  note these are a list of numpy vectors
        #
        means = []
        for elem in new_data:
            means.append(np.mean(elem, axis = 0))
        self.model_d[QDA_MDL_KEY_MODEL][QDA_MDL_KEY_MEANS] = means

        # calculate the cov:
        #  note this is a list of matrices, and each matrix for a class
        #
        # an empty list to save covariances
        #
        t = []

        # loop over classes to calculate covariance of each class
        #
        for i,element in enumerate(new_data):


            covar = nct.compute(element,
                ctype= self.params_d[QDA_PRM_KEY_PARAM][QDA_PRM_KEY_CTYPE],
                center= self.params_d[QDA_PRM_KEY_PARAM][QDA_PRM_KEY_CENTER],
                scale= self.params_d[QDA_PRM_KEY_PARAM][QDA_PRM_KEY_SCALE])

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
                print("Error: %s (line: %s) %s: %s" %
                    (__FILE__, ndt.__LINE__, ndt.__NAME__,
                    "negative eigenvalues"))
                self.model_d[QDA_MDL_KEY_MODEL].clear()
                return None, None

            try:
                eigvals_in = np.linalg.inv(np.diag(eigvals**(1/2)))

            except np.linalg.LinAlgError as e:
                print("Error: %s (line: %s) %s: %s" %
                    (__FILE__, ndt.__LINE__, ndt.__NAME__,
                    "singular matrix model is none"))
                self.model_d[QDA_MDL_KEY_MODEL].clear()
                return None, None

            # calculation of transformation matrix
            #
            trans = eigvecs @ eigvals_in
            t.append(trans)

        self.model_d[QDA_MDL_KEY_MODEL][QDA_MDL_KEY_TRANS] = t

        # compute a goodness of fit measure: use the average weighted
        # mean-square-error computed across the entire data set
        #
        transf = np.identity(new_data[0].shape[1])
        gsum = float(0.0)
        for i, d in enumerate(new_data):

            # iterate over the data in class i and calculate the norms
            #
            sum = float(0.0)
            for v in d:
                sum += np.linalg.norm(transf*(v - means[i]))

            # weight by the prior
            #
            gsum += self.model_d[QDA_MDL_KEY_MODEL][QDA_MDL_KEY_PRIOR][i] * \
                sum
        score = gsum / float(npts)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # check for model validity
        #
        if model[QDA_MDL_KEY_NAME] != QDA_NAME:
            print("Error: %s (line: %s) %s: %s" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__,
                   "incorrect model name"))
            return None, None

        if not model[QDA_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "model parameter is empty"))
            return None, None

        # get the number of features
        #
        ndim = data.data.shape[1]

        # transform data and mean to new space
        #
        t =  model[QDA_MDL_KEY_MODEL][QDA_MDL_KEY_TRANS]
        mu = model[QDA_MDL_KEY_MODEL][QDA_MDL_KEY_MEANS]

        # calculate number of classes
        #
        mt = []
        num_classes = len(mu)
        for i in range(len(mu)):
            m_new = mu[i] @ t[i]
            mt.append(m_new)

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
            count = 0
            post = []

            # loop over number of classes
            #
            for k in range(num_classes):

                sample = data.data[j] @ t[k]

                # manually compute the log likelihood
                # as a weighted Euclidean distance
                #
                # priors of class k
                #
                prior =  model[QDA_MDL_KEY_MODEL][QDA_MDL_KEY_PRIOR][k]

                # mean of class k
                #
                m = mt[k]

                # posterior calculation for sample j
                #
                # @ : short-hand notation for matrix multiplication
                #
                g1 = (sample - m).T @ (sample-m)
                g2 = np.exp(-1/2 * g1)
                g = g2 * scale * prior
                count = count + g
                post.append(g)

            post = post/count

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
    Class: QLDA

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
         this is the default constructor for the class.
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
        self.params_d[QLDA_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[QLDA_PRM_KEY_PARAM] = defaultdict(dict)

        # set the names
        #
        self.model_d[QLDA_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[QLDA_MDL_KEY_MODEL] = defaultdict(dict)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None

        # calculate number of classes
        #
        data = data.sort()
        uni_label = np.unique(data.labels)
        new_data =[]

        for i in range(len(uni_label)):

            class_data =[]

            for j in range(len(data.labels)):

                if uni_label[i]==data.labels[j]:
                    class_data.append(data.data[j])

            new_data.append(np.array(class_data))

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
        mode_prior = self.params_d[QLDA_PRM_KEY_PARAM][QLDA_PRM_KEY_PRIOR]
        if mode_prior == ALG_PRIORS_ML:

            # compute the priors for each class
            #
            priors = np.full(shape = num_classes,
                        fill_value = (float(1.0) / float(num_classes)))
            self.model_d[QLDA_MDL_KEY_MODEL][QLDA_MDL_KEY_PRIOR] = priors

        # if the priors are based on occurrences
        #
        elif mode_prior == ALG_PRIORS_MAP:

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
            self.model_d[QLDA_MDL_KEY_MODEL][QLDA_MDL_KEY_PRIOR] = priors

        else:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "unknown value for priors"))
            self.model_d[QLDA_MDL_KEY_MODEL].clear()
            return None, None

        # calculate the means of each class:
        # note these are a list of numpy vectors
        #
        means = []
        for elem in new_data:
            means.append(np.mean(elem, axis = 0))
        self.model_d[QLDA_MDL_KEY_MODEL][QLDA_MDL_KEY_MEANS] = means

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

        t = []

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
        for i,d in enumerate(new_data):

            # calculation of within class scatter for each class
            #
            sw = priors[i] * nct.compute(d, \
            ctype = self.params_d[QLDA_PRM_KEY_PARAM][QLDA_PRM_KEY_CTYPE],
            center = self.params_d[QLDA_PRM_KEY_PARAM][QLDA_PRM_KEY_CENTER],
            scale = self.params_d[QLDA_PRM_KEY_PARAM][QLDA_PRM_KEY_SCALE])

            # check singularity
            #
            try:
                sw_in = np.linalg.inv(sw)

            except np.linalg.LinAlgError:
                print("Error: %s (line: %s) %s: %s" %
                      (__FILE__, ndt.__LINE__, ndt.__NAME__,
                       "singular matrix"))
                self.model_d[QLDA_MDL_KEY_MODEL].clear()

                return None, None

            # calculation of sw^-1*sb for each class
            #
            j = sw_in.dot(sb)

            # eigen vector and eigen value decomposition
            # for each class
            #
            eigvals, eigvecs = np.linalg.eig(j)

            # sorted eigenvalues and eigvecs and choose
            # the first l-1 columns from eigenvalues
            # and eigenvectors
            #
            sorted_indexes = eigvals.argsort () [::-1]
            eigvals = eigvals[sorted_indexes[0:l]]
            eigvecs = eigvecs[:,sorted_indexes[0:l]]
            if any(eigvals < 0):
                print("Error: %s (line: %s) %s: %s" %
                    (__FILE__, ndt.__LINE__, ndt.__NAME__,
                    "negative eigenvalues"))
                self.model_d[QLDA_MDL_KEY_MODEL].clear()
                return None, None

            try:
                eigvals_in = np.linalg.inv(np.diag(eigvals**(1/2)))
                det = np.linalg.det(np.diag(eigvals**(1/2)))

            except np.linalg.LinAlgError:
                print("Error: %s (line: %s) %s: %s" %
                    (__FILE__, ndt.__LINE__, ndt.__NAME__,
                    "singular matrix model is none"))
                self.model_d[QLDA_MDL_KEY_MODEL].clear()
                return None, None

            # calculation of transformation matrix
            #
            trans = eigvecs @ eigvals_in
            t.append(trans)

        self.model_d[QLDA_MDL_KEY_MODEL][QLDA_MDL_KEY_TRANS] = t

        transf = np.identity(new_data[0].shape[1])
        gsum = float(0.0)
        for i, d in enumerate(new_data):

            # iterate over the data in class i and calculate the norms
            #
            sum = float(0.0)
            for v in d:
                sum += np.linalg.norm(transf*(v - means[i]))

            # weight by the prior
            #
            gsum += self.model_d[QLDA_MDL_KEY_MODEL][QLDA_MDL_KEY_PRIOR][i] * \
                sum
        score = gsum / float(npts)

        # exit gracefully
        #
        return self.model_d, score

    #
    # end of method

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        # check for model validity
        #
        if model[QLDA_MDL_KEY_NAME] != QLDA_NAME:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "incorrect model name"))
            return None, None

        if not model[QLDA_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "model parameter is empty"))
            return None, None

        # get the number of features
        #

        ndim = data.data.shape[1]

        # transform data and mean to new space
        #
        t =  model[QLDA_MDL_KEY_MODEL][QLDA_MDL_KEY_TRANS]
        mu = model[QLDA_MDL_KEY_MODEL][QLDA_MDL_KEY_MEANS]

        # get the number of classes
        #
        mt = []
        num_classes = len(mu)
        for i in range(len(mu)):
            #mu[i]= mu[i] @ t[i]
            m_new = mu[i]@t[i]
            mt.append(m_new)



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

            count = 0
            post = np.zeros((1, num_classes))
            d = data.data[j]

            # loop over number of classes
            #
            for k in range(num_classes):

                sample = d @ t[k]

                # manually compute the log likelihood
                # as a weighted Euclidean distance
                #
                # priors of class k
                #
                prior =  model[QLDA_MDL_KEY_MODEL][QLDA_MDL_KEY_PRIOR][k]

                # mean of class k
                #
                m = mt[k]

                # posterior calculation for sample j
                #
                # @ : short-hand notation for matrix multiplication
                #
                g1 = (d - m).T @ (d - m)
                g2 = np.exp(-1/2 * g1)
                g = g2 * scale * prior
                count = count + g
                post[0,k] += g

            post = post/count

            # choose the class label with the highest posterior
            #
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
    Class: NB

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
         this is the default constructor for the class.
        """

        # set the class name
        #
        NB.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d[NB_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[NB_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[NB_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[NB_MDL_KEY_MODEL] = GaussianNB()
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None

        # calculate number of classes
        #
        data = data.sort()
        uni_label = np.unique(data.labels)
        new_data =[]
        for i in range(len(uni_label)):
            class_data =[]
            for j in range(len(data.labels)):
                if uni_label[i]==data.labels[j]:
                    class_data.append(data.data[j])
            new_data.append(np.array(class_data))

        num_classes = len(new_data)

        # calculate number of data points
        #
        npts = 0
        for element in new_data:
            npts += len(element)

        # case: (ml) equal priors
        #
        mode_prior = self.params_d[NB_PRM_KEY_PARAM][NB_PRM_KEY_PRIOR]
        if mode_prior == ALG_PRIORS_ML:

            # compute the priors for each class
            #
            priors = np.full(shape = num_classes,
                            fill_value = (float(1.0) / float(num_classes)))

        # if the priors are based on occurrences
        #
        elif mode_prior == ALG_PRIORS_MAP:

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
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "unknown value for priors"))
            return None

        # make the final data
        #
        f_data = np.vstack((new_data))

        # getting the labels
        #
        labels = []
        for i in range(len(new_data)):
            for j in range(len(new_data[i])):
                labels.append(i)
        labels = np.array(labels)

        # fit the model
        #
        self.model_d[NB_MDL_KEY_MODEL] = GaussianNB(priors = priors).fit(f_data, labels)

        # prediction
        #
        ypred = self.model_d[NB_MDL_KEY_MODEL].predict(f_data)

        # score calculation using auc ( f1 score f1_score(y_true, y_pred, average=None))
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

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        data = np.array(data.data)
        if model is None:
            model = self.model_d

        p_labels = model[NB_MDL_KEY_MODEL].predict(data)

        # posterior calculation
        #
        posteriors = model[NB_MDL_KEY_MODEL].predict_proba(data)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of NB

#------------------------------------------------------------------------------
# ML Tools Class: GMM
#------------------------------------------------------------------------------

class GMM:
    """
    Class: GMM

    description:
     This is a class that implements classification using Gaussian Mixture
     Models (GMM). A separate GMM is trained for each class and the final
     decision is based on the posterior probability computed via Bayes' rule.
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
        GMM.__CLASS_NAME__ = self.__class__.__name__

        # Initialize variables for the parameter block and model
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # Initialize a parameter dictionary
        self.params_d[GMM_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[GMM_PRM_KEY_PARAM] = defaultdict(dict)

        # Set the model; we will store a dictionary of trained GMMs per class
        self.model_d[GMM_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[GMM_MDL_KEY_MODEL] = defaultdict(dict)

    #---------------------------------------------------------------------------
    # Computational methods: train / predict
    #---------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
         model: a dictionary of GMM models (and additional parameters such as
         priors)
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None


        # Identify unique labels and group the data by class
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

        # Get the number of mixture components from the parameters
        n_components= int(self.params_d[GMM_PRM_KEY_PARAM][GMM_PRM_KEY_COMP])
        if not(0 < n_components <= new_data[0].shape[1]):
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "features out of range"))
            self.model_d[GMM_MDL_KEY_MODEL].clear()
            return None, None

        # calculate number of data points
        #
        npts = 0
        for element in new_data:
            npts += len(element)

        # initialize an empty array to save the priors
        #
        priors = np.empty((0,0))

        # Determine the prior mode from parameters
        mode_prior = self.params_d[GMM_PRM_KEY_PARAM][GMM_PRM_KEY_PRIOR]
        if mode_prior == ALG_PRIORS_ML:
            # Equal priors for each class
            priors = np.full(shape=num_classes, fill_value=(1.0 / num_classes))
        elif mode_prior == ALG_PRIORS_MAP:
            # Priors based on the relative frequency of each class
            counts = np.array([len(class_data) for class_data in new_data])
            priors = counts / float(npts)
        else:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "unknown value for priors"))
            self.model_d[GMM_MDL_KEY_MODEL].clear()
            return None, None

        # Save the priors in the model dictionary
        self.model_d[GMM_MDL_KEY_MODEL][GMM_MDL_KEY_PRIOR] = priors

        random_state = \
            int(self.params_d[GMM_PRM_KEY_PARAM][GMM_PRM_KEY_RANDOM])

        # Train a GMM for each class
        class_models = {}
        for idx, class_data in enumerate(new_data):
            gm = GaussianMixture(n_components=n_components,
                                 random_state=random_state)
            gm.fit(class_data)
            class_models[idx] = gm

        # Save the trained class models and class labels
        self.model_d[GMM_MDL_KEY_MODEL][GMM_MDL_KEY_CLASS_MODELS] = class_models
        self.model_d[GMM_MDL_KEY_MODEL][GMM_MDL_KEY_CLASS_LABELS] = uni_label

        # Prepare a full data matrix and labels for computing the training score
        f_data = np.vstack(new_data)
        labels = []
        for idx, class_data in enumerate(new_data):
            labels.extend([idx] * len(class_data))
        labels = np.array(labels)

        # Prediction on the training data via posterior probability calculation:
        # For each sample, compute for each class:
        #    likelihood = exp(log-likelihood) * prior
        # then choose the class with the highest resulting value.
        ypred = []
        posteriors = []
        for x in f_data:
            likelihoods = []
            for idx in range(num_classes):
                # Compute log likelihood for sample x under the class's GMM
                log_likelihood = class_models[idx].score_samples(
                    x.reshape(1, -1))[0]
                likelihood = np.exp(log_likelihood) * priors[idx]
                likelihoods.append(likelihood)
            likelihoods = np.array(likelihoods)
            pred_label = np.argmax(likelihoods)
            ypred.append(pred_label)
            # Compute normalized posterior probabilities
            post = likelihoods / np.sum(likelihoods)
            posteriors.append(post)
        ypred = np.array(ypred)

        # Compute the macro-averaged f1 score on the training data
        score = f1_score(labels, ypred, average="macro")

        # Optionally, write the predicted training labels to file using the
        # MLToolsData.write() method
        if write_train_labels:
            data.write(oname=fname_train_labels, label=ypred)

        return self.model_d, score

    def predict(self,
                data: MLToolsData,
                model=None):
        """
        method: predict

        arguments:
         data: an MLToolsData object containing feature vectors (each row is a
          vector)
         model: a GMM model dictionary (if None, the internal model is used)

        return:
         p_labels: a numpy array of predicted labels
         posteriors: a numpy array of posterior probabilities (one row per sample)

        description:
         This method predicts the class labels for the given data using the
         trained GMM models.
         It computes the posterior probabilities for each sample and selects
         the class with the highest probability.
        """
        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        data = np.array(data.data)
        if model is None:
            model = self.model_d

        # retrieving the priors, class models and class labels from the trained
        # model
        #
        priors = model[GMM_MDL_KEY_MODEL][GMM_MDL_KEY_PRIOR]
        class_models = model[GMM_MDL_KEY_MODEL][GMM_MDL_KEY_CLASS_MODELS]
        class_labels = model[GMM_MDL_KEY_MODEL][GMM_MDL_KEY_CLASS_LABELS]
        num_classes = len(class_models)

        p_labels = []
        # Posterior calculation
        #
        posteriors = []
        for single_data in data:

            # Likelihoods calculation
            #
            likelihoods = []
            for idx in range(num_classes):
                # Log Likelihood calculation for each sample
                #
                log_likelihood = class_models[idx].score_samples(
                    single_data.reshape(1, -1))[0]
                likelihood = np.exp(log_likelihood) * priors[idx]
                likelihoods.append(likelihood)
            likelihoods = np.array(likelihoods)
            # predicting labels based on maximum likelihood
            #
            pred_label = np.argmax(likelihoods)
            p_labels.append(class_labels[pred_label])
            post = likelihoods / np.sum(likelihoods)
            posteriors.append(post)
        p_labels = np.array(p_labels)
        posteriors = np.array(posteriors)

        # exit gracefully
        #
        return p_labels, posteriors

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
    Class: KNN

    description:
     This is a class that implements KNN
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
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
        self.params_d[KNN_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[KNN_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[KNN_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[KNN_MDL_KEY_MODEL] = KNeighborsClassifier()
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # fit the model
        #
        n = int(self.params_d[KNN_PRM_KEY_PARAM][KNN_PRM_KEY_NEIGHB])
        self.model_d[KNN_MDL_KEY_MODEL] = KNeighborsClassifier(n_neighbors = n).fit(samples, labels)

        # prediction
        #
        ypred = self.model_d[KNN_MDL_KEY_MODEL].predict(samples)

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

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        samples = np.array(data.data)

        p_labels = model[KNN_MDL_KEY_MODEL].predict(samples)

        # posterior calculation
        #
        posteriors = model[KNN_MDL_KEY_MODEL].predict_proba(samples)

        return p_labels, posteriors
    #
    # end of method

#------------------------------------------------------------------------------
# ML Tools Class: KMEANS
#------------------------------------------------------------------------------

class KMEANS:
    """
    Class: KMEANS

    description:
     This is a class that implements KMEANS
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
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
        self.params_d[KMEANS_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[KMEANS_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[KMEANS_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[KMEANS_MDL_KEY_MODEL] = KMeans()
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # fit the model
        #
        n_cluster = int(self.params_d[KMEANS_PRM_KEY_PARAM][KMEANS_PRM_KEY_NCLUSTER])
        random_state =int(self.params_d[KMEANS_PRM_KEY_PARAM][KMEANS_PRM_KEY_RANDOM])
        n_init = int(self.params_d[KMEANS_PRM_KEY_PARAM][KMEANS_PRM_KEY_NINIT])
        m_iter = int(self.params_d[KMEANS_PRM_KEY_PARAM][KMEANS_PRM_KEY_MITER])

        self.model_d[KMEANS_MDL_KEY_MODEL] = KMeans(n_clusters=n_cluster,
        random_state = random_state,
        n_init = n_init,
        max_iter = m_iter).fit(samples)

        predicted_labels = self.model_d[KMEANS_MDL_KEY_MODEL].labels_

        score = silhouette_score(samples, predicted_labels)

        if write_train_labels:
            data.write(oname = fname_train_labels, label = predicted_labels)

        # exit gracefully
        #
        return self.model_d, score
    #
    # end of method

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d
        posteriors = []

        data = np.array(data.data)

        p_labels = model[KMEANS_MDL_KEY_MODEL].predict(data)

        # cluster centers
        #
        centers = model[KMEANS_MDL_KEY_MODEL].cluster_centers_

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
# end of KMEANS

#------------------------------------------------------------------------------
# ML Tools Class: RNF
#------------------------------------------------------------------------------

class RNF:
    """
    Class: RNF

    description:
     This is a class that implements RNF
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
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
        self.params_d[RNF_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[RNF_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[RNF_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[RNF_MDL_KEY_MODEL] = RandomForestClassifier()
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # fit the model
        #
        n_estimators = \
            int(self.params_d[RNF_PRM_KEY_PARAM][RNF_PRM_KEY_ESTIMATOR])

        max_depth = \
            int(self.params_d[RNF_PRM_KEY_PARAM][RNF_PRM_KEY_MAXDEPTH])

        criterion = \
            self.params_d[RNF_PRM_KEY_PARAM][RNF_PRM_KEY_CRITERION]

        random_state = \
            int(self.params_d[RNF_PRM_KEY_PARAM][RNF_PRM_KEY_RANDOM])

        self.model_d[RNF_MDL_KEY_MODEL] = \
            RandomForestClassifier(n_estimators = n_estimators,
                                   max_depth = max_depth,
                                   criterion = criterion,
                                   random_state= random_state).fit(samples, labels)

        # prediction
        #
        ypred = self.model_d[RNF_MDL_KEY_MODEL].predict(samples)

        # score calculation using auc ( f1 score f1_score(y_true, y_pred, average=None))
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

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        samples = np.array(data.data)

        p_labels = model[RNF_MDL_KEY_MODEL].predict(samples)

        # posterior calculation
        #
        posteriors = model[RNF_MDL_KEY_MODEL].predict_proba(samples)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of RNF

#------------------------------------------------------------------------------
# ML Tools Class: SVM
#------------------------------------------------------------------------------

class SVM:
    """
    Class: SVM

    description:
     This is a class that implements SVM
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
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
        self.params_d[SVM_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[SVM_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[SVM_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[SVM_MDL_KEY_MODEL] = SVC(probability = True)
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # fit the model
        #
        c = float(self.params_d[SVM_PRM_KEY_PARAM][SVM_PRM_KEY_C])
        gamma =float(self.params_d[SVM_PRM_KEY_PARAM][SVM_PRM_KEY_GAMMA])
        kernel = self.params_d[SVM_PRM_KEY_PARAM][SVM_PRM_KEY_KERNEL]

        self.model_d[SVM_MDL_KEY_MODEL] = SVC(C = c,
                        gamma = gamma,
                        kernel = kernel,
                        probability=True).fit(samples, labels)

        # prediction
        #
        ypred = self.model_d[SVM_MDL_KEY_MODEL].predict(samples)

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

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        samples = np.array(data.data)

        p_labels = model[SVM_MDL_KEY_MODEL].predict(samples)

        # posterior calculation
        #
        posteriors = model[SVM_MDL_KEY_MODEL].predict_proba(samples)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method
#
# end of SVM

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
    Class: MLP

    description:
     This is a class that implements MLP
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
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
        self.params_d[MLP_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[MLP_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[MLP_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[MLP_MDL_KEY_MODEL] = MLPClassifier()
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # fit the model
        #
        h_s = int(self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_HSIZE])
        act = self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_ACT]
        b_s = self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_BSIZE]
        sol = self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_SOLVER]
        lr = self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_LR]
        lr_init = float(self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_LRINIT])
        e_stop = bool(self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_STOP])
        sh = bool(self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_SHUFFLE])
        val= float(self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_VAL])
        m = float(self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_MOMENTUM])
        r_state = int(self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_RANDOM])
        m_iter = int(self.params_d[MLP_PRM_KEY_PARAM][MLP_PRM_KEY_MITER])

        self.model_d[MLP_MDL_KEY_MODEL] = \
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
        ypred = self.model_d[MLP_MDL_KEY_MODEL].predict(samples)

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

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        samples = np.array(data.data)

        p_labels = model[MLP_MDL_KEY_MODEL].predict(samples)

        # posterior calculation
        #
        posteriors = model[MLP_MDL_KEY_MODEL].predict_proba(samples)

        # exit gracefully
        #
        return p_labels, posteriors
    #
    # end of method

#
# end of MLP

#------------------------------------------------------------------------------
# ML Tools Class: RBM
#------------------------------------------------------------------------------

class RBM:
    """
    Class: RBM

    description:
     This is a class that implements RBM
    """

    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
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
        self.params_d[RBM_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[RBM_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[RBM_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[RBM_MDL_KEY_MODEL] = []
    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):
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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        if self.model_d[ALG_MDL_KEY_MODEL]:
            print("Error: %s (line: %s) %s: %s" %
                (__FILE__, ndt.__LINE__, ndt.__NAME__,
                "Doesn't support training on pre-trained model"))
            return None, None

        # making the final data
        #
        samples = np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # fit the model
        #
        n_comp = int(self.params_d[RBM_PRM_KEY_PARAM][RBM_PRM_KEY_COMP])
        n_iter = int(self.params_d[RBM_PRM_KEY_PARAM][RBM_PRM_KEY_NITER])
        random_state = int(self.params_d[RBM_PRM_KEY_PARAM][RBM_PRM_KEY_RANDOM])
        lr = float(self.params_d[RBM_PRM_KEY_PARAM][RBM_PRM_KEY_LR])
        b_size = int(self.params_d[RBM_PRM_KEY_PARAM][RBM_PRM_KEY_BSIZE])
        verbose = bool(self.params_d[RBM_PRM_KEY_PARAM][RBM_PRM_KEY_VERBOSE])
        classifier = str(self.params_d[RBM_PRM_KEY_PARAM][RBM_PRM_KEY_CLASSIF])
        rbm= BernoulliRBM(n_components=n_comp, learning_rate=lr,
                          batch_size=b_size, n_iter=n_iter,
                          verbose=verbose, random_state=random_state)

        self.model_d[RBM_MDL_KEY_MODEL] = \
            Pipeline(steps = [
                ('rbm', rbm),
                ('classifier', ALGS[classifier].model_d[ALG_MDL_KEY_MODEL])
            ]).fit(samples, labels)

        # prediction
        #
        ypred = self.model_d[RBM_MDL_KEY_MODEL].predict(samples)

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

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # check if model is none
        #
        if model is None:
            model = self.model_d

        samples = np.array(data.data)

        p_labels = model[RBM_MDL_KEY_MODEL].predict(samples)

        # posterior calculation
        #
        posteriors = model[RBM_MDL_KEY_MODEL].predict_proba(samples)

        return p_labels, posteriors
    #
    # end of method
#
# end of RBM

#------------------------------------------------------------------------------
# ML Tools Class: TRANSFORMER
#------------------------------------------------------------------------------

class TRANSFORMER:
    """
    Class: TRANSFORMER

    description:
     This is a class that implements TRANSFORMER
    """
    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
        """

        # set the class name
        #
        TRANSFORMER.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d[TRANS_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[TRANS_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[TRANS_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[TRANS_MDL_KEY_MODEL] = defaultdict(dict)


    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):

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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get the samples
        #
        samples =  np.array(data.data)

        # getting the labels
        #
        labels = data.labels

        # get the parameters
        #
        self.lr = float(self.params_d[TRANS_PRM_KEY_PARAM]
                                     [TRANS_PRM_KEY_LR])
        self.batch_size = int(self.params_d[TRANS_PRM_KEY_PARAM]
                                           [TRANS_PRM_KEY_BSIZE])
        self.embed_size = int(self.params_d[TRANS_PRM_KEY_PARAM]
                                           [TRANS_PRM_KEY_EMBED_SIZE])
        self.nheads = int(self.params_d[TRANS_PRM_KEY_PARAM]
                                       [TRANS_PRM_KEY_NHEADS])
        self.num_layers = int(self.params_d[TRANS_PRM_KEY_PARAM]
                                           [TRANS_PRM_KEY_NLAYERS])
        self.MLP_dim = int(self.params_d[TRANS_PRM_KEY_PARAM]
                                        [TRANS_PRM_KEY_MLP_DIM])
        self.dropout = float(self.params_d[TRANS_PRM_KEY_PARAM]
                                          [TRANS_PRM_KEY_DROPOUT])


        # create the model
        #
        self.model_d[TRANS_MDL_KEY_MODEL] = ntt.NEDCTransformer(
            input_dim=samples.shape[1], num_classes=len(np.unique(labels)),
            d_model=self.embed_size, nhead=self.nheads,
            num_layers=self.num_layers,dim_feedforward=self.MLP_dim,
            dropout=self.dropout)

        # get the epochs
        #
        self.epochs = int(self.params_d[TRANS_PRM_KEY_PARAM]
                          [TRANS_PRM_KEY_EPOCH])

        # initialize the accuracies
        #
        accuracies = []


        # get the corss entropy loss function and ADAM optimizer
        #
        criterion = self.model_d[TRANS_MDL_KEY_MODEL]\
        .get_cross_entropy_loss_function()
        optimizer = self.model_d[TRANS_MDL_KEY_MODEL]\
        .get_adam_optimizer(lr=self.lr)


        # train the model
        #
        for epoch in range(self.epochs):

            # set the model to train mode
            #
            self.model_d[TRANS_MDL_KEY_MODEL].train()

            # move the model to the default device
            #
            self.model_d[TRANS_MDL_KEY_MODEL] = \
                self.model_d[TRANS_MDL_KEY_MODEL] \
                    .to_device(self.model_d[TRANS_MDL_KEY_MODEL])

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
                output = self.model_d[TRANS_MDL_KEY_MODEL](current_samples)

                # calculate the loss
                #
                loss = criterion(output,
                                 self.model_d[TRANS_MDL_KEY_MODEL]\
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
        state_dict = self.model_d[
            TRANS_MDL_KEY_MODEL].state_dict()

        # assign the model to the model_d['model']
        #
        self.model_d[TRANS_MDL_KEY_MODEL] = state_dict

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

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        # get the samples
        #
        samples = np.array(data.data)

        # get the parameters
        #
        self.lr = float(self.params_d[TRANS_PRM_KEY_PARAM]
                                     [TRANS_PRM_KEY_LR])
        self.batch_size = int(self.params_d[TRANS_PRM_KEY_PARAM]
                                           [TRANS_PRM_KEY_BSIZE])
        self.embed_size = int(self.params_d[TRANS_PRM_KEY_PARAM]
                                           [TRANS_PRM_KEY_EMBED_SIZE])
        self.nheads = int(self.params_d[TRANS_PRM_KEY_PARAM]
                                        [TRANS_PRM_KEY_NHEADS])
        self.num_layers = int(self.params_d[TRANS_PRM_KEY_PARAM]
                                           [TRANS_PRM_KEY_NLAYERS])
        self.MLP_dim = int(self.params_d[TRANS_PRM_KEY_PARAM]
                                        [TRANS_PRM_KEY_MLP_DIM])
        self.dropout = float(self.params_d[TRANS_PRM_KEY_PARAM]
                                          [TRANS_PRM_KEY_DROPOUT])

        # create the model
        #
        trans_model = ntt.NEDCTransformer(input_dim=samples.shape[1],
                                          num_classes=data.num_of_classes,
                                          d_model=self.embed_size,
                                          nhead=self.nheads,
                                          num_layers=self.num_layers,
                                          dim_feedforward=self.MLP_dim,
                                          dropout=self.dropout)
        # move the model to the default device
        # CPU/GPU based on their availability
        #
        trans_model = trans_model.to_device(trans_model)

        # load the model's weights
        #
        trans_model.load_state_dict(self.model_d[TRANS_MDL_KEY_MODEL])

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
                current_samples = samples[batch:batch+self.batch_size]

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
# end of TRANSFORMER

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
    Class: QSVM

    description:
     This is a class that implements QSVM
    """
    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
        """

        # set the class name
        #
        QSVM.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d[QSVM_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[QSVM_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[QSVM_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[QSVM_MDL_KEY_MODEL] = defaultdict(dict)

    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):

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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get the samples
        #
        samples =  np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # get the parameters
        #
        self.mdl_name = str(self.params_d[QSVM_PRM_KEY_PARAM]
                                         [QSVM_PRM_KEY_MDL_NAME])
        self.provider_name = str(self.params_d[QSVM_PRM_KEY_PARAM]
                                              [QSVM_PRM_KEY_PROVIDER])
        self.hardware = str(self.params_d[QSVM_PRM_KEY_PARAM]
                                         [QSVM_PRM_KEY_HARDWAR])
        self.encoder = str(self.params_d[QSVM_PRM_KEY_PARAM]
                                        [QSVM_PRM_KEY_ENCODER])
        self.entanglement = str(self.params_d[QSVM_PRM_KEY_PARAM]
                                             [QSVM_PRM_KEY_ENTANGLEMENT])
        self.n_qubits = int(self.params_d[QSVM_PRM_KEY_PARAM]
                                         [QSVM_PRM_KEY_NQUBITS])
        self.feat_reps = int(self.params_d[QSVM_PRM_KEY_PARAM]
                                          [QSVM_PRM_KEY_FEAT_REPS])
        self.shots = int(self.params_d[QSVM_PRM_KEY_PARAM]
                                      [QSVM_PRM_KEY_SHOTS])
        self.kernel_name = str(self.params_d[QSVM_PRM_KEY_PARAM]
                                            [QSVM_PRM_KEY_KERNEL])

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
        self.model_d[QSVM_MDL_KEY_MODEL] = qsvm_model.fit(samples, labels)

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

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get the samples
        #
        samples = np.array(data.data)

        # get the trained model
        #
        model = self.model_d[QSVM_MDL_KEY_MODEL]

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
# end of QSVM

#------------------------------------------------------------------------------
# ML Tools Class: QNN
#------------------------------------------------------------------------------

class QNN:
    """
    Class: QNN

    description:
     This is a class that implements QNN
    """
    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
        """

        # set the class name
        #
        QNN.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d[QNN_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[QNN_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[QNN_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[QNN_MDL_KEY_MODEL] = defaultdict(dict)


    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):

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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get the samples
        #
        samples =  np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # get the parameters
        #
        self.mdl_name = str(self.params_d[QNN_PRM_KEY_PARAM]
                                         [QNN_PRM_KEY_MDL_NAME])
        self.provider_name = str(self.params_d[QNN_PRM_KEY_PARAM]
                                              [QNN_PRM_KEY_PROVIDER])
        self.hardware = str(self.params_d[QNN_PRM_KEY_PARAM]
                                         [QNN_PRM_KEY_HARDWAR])
        self.encoder = str(self.params_d[QNN_PRM_KEY_PARAM]
                                        [QNN_PRM_KEY_ENCODER])
        self.entanglement = str(self.params_d[QNN_PRM_KEY_PARAM]
                                             [QNN_PRM_KEY_ENTANGLEMENT])
        self.n_qubits = int(self.params_d[QNN_PRM_KEY_PARAM]
                                         [QNN_PRM_KEY_NQUBITS])
        self.feat_reps = int(self.params_d[QNN_PRM_KEY_PARAM]
                                          [QNN_PRM_KEY_REPS])

        self.ansatz = str(self.params_d[QNN_PRM_KEY_PARAM]
                                       [QNN_PRM_KEY_ANSATZ])
        self.ansatz_reps = int(self.params_d[QNN_PRM_KEY_PARAM]
                                            [QNN_PRM_KEY_ANSATZ_REPS])
        self.optim_name = str(self.params_d[QNN_PRM_KEY_PARAM]
                                            [QNN_PRM_KEY_OPTIM])
        self.optim_max_steps = int(self.params_d[QNN_PRM_KEY_PARAM]
                                                [QNN_PRM_KEY_MAXSTEPS])
        self.meas_type = str(self.params_d[QNN_PRM_KEY_PARAM]
                                          [QNN_PRM_KEY_MEAS_TYPE])

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
        self.model_d[QNN_MDL_KEY_MODEL] = qnn_model.fit(samples, labels)

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

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get the samples
        #
        samples = np.array(data.data)

        # get the trained model
        #
        model = self.model_d[QNN_MDL_KEY_MODEL]

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
# end of QNN

#------------------------------------------------------------------------------
# ML Tools Class: QRBM
#------------------------------------------------------------------------------

class QRBM:
    """
    Class: QRBM

    description:
     This is a class that implements QRBM
    """
    def __init__(self):
        """
        method: constructor

        arguments:
         none

        return:
         none

        description:
         this is the default constructor for the class.
        """

        # set the class name
        #
        QRBM.__CLASS_NAME__ = self.__class__.__name__

        # initialize variables for the parameter block and model
        #
        self.params_d = defaultdict(dict)
        self.model_d = defaultdict()

        # initialize a parameter dictionary
        #
        self.params_d[QRBM_PRM_KEY_NAME] = self.__class__.__name__
        self.params_d[QRBM_PRM_KEY_PARAM] = defaultdict(dict)

        # set the model
        #
        self.model_d[QRBM_MDL_KEY_NAME] = self.__class__.__name__
        self.model_d[QRBM_MDL_KEY_MODEL] = defaultdict(dict)


    #
    # end of method

    #--------------------------------------------------------------------------
    #
    # computational methods: train/predict
    #
    #--------------------------------------------------------------------------

    def train(self,
              data: MLToolsData,
              write_train_labels: bool,
              fname_train_labels: str,
              ):

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
            print("%s (line: %s) %s: training a model" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get the samples
        #
        samples =  np.array(data.data)

        # getting the labels
        #
        labels = np.array(data.labels)

        # get the parameters
        #
        self.mdl_name = str(self.params_d[QRBM_PRM_KEY_PARAM]
                                         [QRBM_PRM_KEY_MDL_NAME])
        self.provider_name = str(self.params_d[QRBM_PRM_KEY_PARAM]
                                              [QRBM_PRM_KEY_PROVIDER])
        self.n_hidden = int(self.params_d[QRBM_PRM_KEY_PARAM]
                                         [QRBM_PRM_KEY_NHIDDEN])
        self.shots = int(self.params_d[QRBM_PRM_KEY_PARAM]
                                      [QRBM_PRM_KEY_SHOTS])
        self.cs = int(self.params_d[QRBM_PRM_KEY_PARAM]
                                    [QRBM_PRM_KEY_CS])
        self.n_neighbors = int(self.params_d[QRBM_PRM_KEY_PARAM]
                                      [QRBM_PRM_KEY_N_NEIGHBORS])
        self.encoder = str(self.params_d[QRBM_PRM_KEY_PARAM]
                                      [QRBM_PRM_KEY_ENCODER])

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
        self.model_d[QRBM_MDL_KEY_MODEL] = qrbm_knn_model.fit(samples, labels)

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

    def predict(self,
                data: MLToolsData,
                model = None):
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
            print("%s (line: %s) %s: entering predict" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        # get the samples
        #
        samples = np.array(data.data)

        # get the number of visible node which is the number of features
        # in the dataset
        #
        self.n_visible = samples[0].shape[0]

        # get the trained model
        #
        model = self.model_d[QRBM_MDL_KEY_MODEL]

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
# end of QRBM

#******************************************************************************
#
# Section 5: definitions dependent on the above classes go here
#
#******************************************************************************

# define variables to configure the machine learning algorithms
#
ALGS = {EUCLIDEAN_NAME:EUCLIDEAN(), PCA_NAME:PCA(), LDA_NAME:LDA(),
        QDA_NAME:QDA(), QLDA_NAME:QLDA(), NB_NAME:NB(), GMM_NAME:GMM(),
        KNN_NAME:KNN(), KMEANS_NAME:KMEANS(), RNF_NAME:RNF(), SVM_NAME:SVM(),
        MLP_NAME:MLP(), RBM_NAME:RBM(), TRANSFORMER_NAME:TRANSFORMER(),
        QSVM_NAME:QSVM(), QNN_NAME:QNN(), QRBM_NAME:QRBM()}

#
# end of file

#------------------------------------------------------------------------------

