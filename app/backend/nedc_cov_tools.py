#!/usr/bin/env python

# file: $(NEDC_NFC)/class/python/nedc_ml_tools/nedc_cov_tools.py
#
# revision history: 
#
# 20250404 (JP): refactored and reviewed
# 20230623 (AB): code refactored to new comment style
# 20230309 (SS): added an option for scaling
# 20230226 (JP): final review
# 20230222 (SS): updated implementations
# 20230218 (JP): refactored the code based on what we have learned recently
# 20230113 (JP): changed the interface to handle shrinkage properly
# 20230110 (JP): converted to functions and simplified things
# 20230106 (JP): refactored
# 20230106 (SS): initial version
#
# This file contains an implementation of several popular techniques to
# compute a covariance matrix. There are three basic options: ctype,
# center and scale:
#   - ctype: full (def) or diagonal
#   - center: none, tied or untied (def)
#   - scale:  none, biased (def), unbiased or empirical
#
# A good reference on this can be found here:
#
#  Hertzog, C. (1986). On Pooling Covariance Matrices for Multivariate
#  Analysis. Educational and Psychological Measurement, 46(2), 349–352.
#  https://doi.org/10.1177/001316448604600208
#
# The techniques used here integrate approaches found in popular tools
# like JMP (https://www.jmp.com/en_us/home.html), which is based on SAS,
# an scikit-learn (https://scikit-learn.org/stable/). These options
# should support all known ways to compute the covariance based
# on what we have seen in Python.
#
# The shrinkage algorithm implemented here comes from scikit-learn:
#
#  https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html
#
# and is described here:
#
#  W. Rayens and T. Greene, Covariance pooling and stabilization for
#  classification, Computational Statistics & Data Analysis, Volume 11,
#  Issue 1, 1991, Pages 17-42, ISSN 0167-9473,
#  https://doi.org/10.1016/0167-9473(91)90050-C.
#
#------------------------------------------------------------------------------

# import required system modules
#
import numpy as np
import os
import sys

# import various machine learning tools
#
from sklearn import covariance as sklc

# import required NEDC modules
#
import nedc_debug_tools as ndt
import nedc_file_tools as nft

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
PRM_CTYPE = "ctype"
CTYPE_FULL = "full"
CTYPE_DIAG = "diagonal"
DEF_CTYPE = CTYPE_FULL

PRM_CENTER = "center"
CENTER_NONE = "none"
CENTER_TIED = "tied"
CENTER_UNTIED = "untied"
DEF_CENTER = CENTER_UNTIED

PRM_SCALE = "scale"
SCALE_NONE = "none"
SCALE_BIASED = "biased"
SCALE_UNBIASED = "unbiased"
SCALE_EMP = "empirical"
DEF_SCALE = SCALE_BIASED

# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()

#------------------------------------------------------------------------------
#
# functions listed here
#
#------------------------------------------------------------------------------

def compute(data, ctype = DEF_CTYPE, center = DEF_CENTER, scale = DEF_SCALE):
    """
    function: compute

    arguments:
     data: a list of matrices
     ctype: covariance type
     center: the type of centering
     scale: method of scaling

    return:
     a covariance matrix or None (if it fails)

    description:
     If the scaling is SCALE_EMP, we call a separate function. Otherwise,
     we handle it in this function.
    """

    # display informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s::%s: ctype = %s, center = %s, scale = %s" %
              (__FILE__, ndt.__LINE__, Covar.__CLASS_NAME__, ndt.__NAME__,
               ctype, center, scale))

    # check for empirical scaling
    #
    if scale == SCALE_EMP:
        return compute_emp(data, ctype, center, scale)

    # make a single matrix from the entire data set:
    #  collapse the individual classes into one data set
    #
    whole_data = np.vstack((data))

    # branch on the type of centering:
    #
    # case 1: no centering or debiasing (NONE)
    #
    if center == CENTER_NONE:
        return calculate(whole_data, ctype, scale)

    # case 2: center using a global mean (TIED)
    #
    elif center == CENTER_TIED:
        data_center = debias(whole_data)
        return calculate(data_center, ctype, scale)

    # case 3: center using a mean per class (UNTIED)
    #
    elif center == CENTER_UNTIED:

        # loop over all the data and center using a class-specific mean
        #
        db = []
        for d in data:
            db.append(debias(d))

        # calculate the covariance
        #
        return calculate(np.vstack((db)), ctype, scale)

    # exit ungracefully: unknown mode
    #
    else:
        return None
#
# end of function

#------------------------------------------------------------------------------
#
# supporting functions go here
#
#------------------------------------------------------------------------------

def compute_emp(data, ctype, center, scale):
    """
    function: compute_emp

    arguments:
     data: a list of matrices
     ctype: covariance type
     center: the type of centering
     scale: method of scaling

    return:
     a covariance matrix or None (if it fails)

    description:
     This function computes an empirical covariance estimate.
    """

    # display informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s::%s: ctype = %s, center = %s, scale = %s" %
              (__FILE__, ndt.__LINE__, Covar.__CLASS_NAME__, ndt.__NAME__,
               ctype, center, scale))

    # case 1: no centering
    #
    if center == CENTER_NONE:
        return calculate(data, ctype, scale)

    # case 2: tied
    #
    elif center == CENTER_TIED:

        # subtract a class-specific mean
        #
        mu = np.mean (np.vstack((data)), axis = 0)
        for i in range(len(data)):
            data[i] = data[i] - mu

        # compute the covariance
        #
        return calculate(data, ctype, center, scale)

    # case 3: untied
    #
    elif center == CENTER_UNTIED:

        # loop over the data in each class and subtract the data from
        # the mean of the class
        #
        db = []
        for d in data:
            db.append(debias(d))

    # calculate the covariance
    #
    return calculate(db, ctype, scale = SCALE_EMP)

    # exit ungracefully: we should never reach this line
    #
    return None
#
# end of function

def debias(data):
    """
    function: debias

    arguments:
     data: a numpy matrix of data

    return:
     the debiased data

    description:
     This function computes the mean and debiases the data by subtracting
     the mean.
    """

    # display informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s::%s: debiasing data" %
              (__FILE__, ndt.__LINE__, Covar.__CLASS_NAME__, ndt.__NAME__))

    # calculate the mean of data
    #
    mean = np.mean(data, axis = 0)

    # subtract the mean
    #
    data_debiased = data - mean

    # exit gracefully
    #
    return data_debiased
#
# end of function

def calculate(data, ctype , scale):
    """
    function: calculate

    arguments:
     data: a matrix
     ctype: covariance type (full, diagonal)
     scale : type of scaling (e.g., biased)

    return:
     a covariance matrix or None (if it fails)

    description:
     This function calculates a raw covariance from the data. it has two
     modes - full and diagonal. note that subtraction of the mean (debiasing)
     is done within scikit-learn.
    """

    # display informational message
    #
    if dbgl > ndt.BRIEF:
        print("%s (line: %s) %s::%s: ctype = %s, scale = %s" %
              (__FILE__, ndt.__LINE__, Covar.__CLASS_NAME__, ndt.__NAME__,
               ctype, scale))

    # compute the covariance
    #
    # if scale is biased, ctype is full
    #
    if scale == SCALE_BIASED and ctype == CTYPE_FULL:
        cov = np.cov(data.T, bias = True)

    # if scale is biased and ctype is diag
    #
    elif scale == SCALE_BIASED and ctype == CTYPE_DIAG:
        cov = np.diag (np.var(data, ddof = 0, axis = 0))

    # if scale unbiased and ctype full
    #
    elif scale == SCALE_UNBIASED and ctype == CTYPE_FULL:
        cov = np.cov(data.T, bias = False)

    # if scale is unbiased and ctype is diag
    #
    elif scale == SCALE_UNBIASED and ctype == CTYPE_DIAG:
        cov = np.diag (np.var(data, ddof = 1, axis = 0))

    # if scale is none and ctype is full
    #
    elif scale == SCALE_NONE and ctype == CTYPE_FULL:
        data = debias(data)
        cov = data.T @ data

    # if scale is none and ctype is diag
    #
    elif scale == SCALE_NONE and ctype == CTYPE_DIAG:
        data = debias(data)
        cov = np.diag(np.diag(data.T @ data))

    # if scale is empirical
    #
    elif  scale == SCALE_EMP:

        # initialize a variable to add covariance of each class
        #
        cov = 0

        # calculate empirical weights
        #
        weights = calculate_empirical_weights(data)

        for i in range(len(data)):

            # ctype full
            #
            if ctype == CTYPE_FULL:

                # calculate covariance
                #
                cov += weights[i] * np.cov(data[i].T, bias = False)

            # ctype is diagonal
            #
            elif ctype == CTYPE_DIAG:

                # calculate covariance
                #
                cov += weights[i] * np.diag(np.var(data[i],ddof = 1, axis = 0))

    # exit gracefully
    #
    return cov
#
# end of function

def calculate_empirical_weights(data):
    """
    function: calculate_empirical_weights

    arguments:
     data: a matrix

    return:
     a numpy vector containing the empirical priors

    description:
     This function calculates the priors from the data using
     an empirical estimate:

     weight = (Nt - 1) / (N - num_classes)

     This formulation is popular in tools like JMP.
    """

    # calculate the number of classes
    #
    num_classes = len(data)

    # initialize an output vector
    #
    weights = np.zeros(num_classes)

    # calculate the total amount of data in each class, and set the
    # weights to the numerator term (Nt - 1).
    #
    num_data = int(0)
    for i in range(num_classes):
        npts = data[i].shape[0]
        num_data += npts
        weights[i] = (npts - int(1));

    # normalize
    #
    weights *= (float(1.0) / float(num_data - num_classes))

    # exit gracefully
    #
    return weights
#
# end of function

#
# end of file
