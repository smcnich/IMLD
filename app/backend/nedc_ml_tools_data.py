#!/usr/bin/env python
#
# file: $NEDC_NFC/class/python/nedc_ml_tools_data/nedc_ml_tools_data.py
#
# revision history:
# 20250304 (JP): reviewed and updated for the new release
# 20241014 (SM): initial version
#
# This class encapsulates data that is to be used for ML Tools.
#------------------------------------------------------------------------------

# import required system modules
#
from collections import defaultdict
import copy
import os
import numpy as np
import pandas as pd

# import required NEDC modules
#
import nedc_debug_tools as ndt

# declare global debug and verbosity objects so we can use them
# in both functions and classes
#
dbgl_g = ndt.Dbgl()
vrbl_g = ndt.Vrbl()

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define names of distributions so they can be generated by the class
#
GAUSSIAN = "gaussian"
TORODIAL = "toroidal"
YIN_YANG = "yin_yang"

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def generate_toroidal(mean:list,
                      cov:list,
                      npts_mass:int,
                      npts_ring:int,
                      inner_rad:float,
                      outer_rad:float) -> tuple:
    """
    function: generate_toroidal

    args:
     mean (1D list)   : the mean values for the distribution. should have the 
                        same length as the number of features. 
                        (e.g. [1, 2, 3] for 3 features)
     cov  (2D list)   : the covariance matrix for the inner gaussian mass
     npts_mass (int)  : the number of points to generate for the inner mass
     npts_ring (int)  : the number of points to generate for the ring
     inner_rad (float): the inner radius of the ring
     outer_rad (float): the outer radius of the ring

    return:
     X (np.ndarray): a n-D array containing all of the data points 
                              generated. should contain the data for both 
                              classes.
     y (list): a 1-D list containing the labels for each sample in X  

    description:
     generate a torodial distribution. This includes an inner gaussian mass
     and a hollow ring outside of it. This will only have two labels, but works
     with N-dimensionality.
    """

    # error check for bad parameters
    # (PM): needs to display our standard messages
    #
    if not (outer_rad > inner_rad):
        raise ValueError("Outer radius must be greater than inner radius")
    if (inner_rad < 0) or (outer_rad < 0):
        raise ValueError("Radius values cannot be less than 0")

    # generate inner gaussian mass (class 0)
    #
    X, y = generate_gaussian([{'npts': npts_mass, 'mean': mean, 'cov': cov}])

    # get the number of dimensions
    #
    dims = len(mean)

    # generate random points on an N-dimensional unit sphere
    #
    random_vectors = np.random.normal(size=(npts_ring, dims))
    unit_vectors = random_vectors / np.linalg.norm(random_vectors,
                                                   axis = 1, keepdims = True)

    # scale vectors to lie within the toroidal ring (between inner_rad
    # and outer_rad). create the toroidal ring (class 1)
    #
    ring_radii = np.random.uniform(inner_rad, outer_rad,
                                   npts_ring).reshape(-1, 1)
    class_1_data = np.array(mean).flatten() + unit_vectors * ring_radii

    # concatenate data w labels
    #
    X = np.vstack((X, class_1_data))
    y += ([1] * npts_ring)
    
    # exit gracefully
    #
    return X, y

#
# end of function

def generate_yin_yang(means:list,
                      radius:float,
                      npts_yin: int,
                      npts_yang: int,
                      overlap: float) -> tuple:
    """
    function: generate_yin_yang

    args:
     means (1D list): the mean values for each feature. Should be of length 2 
                      (2D) or 3 (3D). For example, [0, 0] for 2D or 
                      [0, 0, 0] for 3D.
     radius (float) : the radius of the yin-yang circle
     npts_yin (int) : the number of points in the yin class (class 0)
     npts_yang (int): the number of points in the yang class (class 1)
     overlap (float): the amount of overlap between the yin and yang classes

    return:
     X (np.ndarray): a n-D array containing all of the data points 
                              generated. should contain the data for both 
                              classes.
     y (list): a 1-D list containing the labels for each sample in X 

    description:
     generate a yin-yang distribution. The yin class is defined in one half of
     the circle (or extruded sphere) and the yang class in the other half. 
     when working in 3D, the pattern is extruded along the z-axis (with z 
     sampled from a similar normal distribution) while keeping the yin-yang 
     decision based on the x and y coordinates.
    """

    # determine the dimensionality from the length of means
    #
    ndim = len(means)
    if ndim not in [2, 3]:
        raise ValueError("Please provide means for 2D or 3D data.")
    
    # use the provided means for the x and y (and z if available)
    #
    if ndim == 2:
        xmean, ymean = np.array(means).flatten()
    else:  # ndim == 3
        xmean, ymean, zmean = np.array(means).flatten()

    # boundary, mean, and standard deviation of plot
    #
    stddev_center = 1.5 * (radius) / 2

    # calculate radii for yin-yang regions
    #
    radius1 = radius / 2
    radius2 = radius / 4

    # create empty lists for storing points
    #
    yin = []
    yang = []

    # counters to track generated points for each class
    #
    n_yin_counter = 0
    n_yang_counter = 0

    # generate points for yin and yang
    #
    while n_yin_counter < npts_yin or n_yang_counter < npts_yang:

        # generate x and y coordinates
        #
        xpt = np.random.normal(xmean, stddev_center)
        ypt = np.random.normal(ymean, stddev_center)

        # generate z coordinate if 3D
        #
        if ndim == 3:
            zpt = np.random.normal(zmean, stddev_center)

        # calculate distances for each generated point
        #
        distance1 = np.sqrt(xpt ** 2 + ypt ** 2)
        distance2 = np.sqrt(xpt ** 2 + (ypt + radius2) ** 2)
        distance3 = np.sqrt(xpt ** 2 + (ypt - radius2) ** 2)

        # Determine point class based on position and distances
        #
        if distance1 <= radius1:

            #(PM) needs explanation
            #
            if -radius1 <= xpt <= 0:

                #(PM) needs explanation
                #
                if ((distance1 <= radius1 or distance2 <= radius2) and
                    distance3 > radius2):

                    #(PM) needs explanation
                    #
                    if n_yin_counter < npts_yin:

                        #(PM) needs explanation
                        #
                        if ndim == 2:
                            yin.append([xpt, ypt])
                        else:
                            yin.append([xpt, ypt, zpt])
                        n_yin_counter += 1

                #(PM) needs explanation
                #
                elif n_yang_counter < npts_yang:

                    #(PM) needs explanation
                    #
                    if ndim == 2:
                        yang.append([xpt, ypt])
                    else:
                        yang.append([xpt, ypt, zpt])
                    n_yang_counter += 1

            #(PM) needs explanation
            #
            elif 0 < xpt <= radius1:
                if ((distance1 <= radius1 or distance3 <= radius2) and
                    distance2 > radius2):

                    #(PM) needs explanation
                    #
                    if n_yang_counter < npts_yang:

                        #(PM) needs explanation
                        #
                        if ndim == 2:
                            yang.append([xpt, ypt])
                        else:
                            yang.append([xpt, ypt, zpt])
                        n_yang_counter += 1

                #(PM) needs explanation
                #
                elif n_yin_counter < npts_yin:
                    if ndim == 2:
                        yin.append([xpt, ypt])
                    else:
                        yin.append([xpt, ypt, zpt])
                    n_yin_counter += 1

    # translate yin and yang points along the y-axis to adjust overlap.
    # in 3D, we leave the z coordinate unchanged.
    #
    if ndim == 2:
        yin = np.array(yin) + np.array([0, overlap * radius2])
        yang = np.array(yang) - np.array([0, overlap * radius2])
    else:
        yin = np.array(yin) + np.array([0, overlap * radius2, 0])
        yang = np.array(yang) - np.array([0, overlap * radius2, 0])

    # return generated data as a dictionary:
    #  combine the yin and yang classes and create the labels
    #
    X = np.concatenate((yin, yang), axis=0)
    y = [0] * npts_yin + [1] * npts_yang

    # exit gracefully
    #
    return X, y

#
# end of function 

def generate_gaussian(params:list) -> tuple:
    """
    function: generate_gaussian

    args:
     params (list): a list containing a dictionary of parameters for each
                    gaussian distribution to make.
                     params = [
                        {
                        'npts' (int): the number of points for the distribution
                        'mean' (1D list): the mean values for the distribution.
                                          should have the same length as the 
                                          number of features. 
                                          (e.g. [1, 2, 3] for 3 features)
                        'cov' (2D list): the covariance matrix for the 
                                         distribution. should be a square 
                                         matrix with the same dimensions as 
                                         the number of features.
                                         (e.g. [[0.1, 0.01, 0.02],
                                                [0.01, 0.1, 0.03],
                                                [0.02, 0.03, 0.1]] 
                                         for 3 features)
                        },
                        ...
                     ] 

    return:
     X (np.ndarray): a n-D array containing all of the data points 
                              generated. should contain the data for both 
                              classes.
     y (list): a 1-D list containing the labels for each sample in X   

     description:
      generate a gaussian distribution for a given number of labels. Works for 
      N-dimensionality. The number of features is determined by the length of 
      the mean and covariance matrix.
    """

    # make sure parameters are provided
    # (PM) use our standard error checking
    #
    if len(params) == 0:
        raise ValueError("No parameters provided for gaussian distribution.")

    # iterate through the parameters for each gaussian distribution
    # and generate the data and labels for each distribution
    #
    for i, param in enumerate(params):

        # get parameters
        # (PM) use our standard error checking
        #
        try:
            npts, mean, cov = param['npts'], param['mean'], param['cov']
        except KeyError as e:
            missing_param = e.args[0]
            raise KeyError(
                f"Missing parameter '{missing_param}' for distribution {i}.")

        # check if the mean and covariance matrix are valid
        # (PM) use our standard error checking
        #
        if len(mean) != len(cov) or len(cov) != len(cov[0]):
            raise ValueError(
                "Mean and covariance matrix dimensions do not match.")
        
        # gaussian distribution for each class
        #
        data = np.random.multivariate_normal(np.array(mean).flatten(),
                                             cov, npts)
        labels = [i] * npts

        # if this is the first iteration, set the class data and labels
        # to the data and labels generated in this iteration
        #
        if i == 0:
            X = data
            y = labels

        # if this is not the first iteration, append the data and labels
        # to the class data and labels
        #
        else:

            # check if the dimensions of the previous class data and the
            # current data match. if not, raise an error
            # (PM) use our standard error checking
            #
            if X.shape[1] != data.shape[1]:
                raise ValueError("Data dimensions do not match.")
            
            # if the dimensions match, append the data and labels to the
            # class data and labels
            #
            else:
                X = np.vstack((X, data))
                y += labels

    # exit gracefully
    #
    return X, y

#
# end of function

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

class MLToolsData:
    """
    Class: MLToolsData

    arguments:
     none

    description:
     This is a class that encapsulates data that can be used with ML Tools.
    """

    def __init__(self, dir_path = "", lndx = 0, nfeats = -1):
        """
        method: constructor

        arguments:
         dir_path: directory path to the file ("")
         lndx: the label index (0)
         nfeats: number of features (-1)

        return:
         none

        description:
         none

        note:
         for nfeats, -1 means that we choose all of the features.
        """

        # set the class name
        #
        MLToolsData.__CLASS_NAME__ = self.__class__.__name__

        # set internal variables
        #
        self.dir_path = dir_path
        self.lndx = lndx
        self.nfeats = nfeats

        self.data = []
        self.labels = []
        self.num_of_classes = 0
        self.mapping_label = {}

        if (dir_path != ""):
            self.load()

    #
    # end of method

    def __repr__(self) -> str:
        """
        method: __repr__

        arguments: none

        return:
         a format specification that determines the default format for
         displaying an object
        
        description:
         this method sets the default format for printing an object
        """
        
        # exit gracefully
        #
        return (
            f"MLToolData({self.dir_path}, "
            f"label index = {self.lndx}, "
            f"# of features = {self.nfeats if self.nfeats != -1 else 'all'})"
        )
    #
    # end of method

    @classmethod
    def generate_data(cls, dist_name:str, params):
        """
        method: generate_data

        arguments:
         dist_name: name of the distribution to generate data from
                    ('gaussian', 'toroidal', 'yin_yang')
         params: the parameters for the distribution. Should be a list of
                 dictionaries if guassian. Should be a dictionary if toroidal
                 or yin-yang.

        return:
         a MLToolsData object populated with the data from the distribution
        
        description:
         generate data according to a specific set of parameters
        """

        # generate the data for the distribution
        #
        if dist_name == GAUSSIAN: 
            if not isinstance(params, list):
                raise ValueError(
                    "Gaussian parameters must be a list of dictionaries.")
            else:
                X, y = generate_gaussian(params)

        elif dist_name == TORODIAL: 
            if not isinstance(params, dict):
                raise ValueError("Toroidal parameters must be a dictionary.")
            else:
                X, y = generate_toroidal(**params)

        elif dist_name == YIN_YANG:
            if not isinstance(params, dict):
                raise ValueError("Yin-Yang parameters must be a dictionary.")
            else:
                X, y = generate_yin_yang(**params)

        # exit gracefully:
        #  take the data and labels and create a new MLToolsData object
        #  exit gracefully
        #
        return cls.from_data(X, y)
    #
    # end of method

    @classmethod
    def from_data(cls, X:np.ndarray, y:list):
        """
        method: from_data

        argument:
         X (np.ndarray): the data to be used. can be a N-dimensional array
         y (list)      : a 1-D list of labels for the data. should be the same
                         length as the number of rows in X.

        return:
         a MLToolData object

        description:
         this method creates a new ML Tools Data object from a numpy array
         and a list of labels. This is useful for creating a class object
         from data that is not in a file.
        """

        # initialize data
        #
        self = cls.__new__(cls)
        self.dir_path = ""
        self.lndx = 0
        self.nfeats = -1
        self.num_of_classes = len(set(y))

        # save the data and labels
        #
        self.labels = np.asarray(y)
        self.data = np.asarray(X)

        # create the mapping label
        #
        self.mapping_label = {i: label for i, label in enumerate(set(y))}

        # convert the labels to numbers
        #
        self.labels = self.map_label()

        # save the mapped labels back into ints
        #
        self.labels = np.array(self.labels, dtype = int)

        # exit gracefully:
        #  return the MLToolsData object
        #
        return self
    #
    # end of method

    @staticmethod
    def is_excel(fname):
        """
        method: is_excel

        arguments:
         fname: filename of the data

        return:
         a boolean value indicating status

        description:
         This method checks if file is an excel spreadsheet.
        """

        # use Pandas to open and parse the file. if this errors,
        # we assume it is a csv file.
        #
        try:
            pd.read_excel(fname)
        except ValueError:
            return False

        # exit gracefully
        #
        return True

    #
    # end of method

    def map_label(self, labels:np.array=None):
        # -> type[list[_T]] | ndarray | NDArray:

        """
        method: map_label

        arguments: (PM)
         fname: filename of the data

        return: (PM)
         a labels...

        description: (PM)
         This method checks if file is an excel spreadsheet.
        """

        # (PM): explain this
        #
        if labels is None:
            labels = np.array(self.labels)
            unique_labels = np.unique(labels)

        else:
            labels = np.array(labels)
            unique_labels = np.unique(labels)

        # (PM): explain this
        #
        for i in range(len(unique_labels)):
            for j in range(len(labels)):
                if labels[j] == unique_labels[i]:
                    labels[j]=i

        # exit gracefully
        #
        return labels

    #
    # end of method

    def load(self):
        """
        method: load_data

        arguments: None

        return:
         a list of numpy arrays or None if it fails

        description:
         This function reads data from either an excel sheet or csv file
         and converts it to a dictionary representing the labels and the data.

        Ex: data: {
            "labels": numpy.ndarray[0, 0, 0, 1, 1, 1, 1],
            "data"  : [np.ndarray[01,02,03],
                    [04,05,06],
                    [07,08,09],
                    [60,61,62],
                    [70,71,72],
                    [80,81,82],
                    [90,91,92]]
        }

        The example data above has 2 classes and 3 features.The labels ordering
        and data ordering are the same. The first three vectors are in
        class "0" and the last four are in class "1".

        for nfeats, it will use all the feature from the start to the
        specified value not counting the label column.

        Ex: if nfeats = 3, then we assume column [0,1,2].

        Ex: If we have [0,1,2,3,4,5] and lndx = 1, nFeatures = 3 then the
            column features would be [0,2,3] since we exclude the column label.

        If the data fails to be loaded, an error is generated and
        None is returned.
        """

        # display an informational message
        #
        if dbgl_g == ndt.FULL:
            print("%s (line: %s) %s: reading data" %
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))

        try:
            if self.is_excel(self.dir_path):
                df = pd.read_excel(self.dir_path, header = None)
            else:
                df = pd.read_csv(self.dir_path, header = None,
                                 engine = "c", comment = "#")
        except Exception:
            raise("Error: %s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   MLToolsData.__CLASS_NAME__, ndt.__NAME__,
                   "unknown file or data format", self.dir_path))

        if self.lndx >= df.shape[1]:
            print("Error: %s (line: %s) %s::%s: %s" %
                  (__FILE__, ndt.__LINE__,
                   MLToolsData.__CLASS_NAME__, ndt.__NAME__,
                   "Label index out of range"))
            return None

        # pop the label column
        #
        label_column = df.pop(self.lndx)

        # clear label map if there was one already
        #
        if not self.mapping_label:
            self.mapping_label.clear()

        # create a label map for readable label to an index.
        #  Note: Since we are sorting, the mapping will not always be in
        #  order (PM) if string because sorting uses string comparison
        #
        for ind, val in enumerate(sorted(label_column.unique())):

            # (PM): needs explanation
            #
            if isinstance(val, str):
                self.mapping_label[ind] = val

            # assume any label that is not a string to be a integer
            #
            else:
                self.mapping_label[ind] = int(val)

        # (PM): needs explanation

        if self.nfeats >= df.shape[1] or self.nfeats < -1:
            self.nfeats = -1

        # if the number of feature is specified, then we need to reshape the
        # data frame
        #
        if self.nfeats != -1:
            df = df.iloc[:, : self.nfeats]

        # append the label column at the beginning of the dataframe
        # and rename its column
        #
        df = pd.concat([label_column, df], axis = 1)
        df.columns = list(range(df.shape[1]))

        # set the index of the table using the label column
        #
        df.set_index(df.keys()[0], inplace = True)

        # (PM): needs explanation
        #
        self.data = df.values
        self.labels = df.index.to_numpy()
        self.num_of_classes = len(set(self.labels))

        # (PM) exit gracefully --- what does this return???
        #
        return

    #
    # end of method

    def sort(self, inplace = False):
        """
        method: sort

        arguments:
         inplace: flag to sort the data inplace (False)

        return:
         If inplace = True -> returns None
         If inplace = False -> returns the sorted data

        description:
         This function sorts the given data model.
        """

        # samples and labels
        #
        samples = self.data
        labels = np.array(self.labels)

        # np.unique() returns a set of unique values that
        # is in order
        #
        uniq_labels = np.unique(labels)

        # empty list to save sorted data snd labels
        #
        sorted_data = []
        sorted_labels = []

        # loop through the unique labels
        #
        for element in uniq_labels:

            # empty list to save class labels and class data
            #
            class_data = []
            class_labels = []

            # loop through the len labels and compare labels with unique label
            #
            for i in range(len(labels)):
                if labels[i] == element:
                    class_data.append(samples[i])
                    class_labels.append(labels[i])

            # (PM): needs explanation
            #
            sorted_data.extend(class_data)
            sorted_labels.extend(class_labels)

        # (PM): needs explanation
        #
        sorted_data = np.array(sorted_data)
        sorted_labels = np.array(sorted_labels)

        # exit gracefully:
        #  return None if in place
        #  return the sorted data if not in place
        #
        if inplace:
            self.data = sorted_data
            self.labels = sorted_labels

            return None

        else:
            MLToolDataNew = copy.deepcopy(self)
            MLToolDataNew.data = sorted_data
            MLToolDataNew.labels = sorted_labels

            return MLToolDataNew

    #
    # end of method

    def write(self, oname, label):
        """
        method: write

        argument:
         oname: the output file name
         label: the label to write

        return:
         a boolean indicating the status

        description:
         This function writes the data with new label to a file.
        """

        # (PM): needs explanation
        #
        d = pd.DataFrame(self.data)

        #  add the label to the first column of the file
        #
        try:
            d.insert(0, column = "labels", value = label)
        except ValueError:
            print("Error: %s (line: %s) %s::%s: %s" %
                (__FILE__, ndt.__LINE__,
                 MLToolsData.__CLASS_NAME__, ndt.__NAME__,
                "Labels column already existed within the data"))
            return False

        # (PM): needs explanation
        #
        if self.is_excel(self.dir_path):
            d.to_excel(oname)
        else:
            d.to_csv(oname, index = False, header = False)

        # exit gracefully
        #
        return True

    #
    # end of method

    def group_by_class(self):
        """
        method: group_by_class

        argument:
         none

        return:
         group data

        description:
         This function group the data by the label.
        """

        # (PM): needs explanation
        #
        group_data = defaultdict(list)

        # (PM): needs explanation
        #
        for label, data in zip(self.labels, self.data):
            group_data[label].append(data)

        # exit gracefully
        #
        return group_data

    #
    # end of method

#
# end of MLToolsData
    
#
# end of file
