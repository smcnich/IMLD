#!/usr/bin/env python

# file: $(NEDC_NFC)/class/python/nedc_qml_tools/nedc_qml_tools.py
#
# revision history: 
#
# 20250203 (SP): initial version
#
# This file contains the implementation of the nedc_qml_tools module. This
# module contains classes and functions that are used to implement quantum 
# machine learning algorithms. Available quantum models include QSVM and QNN.
#  
#------------------------------------------------------------------------------

# import reqired system modules
#
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Any, Optional, Dict

# import reqired third-party modules
#
from sklearn.svm import SVC as SVM
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score

# import required NEDC modules
#
import nedc_debug_tools as ndt
import nedc_qml_base_providers_tools as nbpt
import nedc_qml_providers_tools as nqpt
import nedc_qml_tools_constants as const


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
# classes listed here
#
#------------------------------------------------------------------------------

class QML:
    """
    Quantum Machine Learning (QML) class that implements quantum algorithms.
    
    This class provides a high-level interface for quantum machine learning 
    algorithms. It manages the quantum provider and model selection, and provides
    methods for training and prediction using quantum circuits.
    
    The class follows a modular design where different quantum providers 
    (e.g., Qiskit, Pennylane) and models (QSVM, QNN) can be plugged in
    through a registry system.
    """
    def __init__(self, **kwargs):
        """
        method: constructor
        
        arguments:
         **kwargs: configuration parameters for QML setup
        
        return: 
         none
        
        description:
         initializes the QML class with specified parameters
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating QML instance with params: %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(kwargs)))
        
        self.provider = None
        self.model = None
        
        # create a new QMLParams instance with the provided parameters and set
        # them as the global parameters for the QML instance 
        #
        self.params = nbpt.QMLParams(**kwargs)
        
        # set the quantum provider and model based on the provided parameters
        #
        self.set_parameters()
        
    # end of constructor
    #
    
    def __repr__(self) -> str:
        """
        method: __repr__
        arguments: none
        return: str
        description: returns string representation of the QML instance
        """
        return f"{self.__class__.__name__}(provider={self.provider.__class__.__name__}, \
               model={self.model.__class__.__name__})"
    
    def set_parameters(self):
        """
        method: set_parameters
        arguments: 
         none
        return: 
         none
        description: 
         configures the quantum provider and model based on parameters
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: setting parameters for provider=%s, model=%s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, 
                   self.params.provider_name, self.params.model_name))
        
        # set parameters in the registry 
        #
        Registry.set_params(self.params)
        
        # get the provider instances from the registry, return back a concrete
        # class of QuantumProvider interface, e.g., QiskitProvider based on the
        # provider name in the parameters
        #
        self.provider = Registry.get_provider(self.params.provider_name)
        
        # get the model instances from the registry, return back a concrete 
        # class of QuantumModel interface, e.g., QSVM based on the model name
        # in the parameters
        #
        self.model = Registry.get_model(self.params.model_name)
        
        # set the parameters for the provider instance
        #
        self.provider.set_parameters(self.params)
        
        # set the provider for the model instance 
        #
        self.model.set_provider(self.provider)
        
        # exit gracefully
        #
    
    def fit(self, X, y=None):
        """
        method: fit
        arguments:
         X: training data features
         y: training data labels
        return: model
        description: trains the quantum model on provided data
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: fitting model with data shape: %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(X.shape)))
        
        # call the fit method of the model instance to train the model
        #
        self.model.fit(X, y)
        
        # exit gracefully: return the model instance
        #
        return self.model
    
    def predict(self, X, y=None):
        """
        method: predict
        arguments:
         X: test data features
         y: test data labels
        return: predictions
        description: predicts labels for test data using the trained model
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: predicting labels for data shape: %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(X.shape)))
        
        # call the predict method of the model instance to predict labels
        #
        predictions = self.model.predict(X, y)
        
        # exit gracefully: return the predictions
        #
        return predictions

    def score(self, X, y):
        """
        method: score
        arguments:
         X: test data features
         y: test data labels
        return: score
        description: computes the accuracy score for the model
        """ 
        
        # call the predict method of the model instance to get predicted labels
        #
        predictions = self.predict(X)
        
        # compute the accuracy score using the predicted labels
        #
        accuracy = accuracy_score(y, predictions)
        
        # compute the error rate
        #
        error_rate = 1 - accuracy
        
        # exit gracefully: return the error rate
        #
        return error_rate * const.PERCENTAGE, predictions
                     
#
# end of QML

class QSVM(nbpt.QuantumModel):
    """
    Quantum Support Vector Machine (QSVM) implementation.
    
    This class implements a quantum kernel-based SVM classifier using
    quantum circuits to compute kernel values between data points.
    The quantum kernel replaces classical kernels to leverage quantum
    feature spaces for classification.
    """
    def __init__(self):
        """
        method: constructor
        arguments: none
        return: none
        description: initializes QSVM with precomputed kernel
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: initializing QSVM" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # call the constructor of the parent class which is QuantumModel
        #
        super().__init__()
        
        # create a new SVM instance with precomputed kernel
        #
        self.svm = SVM(kernel=const.SVM_KERNEL_PRECOMPUTED)
    
    # end of constructor
    #
    
    def __repr__(self) -> str:
        """
        method: __repr__
        arguments: none
        return: str
        description: returns string representation of QSVM instance
        """
        return f"{self.__class__.__name__}(provider={self.provider.__class__.__name__ if self.provider else None})"
    
    def fit(self, X, y=None):
        """
        method: fit
        arguments:
         X: training data features
         y: training data labels
        return: none
        description: trains QSVM using quantum kernel
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: computing kernel and fitting QSVM" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        # compute the kernel matrix using the given provider
        #
        K = self.provider.compute_kernel_value(X)
        
        # fit the SVM model using the kernel matrix
        #
        self.svm.fit(K, y)
        
        # exit gracefully
        #
        
    def predict(self, X, y=None):
        """
        method: predict
        arguments:
         X: test data features
         y: test data labels
        return: predictions
        description: predicts labels using QSVM
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: predicting labels using QSVM" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        # compute the kernel matrix using the given provider
        #
        K = self.provider.compute_kernel_value(X, y)
        
        # exit gracefully: return the predicted labels
        #
        return self.svm.predict(K)       
#
# end of QSVM

class QNN(nbpt.QuantumModel):
    """
    Quantum Neural Network (QNN) implementation.
    
    This class implements a quantum neural network using parameterized
    quantum circuits. It supports various quantum architectures and 
    optimization methods for training the quantum circuit parameters.
    """
    def __init__(self):
        """
        method: constructor
        arguments: none
        return: none
        description: initializes QNN classifier
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: initializing QNN" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # call the constructor of the parent class which is QuantumModel
        #
        super().__init__()
        
        # set the _qnn_classifier to None, here _qnn_classifier is a private
        # variable that will be used to store the QNN classifier instance. It's
        # private because it's not meant to be accessed directly from outside
        # the class. To access it, we have a property qnn_classifier defined
        # later in the class.
        #
        self._qnn_classifier = None
        
    # end of constructor
    #
    
    def __repr__(self) -> str:
        """
        method: __repr__
        arguments: none
        return: str
        description: returns string representation of QNN instance
        """
        return f"{self.__class__.__name__}(provider={self.provider.__class__.__name__ if self.provider else None})"
    
    
    # the property decorator is used to define a getter method for the private
    # variable _qnn_classifier. This method is called qnn_classifier and it
    # returns the QNN classifier instance. This is a way to access the private
    # variable _qnn_classifier from outside the class.
    @property
    def qnn_classifier(self):
        """
        method: qnn_classifier
        arguments: none
        return: QNN classifier
        description: returns the QNN classifier instance
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: getting QNN classifier property value" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # if the QNN classifier instance is not set, then get it from the
        # provider instance. The provider instance is set in the parent class
        # QuantumModel. The provider instance is used to get the QNN classifier
        # instance.
        #
        if self._qnn_classifier is None:
            # check if the provider is set, if not raise an ValueError
            #
            if self.provider is None:
                raise ValueError("Provider not set. Call set_provider() first")
            
            # get the QNN classifier instance from the provider
            #
            self._qnn_classifier = self.provider.get_qnn_classifier()
        
        # exit gracefully: return the QNN classifier instance
        #
        return self._qnn_classifier
        
    def fit(self, X, y=None):
        """
        method: fit
        arguments:
         X: training data features
         y: training data labels
        return: none
        description: trains QNN classifier
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: fitting QNN classifier" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # call the fit method of the QNN classifier instance to train the model
        #
        self.qnn_classifier.fit(X, y)
        
        # exit gracefully
        #
        
    def predict(self, X, y=None):
        """
        method: predict
        arguments:
         X: test data features
         y: test data labels
        return: predictions
        description: predicts labels using QNN classifier
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: predicting labels using QNN classifier" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # exit gracefully: call the predict method of the QNN classifier
        # instance to predict labels 
        # 
        return self.qnn_classifier.predict(X)

#
# end of QNN

class QRBMClassifier(nbpt.QuantumModel):
    """Quantum Restricted Boltzmann Machine (QRBM) implementation.
    This class implements a quantum version of the Restricted Boltzmann Machine
    using quantum simulated annelealing provided by DWave to model the probability distribution of data.
    The QRBM can be used for generative tasks and feature learning.
    """
    def __init__(self):
        """
        method: constructor
        arguments: none
        return: none
        description: initializes QRBM classifier
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: initializing QRBM" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # call the constructor of the parent class which is QuantumModel
        #
        super().__init__()
        
        # set the _qrbm_encoder to None, here _qrbm_encoder is a private
        # variable that will be used to store the QRBM encoder instance.  It's
        # private because it's not meant to be accessed directly from outside
        # the class. To access it, we have a property qrbm_encoder defined
        # later in the class.
        #
        self._qrbm_encoder = None
        
        # set the _model to None, here _model is a private variable that will #
        # be used to store the KNN classifier instance.  It's private because
        # it's not meant to be accessed directly from outside the class. 
        # To access it, we have a property model defined later in the class.
        # 
        self._model = None
        
    # end of constructor
    
    def __repr__(self) -> str:
        """
        method: __repr__
        arguments: none
        return: str
        description: returns string representation of QRBM instance
        """
        return f"{self.__class__.__name__}(provider={self.provider.__class__.__name__ if self.provider else None})"
    
    # the property decorator is used to define a getter method for the private
    # variable _qrbm_encoder. This method is called qrbm_encoder and it
    # returns the QRBM classifier instance. This is a way to access the private
    # variable _qrbm_encoder from outside the class.
    @property
    def qrbm_encoder(self):
        """
        method: qrbm_encoder
        arguments: none
        return: QRBM feature encoder object
        description: returns the QRBM encoder instance
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: getting QRBM encoder property value" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # if the QRBM encoder instance is not set, then get it from the
        # provider instance. The provider instance is set in the parent class
        # QuantumModel. 
        #
        if self._qrbm_encoder is None:
            # check if the provider is set, if not raise an ValueError
            #
            if self.provider is None:
                raise ValueError("Provider not set. Call set_provider() first")
            
            # get the QRBM encoder instance from the provider
            #
            self._qrbm_encoder = self.provider.get_qrbm_feature_encoder()
        
        # exit gracefully: return the QRBM encoder instance
        #
        return self._qrbm_encoder
    
    @property
    def model(self):
        """
        method: model
        arguments: none
        return: KNN classifier
        description: returns the KNN classifier instance
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: getting KNN classifier property value" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # if the KNN classifier instance is not set, then create a new instance
        # of KNN classifier with the number of neighbors specified in the 
        # parameters.
        #
        if self._model is None:
            if dbgl == ndt.FULL:
                print("%s (line: %s) %s: creating KNN classifier instance with n=%s" %
                    (__FILE__, ndt.__LINE__, ndt.__NAME__, self.provider.params.n_neighbors))
                
            self._model = KNN(n_neighbors=self.provider.params.n_neighbors)
        
        # exit gracefully: return the KNN classifier instance
        #
        return self._model
    
    def fit(self, X, y=None):
        """
        method: fit
        arguments:
         X: training data features
         y: training data labels
        return: none
        description: trains QRBM classifier
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: fitting QRBM classifier" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # call the fit method of the QRBM encoder instance to encode the
        # features 
        #
        self.qrbm_encoder.fit(X)
        
        # compute the hidden representations using the QRBM encoder
        # 
        hidden_representations = self.qrbm_encoder.compute_layer_activations(X, bias = self.qrbm_encoder.hidden_bias)
        
        # train the KNN model using the hidden representations and labels
        #
        self.model.fit(hidden_representations, y)
        
        # exit gracefully
        #
        
    def predict(self, X, y=None):
        """
        method: predict
        arguments:
         X: test data features
         y: test data labels
        return: predictions
        description: predicts labels using QRBM classifier
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: predicting labels using QRBM classifier" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # compute the hidden representations using the QRBM encoder
        #
        hidden_representations = self.qrbm_encoder.compute_layer_activations(X,
                                      bias=self.qrbm_encoder.hidden_bias)
        
        # exit gracefully: call the predict method of the QRBM classifier
        # instance to predict labels 
        # 
        return self.model.predict(hidden_representations)

class Registry:
    """
    Registry for managing quantum providers and models.
    
    This class acts as a central registration system (like a phone book) that
    keeps track of all available quantum computing providers (like Qiskit,
    Pennylane) and quantum models (like QSVM, QNN). It follows the Registry
    pattern combined with Factory pattern to create and manage quantum computing components.
    
    key features:
    - maintains lists of available quantum providers and models
    - stores global parameters used across the quantum system
    - creates new instances of providers and models on demand
    - provides centralized access to quantum computing components
    
    class variables:
    _providers (Dict[str, Dict]): dictionary storing provider name -> provider class mappings
    _models (Dict[str, Dict]): dictionary storing model name -> model class mappings
    _params (QMLParams): global parameters for quantum computation
    
    example:
    ---------------------------------------------------------------------------
    # register a provider
    Registry.register_provider("qiskit", QiskitProvider)
    
    # register a model
    Registry.register_model("qsvm", QSVM)
    
    # get a provider instance
    provider = Registry.get_provider("qiskit")
    """
    # stores quantum providers (e.g., Qiskit, Pennylane)
    #
    _providers: Dict[str, Dict] = {}
    # stores quantum models (e.g., QSVM, QNN)
    #  
    _models: Dict[str, Dict] = {}  
    
    # stores global parameters for quantum computations  
    # 
    _params: Optional[nbpt.QMLParams] = None  
    
    # @classmethod decorator is used to create methods that operate on the 
    # class itself rather than instances of the class. These methods receive 
    # class as their first argument (cls) instead of self.
    #
    # Key benefits:
    # 1. can access/modify class state that applies across all instances 
    # 2. can be called on the class without creating an instance
    # 3. useful for alternative constructors or factory methods
    #
    # Example:
    #   class MyClass:
    #       _shared_data = []
    #       
    #       @classmethod
    #       def add_data(cls, item):
    #           cls._shared_data.append(item)
    #
    #   MyClass.add_data(5)  # Called directly on class
    #   instance = MyClass()
    #   instance.add_data(10) # Also works on instances
    #
    @classmethod
    def set_params(cls, params: nbpt.QMLParams):
        """        
        method: set_params
        
        arguments:
         params (QMLParams): configuration parameters including:
          - provider settings (e.g., backend selection)
          - model parameters (e.g., number of qubits)
          - optimization settings
        
        return: none
        
        description: 
         stores the provided parameters in the registry so they can be used
         when creating new provider instances. Think of this like setting
         global configuration that all quantum components will use.
        
        example:
        -----------------------------------------------------------------------
         params = QMLParams(n_qubits=2, backend='simulator')
         Registry.set_params(params)
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: setting registry params: %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(params)))
        
        # store the parameters in the registry, now cls._params will have the
        # provided parameters that can be used later to create provider
        # instances with these parameters. _params is a private variable 
        # because it's not meant to be accessed directly from outside the 
        # class. 
        #
        cls._params = params

    @classmethod
    def register_provider(cls, name: str, provider: nqpt.QuantumProvider):
        """        
        method: register_provider
        
        arguments:
         name (str): unique identifier for the provider (e.g., "qiskit")
         provider (QuantumProvider): the provider class to register
        
        return: none
        
        description:
         Adds a new quantum computing provider to the registry. This is like
         adding a new entry to a phone book. The provider class will be used
         later to create provider instances when needed.
        
        example:
        -----------------------------------------------------------------------
         Registry.register_provider("qiskit", QiskitProvider)
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: registering provider %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, name))
            
        # store the provider class in the registry
        #
        cls._providers[name] = provider

    @classmethod
    def register_model(cls, name: str, model: nbpt.QuantumModel):
        """        
        method: register_model
        
        arguments:
         name (str): unique identifier for the model (e.g., "qsvm")
         model (QuantumModel): the model class to register
        
        return: none
        
        description:
         Adds a new quantum model to the registry. Similar to register_provider,
         but for quantum models instead of providers. This allows the system
         to know what models are available for use.
        
        example:
        -----------------------------------------------------------------------
         Registry.register_model("qsvm", QSVM)
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: registering model %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, name))
            
        # store the model class in the registry
        #
        cls._models[name] = model

    @classmethod
    def get_provider(cls, name: str):
        """        
        method: get_provider
        
        arguments:
         name (str): name of the provider to instantiate, such as "qiskit"
        
        return:
         QuantumProvider: a new instance of the requested provider
        
        description:
         Looks up the provider class in the registry and creates a new
         instance with the current global parameters. This is like a
         factory that creates new provider objects on demand.
        
        example:
        -----------------------------------------------------------------------
         provider = Registry.get_provider("qiskit")
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating provider instance for %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, name))
        
        # get the provider class from the registry
        #
        provider_class = cls._providers.get(name)
        
        # exit gracefully: create and return a new instance with a 
        # provider instance, e.g., QiskitProvider
        #
        return provider_class(params=cls._params)

    @classmethod
    def get_model(cls, name: str):
        """        
        method: get_model
        
        arguments:
         name (str): name of the model to instantiate, such as "qsvm"
        
        return:
         QuantumModel: a new instance of the requested model
        
        description:
         Looks up the model class in the registry and creates a new
         instance. Similar to get_provider, but for quantum models.
         The created model can then be configured with a provider
         for quantum computations.
        
        example:
        -----------------------------------------------------------------------
         model = Registry.get_model("qsvm")
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating model instance for %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, name))
        
        # get the model class from the registry
        #
        model_class = cls._models.get(name)
        
        # exit gracefully: create and return a new model instance e.g., QSVM
        #
        return model_class()

#
# end of Registry

#------------------------------------------------------------------------------
#
# register if a new quantum model is added to the system
#
#------------------------------------------------------------------------------
Registry.register_model(const.MODEL_NAME_QSVM, QSVM)
Registry.register_model(const.MODEL_NAME_QNN, QNN)
Registry.register_model(const.MODEL_NAME_QRBM, QRBMClassifier)


#------------------------------------------------------------------------------
#
# register if a new quantum library/provider is added to the system
#
#------------------------------------------------------------------------------
Registry.register_provider(const.PROVIDER_NAME_QISKIT, nqpt.QiskitProvider)
Registry.register_provider(const.PROVIDER_NAME_DWAVE, nqpt.DWaveProvider)


#
# end of file
#------------------------------------------------------------------------------

if __name__=='__main__':
    import numpy as np
    samples =  np.array([
                    [-0.54585317,  0.45872878],
                    [ 0.19559004, -0.41765453],
                    [ 0.37622609, -1.21337474],
                    [ 0.41991704, -2.62169056],
                    [ 0.72306177, -0.89542115],
                    [ 4.82940720,  6.71913411],
                    [ 7.57186778,  4.80661997],
                    [ 5.08151120,  5.45632660],
                    [ 4.15677117,  6.52675315],
                    [ 5.44251527,  5.21502550]
                    ])

    labels = np.array(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
    qrbm = QML(provider_name=const.PROVIDER_NAME_DWAVE, 
           model_name=const.MODEL_NAME_QRBM,
           shots=2, 
           n_hidden=2, 
           n_visible=2, 
           epochs=2, 
           lr=0.1, 
           n_neighbors=2,
           encoder_name=const.ENCODER_NAME_QRBM_FEATURE_ENCODER)


    qrbm.fit(samples, labels)
    err_rate, predictions = qrbm.score(samples, labels)
    print(f"[QNN] Predictions: {predictions}")
    print(f"[QNN] Error Rate: {err_rate:.2f}%")
