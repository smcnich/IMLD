#!/usr/bin/env python

# file: $(NEDC_NFC)/class/python/nedc_qml_tools/nedc_qml_base_providers_tools. #py
#
# revision history: 
#
# 20250203 (SP): initial version
#
# This file contains the base classes and hardware configuration class for the
# quantum machine learning providers  
#  
#------------------------------------------------------------------------------

# import reqired system modules
#
from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from typing import Dict, Type, Any, Optional

# import reqired third party modules
#
import numpy as np

# import required NEDC modules
#
import nedc_debug_tools as ndt
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

# @dataclass decorator is used to create a data class. This class is used to
# store the parameters that are used to configure the quantum machine learning
# algorithms. It's better to use data classes to store the parameters because
# they are immutable and can be used to store the parameters that are used to
# configure the quantum machine learning algorithms. The __post_init__ method
# is used to validate and preprocess the parameters that are passed to the data class.
#
@dataclass
class QMLParams:
    model_name: str = const.DEFAULT_MODEL_NAME
    provider_name: str = const.DEFAULT_PROVIDER_NAME
    hardware: str = const.DEFAULT_HARDWARE_NAME
    encoder_name: str = const.DEFAULT_ENCODER_NAME
    entanglement: str = const.DEFAULT_ENTANGLEMENT
    kernel_name: str = const.DEFAULT_NONE_VALUE
    measurement_type: str = const.DEFAULT_MESUREMENT_TYPE
    
    # Optional[...] is a shorthand notation for Union[..., None], telling the 
    # type checker that either an object of the specific type is required, or
    # None is allowed.
    #
    kernel_name:  Optional[str] = const.DEFAULT_NONE_VALUE
    ansatz:  Optional[str] = const.DEFAULT_NONE_VALUE
    noise_model: Optional[str] = const.DEFAULT_NOISE_MODEL
    optim_name: Optional[str] = const.DEFAULT_NONE_VALUE
    
    n_classes: int = const.DEFAULT_NUM_CLASSES
    optim_max_steps: int = const.DEFAULT_OPTIM_MAX_STEPS
    n_qubits: int = const.DEFAULT_N_QUBITS
    shots: int = const.DEFAULT_SHOTS
    featuremap_reps: int = const.DEFAULT_REPS
    ansatz_reps: int = const.DEFAULT_REPS
    epochs: int = const.DEFAULT_EPOCHS
    
    # the following parameters are used for the QRBM
    #
    n_visible: int = const.DEFAULT_N_VISIBLE
    n_hidden: int = const.DEFAULT_N_HIDDEN
    chain_strength: float = const.DEFAULT_CHAIN_STRENGTH
    lr: float = const.DEFAULT_LEARNING_RATE
    n_neighbors: int = const.DEFAULT_KNN_N_NEIGHBORS

    def __post_init__(self):
        """
        method: __post_init__
        parameters: self
        return: None
        description: This method is used to validate and preprocess the parameters
        
        """
        
        # convert the parameters to lower case
        #
        self.model_name = self.model_name.lower()
        self.provider_name = self.provider_name.lower()
        self.hardware = self.hardware.lower()
        self.encoder_name = self.encoder_name.lower()
        self.entanglement = self.entanglement.lower()
        self.kernel_name = self.kernel_name.lower() if self.kernel_name else \
            const.DEFAULT_NONE_VALUE
        self.ansatz = self.ansatz.lower() if self.ansatz else \
            const.DEFAULT_NONE_VALUE
        self.measurement_type = self.measurement_type.lower()
        self.noise_model = self.noise_model.lower() if self.noise_model else \
            const.DEFAULT_NONE_VALUE
        self.optim_name = self.optim_name.lower() if self.optim_name else \
            const.DEFAULT_NONE_VALUE
        
        # validate the parameters
        #
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if self.shots < 1:
            raise ValueError("shots must be positive")
        if self.featuremap_reps < 0:
            raise ValueError("reps for featuremap cannot be negative")
        if self.ansatz_reps < 0:
            raise ValueError("reps for ansatz cannot be negative")                
# 
# end of QMLParams
#------------------------------------------------------------------------------

class QMLComponent:
    """
    Base component class for quantum machine learning implementations.
    
    This class serves as a base for all quantum components, providing common
    functionality for parameter management and hardware specification handling.
    It is designed to be inherited by specialized quantum components like
    providers.
    """
    
    def __init__(self, params: QMLParams):
        """
        method: constructor
        
        arguments:
         params: QMLParams object containing configuration parameters
        
        return: 
         none
        
        description:
         initializes the QML component with specified parameters and hardware specs
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating QMLComponent with params: %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(params)))
         
        # set the hardware specification for the component
        # HardwareSpec is a helper class that provides a way to configure the
        # hardware for the component, it take QMLComponent as an argument. In
        # practice, it is used to configure the hardware(cpu, gpu or qpu) for
        # the QuantumProvider. 
        #
        self.hardware_spec = HardwareSpec(self)
        self.hardware = None
        self.set_parameters(params)

    def __repr__(self) -> str:
        """
        method: __repr__
        
        arguments:
         none
        
        return:
         str: string representation of the component
        
        description:
         returns a string representation of the QML component with its parameters
        """
        return f"{self.__class__.__name__}(params={self.params})"

    def set_parameters(self, params: QMLParams):
        """
        method: set_parameters
        
        arguments:
         params: QMLParams object containing configuration parameters
        
        return:
         bool: True if parameters were set successfully
        
        description:
         sets the parameters for the QML component
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: setting parameters: %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(params)))
        
        # set the parameters for the component
        #
        self.params = params
        # exit gracefully
        #
        return True
    
    def get_parameters(self):
        """
        method: get_parameters
        
        arguments:
         none
        
        return:
         QMLParams: current parameter settings
        
        description:
         retrieves the current parameters of the QML component
        """
        # exit gracefully: return the current parameters
        #
        return self.params
        
#
# end of QMLComponent 
#------------------------------------------------------------------------------
   
class QuantumProvider(QMLComponent):
    """
    Base provider class for quantum computing backends.
    
    This class implements the interface for different quantum computing providers
    (e.g., Qiskit, Pennylane). It manages hardware configuration, encoders,
    kernels, and other quantum computing resources.
    """
    
    def __init__(self, params: QMLParams):
        """
        method: constructor
        
        arguments:
         params: QMLParams object containing provider configuration
        
        return: none
        
        description:
         initializes quantum provider with specified parameters
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating QuantumProvider with params: %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(params)))
                  
        self.kernel_func = None
        self.encoder_func = None
        
        # call the parent class (QMLComponent) constructor, passing the parameters
        #
        super().__init__(params=params)
    
    # end of constructor
    #

    def __repr__(self) -> str:
        """
        method: __repr__
        arguments: none
        return: str
        description: string representation of provider instance
        """
        return f"QuantumProvider(hardware={self.hardware}, kernel={self.kernel_func})"

    def set_parameters(self, params: QMLParams):
        """
        method: set_parameters
        
        arguments:
         params: QMLParams configuration parameters
        
        return: bool
        
        description:
         configures provider with specified parameters including hardware
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: setting provider parameters: %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(params)))
        
        # call the parent class set_parameters method
        #        
        super().set_parameters(params)
        
        # configure the hardware for the provider
        #
        self.hardware = self.hardware_spec.configure(
            provider_name=params.provider_name,  
            hardware_type=params.hardware,  
            shots=params.shots,
            noise_model=params.noise_model
        )
        
        # if kernel name is specified, set the kernel function
        # here we are setting the kernel function based on the provider name
        # and the kernel name. 
        # example: if the provider name is qiskit and the kernel name is
        # fidelity then the kernel function will be set to
        # compute_fidelity_kernel because the KERNEL_COMPUTERS dictionary
        # contains the kernel function for the qiskit provider and the fidelity
        # kernel name.
        #
        if self.params.kernel_name:
            self.kernel_func = KERNEL_COMPUTERS[self.params.provider_name]\
                                            [self.params.kernel_name]
        
        # set the encoder function based on the provider name and the encoder
        # name specified in ENCODER_COMPUTERS dictionary.
        #
        self.encoder_func = ENCODER_COMPUTERS[self.params.provider_name]\
                                             [self.params.encoder_name]
        
        # exit gracefully
        #
        return True
    
    def compute_kernel_value(self, X, y=None):
       """
        method: compute_kernel_value
        arguments: X, y
        return: kernel matrix
        description: computes the kernel value for the given data
       """ 
       
       # getattr is used to get the attribute of the object by name
       # here we are getting the kernel function by name (example:   
       # compute_fidelity_kernel). 
       #
       kernel_computer = getattr(self, self.kernel_func)
       
       # exit gracefully: call the kernel function with the data X and y
       # return the kernel matrix
       #
       return kernel_computer(X, y)
   
    def get_encoder(self):
        """
        method: get_encoder
        arguments: None
        return: quantum encoder instance, e.g, ZZEncoder
        description: return an instance of the quantum encoder
       """ 
       
        # getattr is used to get the attribute of the object by name
        # here we are getting the quantum encoder function by name (example:   
        # get_zz_encoder). 
        #
        encoder_computer = getattr(self, self.encoder_func)
        
        # exit gracefully: return the quantum encoder instance
        #
        return encoder_computer()
    
    def get_ansatz(self):
        """
        method: get_ansatz
        arguments: None
        return: quantum ansatz instance, e.g, RealAmplitudes
        description: return an instance of the quantum ansatz
       """ 
       
        # set the ansatz function
        # here we are setting the ansatz function based on the provider name
        # and the ansatz name. 
        # example: if the provider name is qiskit and the ansatz name is
        # real_amplitudes then the ansatz will be set to
        # get_real_amplitudes because the ANSATZ_COMPUTERS dictionary
        # contains the ansatz for the qiskit provider and the real_amplitudes
        # ansatz name.
        #
        ansatz_computer = ANSATZ_COMPUTERS[self.params.provider_name]\
                                          [self.params.ansatz]
        # getattr is used to get the attribute of the object by name
        # here we are getting the quantum encoder function by name (example:   
        # get_real_amplitudes). 
        #
        ansatz_computer = getattr(self, ansatz_computer)
        
        # exit gracefully: return the quantum ansatz instance
        #
        return ansatz_computer()
    
    def get_optimizer(self):
        """
        method: get_optimizer
        arguments: None
        return: optimizer instance, e.g, COBYLA
        description: return an instance of the optimizer
       """ 
       
        # set the optimizer
        # here we are setting the optimizer based on the provider name
        # and the optimizer name. 
        # example: if the provider name is qiskit and the optimizer name is
        # cobyla then the optimizer_computer will be set to
        # get_cobyla_optimizer because of OPTIM_COMPUTERS dictionary
        #
        optimizer_computer = OPTIM_COMPUTERS[self.params.provider_name]\
                                            [self.params.optim_name]
                                            
        # getattr is used to get the attribute of the object by name
        # here we are getting the optimizer by name (example:   
        # get_cobyla_optimizer). 
        #
        optimizer_computer = getattr(self, optimizer_computer)
        
        # exit gracefully: return the optimizer instance
        #
        return optimizer_computer()
    
    #--------------------------------------------------------------------------
    #
    # abstract methods listed here : Quantum Encoders a.k.a FeatureMaps
    #
    #--------------------------------------------------------------------------
    
    # @abstractmethod decorator is used to define an abstract method in the
    # class. The abstract method is a method that is declared in the parent
    # class but does not have an implementation. The child class will implement
    # the abstract method.
    #
    @abstractmethod
    def get_zz_encoder(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_z_encoder(self):
        raise NotImplementedError
    
    #--------------------------------------------------------------------------
    #
    # abstract methods listed here : Quantum Ansatz
    #
    #--------------------------------------------------------------------------
    
    @abstractmethod
    def get_real_amplitudes(self):
        raise NotImplementedError
    
    #--------------------------------------------------------------------------
    #
    # abstract methods listed here : Quantum Kernel functions
    #
    #--------------------------------------------------------------------------

    @abstractmethod
    def compute_fidelity_kernel(self, X: np.array):
        raise NotImplementedError
    
    @abstractmethod
    def computer_qrbf_kernel(self, X: np.array):
        raise NotImplementedError

    #--------------------------------------------------------------------------
    #
    # abstract methods listed here : Optimizers
    #
    #--------------------------------------------------------------------------
    @abstractmethod
    def get_cobyla_optimizer(self):
        raise NotImplementedError
    
    #--------------------------------------------------------------------------
    #
    # abstract methods listed here : Different kind of quantum simluators and
    # hardware resources
    #
    #--------------------------------------------------------------------------

    @abstractmethod
    def get_basic_simulator(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_gpu_simulator(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_qpu(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_sampler(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_estimator(self):
        raise NotImplementedError
    
    #--------------------------------------------------------------------------
    #
    # abstract methods listed here : Quantum Algorithms specific methods
    #
    #--------------------------------------------------------------------------
    @abstractmethod
    def get_qnn_classifier(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_qrbm_feature_encoder(self):
        raise NotImplementedError

# 
# end of QuantumProvider
#------------------------------------------------------------------------------

class HardwareSpec:
    """
    Hardware specification manager for quantum computing resources.
    
    Handles configuration and initialization of quantum computing hardware
    (simulators, QPUs) for different providers.
    """
    
    def __init__(self, provider: QuantumProvider):
        """
        method: constructor
        
        arguments:
         provider: QuantumProvider instance to configure hardware for
        
        return: none
        
        description:
         initializes hardware specification manager
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating HardwareSpec for provider: %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(provider)))
                  
        self.provider = provider

    def __repr__(self) -> str:
        """
        method: __repr__
        arguments: none
        return: str
        description: string representation of hardware spec
        """
        return f"HardwareSpec(provider={self.provider.__class__.__name__})"

    def configure(self, provider_name: str, hardware_type: str, **kwargs):
        """
        method: configure
        
        arguments:
         provider_name: name of quantum provider
         hardware_type: type of hardware to configure
         **kwargs: additional configuration parameters
        
        return: configured hardware instance
        
        description:
         configures specified hardware type for given provider
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: configuring hardware type=%s for provider=%s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, hardware_type, provider_name))
            
        # set the hardware initailization function based on the provider name
        # here we are setting the function based on the provider name
        # and the hardware accelerator name. 
        # example: if the provider name is qiskit and the hardware name is
        # cpu then the hardware_init_func will be set to
        # get_basic_simulator because of HARDWARE_SPECS dictionary
        #
        hardware_init_func = HARDWARE_SPECS[provider_name][hardware_type]
        
        # getattr is used to get the attribute of the object by name
        # here we are getting the hardware by name (example:   
        # get_basic_simulator). 
        #
        hardware_config_func = getattr(self.provider, hardware_init_func)
        
        # exit gracefully: return the configured hardware instance
        #
        return hardware_config_func()

#
# end of HardwareSpec
#------------------------------------------------------------------------------

# ABC is a metaclass for defining abstract base classes. Using ABC ensures all
# quantum models (QSVM, QNN, etc.) have consistent interfaces while allowing
# different implementations.
#
class QuantumModel(ABC):
    """
    Abstract base class for quantum machine learning models.
    
    Defines interface for quantum ML models (e.g., QSVM, QNN) that can be
    trained and used for predictions using quantum circuits.
    """
   
    def __init__(self):
        """
        method: constructor
        arguments: none
        return: none
        description: initializes quantum model instance
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating QuantumModel instance" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
                  
        self.provider = None
        self.params = None

    def __repr__(self) -> str:
        """
        method: __repr__
        arguments: none
        return: str
        description: string representation of model instance
        """
        return f"{self.__class__.__name__}(provider={self.provider.__class__.__name__ if self.provider else None})"

    def set_provider(self, provider: QuantumProvider):
        """
        method: set_provider
        
        arguments:
         provider: QuantumProvider instance to use for computations
        
        return: bool
        
        description:
         sets quantum provider for model computations
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: setting provider: %s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(provider)))
        
        # set the provider for the model, e.g., QiskitProvider.
        #          
        self.provider = provider
        
        # exit gracefully
        #
        return True
    
    #--------------------------------------------------------------------------
    #
    # abstract methods listed here : Quantum Algorithms specific fit/predict  
    #
    #--------------------------------------------------------------------------
    
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError
   
    @abstractmethod
    def predict(self, X):
        raise NotImplementedError               
#
# end of QuantumModel
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
#
# update if a encoder, kernel, ansatz, optimizer or hardware functions are 
# added to the system
#
#------------------------------------------------------------------------------
QISKIT = const.PROVIDER_NAME_QISKIT
DWAVE = const.PROVIDER_NAME_DWAVE
FID = const.KERNEL_NAME_FIDELITY
ZZ = const.ENCODER_NAME_ZZ
Z = const.ENCODER_NAME_Z
QRBM_ENCODER = const.ENCODER_NAME_QRBM_FEATURE_ENCODER
RA = const.ANSATZ_NAME_REAL_AMPLITUDES
COBYLA = const.OPTIM_NAME_COBYLA

KERNEL_COMPUTERS = {QISKIT: {FID: "compute_fidelity_kernel"}}

ENCODER_COMPUTERS = {QISKIT: {ZZ: "get_zz_encoder",
                              Z: "get_z_encoder"},
                    DWAVE: {QRBM_ENCODER: "get_qrbm_feature_encoder"},
                    }


ANSATZ_COMPUTERS = {QISKIT: {RA: "get_real_amplitudes"}}

OPTIM_COMPUTERS = {QISKIT: {COBYLA: "get_cobyla_optimizer"}}

HARDWARE_SPECS = {QISKIT: {const.HARDWARE_NAME_CPU: "get_basic_simulator",    const.HARDWARE_NAME_GPU: "get_gpu_simulator", 
const.HARDWARE_NAME_QPU: "get_qpu"},
                  DWAVE: {const.HARDWARE_NAME_CPU: "get_basic_simulator"}
}

#
# end of file
#------------------------------------------------------------------------------

