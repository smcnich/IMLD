#!/usr/bin/env python

# file: $(NEDC_NFC)/class/python/nedc_qml_tools/nedc_qml_providers_tools.py
#
# revision history: 
#
# 20250203 (SP): initial version
#
# This file contains all the child classes implementing the QuantumProvider
# class.  Each child class is responsible for implementing the methods
# required to interact with a specific quantum provider. There are following
# quantum providers that are supported:
#  - QiskitProvider
#  
#------------------------------------------------------------------------------

# import reqired system modules
#
import os
import random

# import required quantum library modules (Qiskit specific)
#
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.primitives import StatevectorSampler, StatevectorEstimator 
from qiskit.circuit.library import RealAmplitudes
from qiskit_machine_learning.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels.fidelity_quantum_kernel import FidelityQuantumKernel
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.optimizers import COBYLA, L_BFGS_B
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

# import required quantum library modules (Dwave specific)
#
from dimod import BinaryQuadraticModel, Vartype
from neal import SimulatedAnnealingSampler

# import required other libries modules 
#
import numpy as np


# import required NEDC modules
#
import nedc_debug_tools as ndt
from nedc_qml_base_providers_tools import QuantumProvider, QMLParams
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

class QiskitProvider(QuantumProvider):
    """A provider class for Qiskit-based quantum machine learning operations.
    This class implements quantum machine learning functionalities using Qiskit,
    including feature maps, encoders, quantum neural networks, and kernel methods.
    attributes:
        params (QMLParams): Parameters for quantum machine learning operations.
    
    methods (public):
        get_zz_encoder(): returns Qiskit's ZZFeatureMap object.
        get_z_encoder(): returns Qiskit's ZFeatureMap object.
        get_real_amplitudes(): returns Qiskit' RealAmplitudes circuit.
        compute_fidelity_kernel(X, y=None): computes the fidelity kernel matrix
        using FidelityQuantumKernel.
        get_basic_simulator(): returns Qiskit's AerSimulator object.
        get_sampler(): returns Qiskit's Sampler object.
        get_cobyla_optimizer(): returns a COBYLA optimizer.
        get_qnn_classifier(): returns a quantum neural network classifier.
    
    methods (private):
        _parity(x): helper method to compute parity of binary string.
    
    inherits From:
        QuantumProvider: Base class for quantum providers.
    """
    
    def __init__(self, params: QMLParams):
        """
        method: __init__
        arguments: params (QMLParams): Parameters for quantum machine learning
        operations.
        return: none
        description: Initializes the QiskitProvider object with the given
        parameters.
        """
        
        # call the parent class constructor with the given parameters
        #
        super().__init__(params=params)
        self.params = params
    # end of constructor
    #
        
    def __repr__(self) -> str:
        """
        method: __repr__
        arguments: none
        return: str
        description: returns string representation of QiskitProvider instance
        """
        return f"QiskitProvider(params={self.params})"
        
    def get_zz_encoder(self):
        """
        method: get_zz_encoder
        
        arguments:
         none
        
        return: 
         ZZFeatureMap: Qiskit ZZFeatureMap circuit
        
        description:
         creates a ZZ feature map encoder with specified parameters
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating ZZ encoder with n_qubits=%d, reps=%d" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, 
                   self.params.n_qubits, self.params.featuremap_reps))
        
        # create a ZZ feature map encoder with specified parameters
        #        
        zz_encoder = ZZFeatureMap(feature_dimension=self.params.n_qubits,
                                  reps=self.params.featuremap_reps,
                                  entanglement=self.params.entanglement) 
        
        # exit gracefully: return the ZZ encoder object
        #
        return zz_encoder
    
    def get_z_encoder(self):
        """
        method: get_z_encoder
        
        arguments:
         none
        
        return:
         ZFeatureMap: Qiskit ZFeatureMap circuit
        
        description:
         creates a Z feature map encoder with specified parameters
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating Z encoder with n_qubits=%d, reps=%d" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, 
                   self.params.n_qubits, self.params.featuremap_reps))
        
        # create a Z feature map encoder with specified parameters
        #          
        z_encoder = ZFeatureMap(feature_dimension=self.params.n_qubits,
                                reps=self.params.featuremap_reps)
        
        # exit gracefully: return the Z encoder object
        #
        return z_encoder
    
    def get_real_amplitudes(self):
        """
        method: get_real_amplitudes
        
        arguments:
         none
        
        return:
         RealAmplitudes: Qiskit RealAmplitudes circuit
        
        description:
         creates a real amplitudes circuit with specified parameters
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating real amplitudes circuit with n_qubits=%d" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, self.params.n_qubits))
        
        # create a real amplitudes circuit with specified parameters
        #          
        real_amplitudes = RealAmplitudes(num_qubits=self.params.n_qubits,
                                         reps=self.params.ansatz_reps,
                                         entanglement=self.params.entanglement)
        
        # exit gracefully: return the real amplitudes object
        #
        return real_amplitudes
    
    def compute_fidelity_kernel(self, X, y=None):
        """
        method: compute_fidelity_kernel
        
        arguments:
         X: input features
         y: optional target features
        
        return:
         numpy.ndarray: kernel matrix
        
        description:
         computes quantum fidelity kernel matrix for given features
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: computing fidelity kernel for X shape=%s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(X.shape)))
        
        # create a preset pass manager for the backend
        #          
        pm = generate_preset_pass_manager(backend=self.hardware, optimization_level=1)
        
        # get the sampler for the backend, in this case AerSimulator
        #
        sampler = self.get_sampler()
        
        # create a ComputeUncompute object for the fidelity kernel
        #
        fidelity = ComputeUncompute(sampler=sampler, pass_manager=pm)
        
        # create a FidelityQuantumKernel object with the fidelity object
        # and the encoder object provided by the get_encoder method
        #
        fidelity_kernel = FidelityQuantumKernel(fidelity=fidelity, 
                                               feature_map=self.get_encoder())
        # evaluate the kernel matrix for the given features
        #
        K = fidelity_kernel.evaluate(X, y)
        
        # exit gracefully: return the kernel matrix
        #
        return K

    def get_basic_simulator(self):
        """
        method: get_basic_simulator
        
        arguments:
         none
        
        return:
         AerSimulator: Qiskit Aer simulator instance
        
        description:
         creates basic Aer simulator backend
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating basic Aer simulator" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # create a basic Aer simulator backend
        #         
        return AerSimulator()
    
    def get_sampler(self):
        """
        method: get_sampler
        
        arguments:
         none
        
        return:
         Sampler: Qiskit sampler instance
        
        description:
         creates sampler primitive for specified backend
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating sampler for backend=%s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(self.hardware)))
         
        # create a sampler primitive for the specified backend (cpu or gpu or
        # qpu)
        #          
        return Sampler(self.hardware)
    
    def get_cobyla_optimizer(self):
        """
        method: get_cobyla_optimizer
        arguments: none
        return: COBYLA: Qiskit COBYLA optimizer
        description: creates a COBYLA optimizer for Qiskit
        """
        # create a COBYLA optimizer with the maximum number of iterations
        # specified in the parameters
        #
        optimizer = COBYLA(maxiter=self.params.optim_max_steps)
        
        # exit gracefully: return the optimizer object
        #
        return optimizer
    
    def get_qnn_classifier(self):
        """
        method: get_qnn_classifier
        arguments: none
        return: NeuralNetworkClassifier: Qiskit NeuralNetworkClassifier object
        description: creates a quantum neural network classifier for Qiskit
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: creating QNN classifier" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
        # get the feature map, ansatz, and optimizer objects according to the
        # parameters
        #
        feature_map = self.get_encoder()
        ansatz = self.get_ansatz()
        optimizer = self.get_optimizer()
        
        # create a QNN circuit with the feature map and ansatz objects
        #
        qc = QNNCircuit(ansatz=ansatz, feature_map=feature_map,
                        num_qubits=self.params.n_qubits
                        )
        # if the measurement type is sampler, create a SamplerQNN object
        # otherwise create an EstimatorQNN object
        #
        if self.params.measurement_type == const.MEASUREMENT_TYPE_SAMPLER:
            qnn = SamplerQNN(circuit=qc, sampler=StatevectorSampler(),
                            output_shape=self.params.n_classes,
                            interpret=self._parity)
        else:
            qnn = EstimatorQNN(circuit=qc, estimator=StatevectorEstimator())
        
        # create a NeuralNetworkClassifier object with the QNN and optimizer
        # objects
        qnn_classifier = NeuralNetworkClassifier(
        neural_network=qnn, optimizer=optimizer
    )
        
        # exit gracefully: return the QNN classifier object
        #
        return qnn_classifier
    
    def _parity(self, x):
        """
        method: _parity
        arguments: x: input binary string
        return: int: parity of the input binary string
        description: helper method to compute the parity of a binary string
        """
        
        # return the parity of the binary string
        #
        return "{:b}".format(x).count("1") % 2
            
#
# end of QiskitProvider
#------------------------------------------------------------------------------

class DWaveProvider(QuantumProvider):
    """A provider class for D-Wave quantum machine learning operations.
    This class implements quantum machine learning functionalities using D-Wave,
    it's specifically for implementing quantum annealing and related methods.
    attributes:
        params (QMLParams): Parameters for quantum machine learning operations.
    methods (public):   
        get_qrbm_feature_encoder(): returns a quantum restricted boltzmann machine
        classifier.
    
    inherits From:
        QuantumProvider: Base class for quantum providers.
    """
    
    def __init__(self, params: QMLParams):
        """
        method: __init__
        arguments: params (QMLParams): Parameters for quantum machine learning
        operations.
        return: none
        description: Initializes the DWaveProvider object with the given
        parameters.
        """
        
        # call the parent class constructor with the given parameters
        #
        super().__init__(params=params)
        self.params = params
        self.qrbm_n_visible = self.params.n_visible
        self.qrbm_n_hidden = self.params.n_hidden
        
        self.weight = (np.random.rand(self.qrbm_n_visible, self.qrbm_n_hidden) * 2 - 1) * 1
        self.visible_bias = (np.random.rand(self.qrbm_n_visible) * 2 - 1) * 1
        self.hidden_bias = (np.random.rand(self.qrbm_n_hidden) * 2 - 1) * 1
    
    # end of constructor
    #
        
    def __repr__(self) -> str:
        """
        method: __repr__
        arguments: none
        return: str
        description: returns string representation of DWaveProvider instance
        """
        return f"DWaveProvider(params={self.params})"
    
    def get_basic_simulator(self):
        return SimulatedAnnealingSampler() 
    
    def get_qrbm_feature_encoder(self):
        """
        method: get_qrbm_feature_encoder
        arguments: none
        return: DWaveQRBMEncoder: D-Wave quantum restricted boltzmann machine
        feature encoder object
        description: creates a quantum restricted boltzmann machine encoder
        for D-Wave
        """
        
        # create a DWaveQRBMEncoder object with the given parameters
        #
        qrbm_encoder = DWaveQRBMEncoder(params=self.params, sampler=self.get_basic_simulator())
        
        # exit gracefully: return the QRBM encoder object
        #
        return qrbm_encoder
    
 
# end of DWaveProvider
    
class DWaveQRBMEncoder:       
    """A provider class for D-Wave quantum restricted boltzmann machine
    feature encoder.
    This class implements quantum restricted boltzmann machine classifier using
    D-Wave.
    
    The training pipeline was inspired by the following paper:
    Krzysztof, K., Mateusz, S., Marek, S., & Rafał, R. (2021). Applying a
    quantum annealing based restricted Boltzmann machine for mnist handwritten
    digit classification. CMST, 27(3), 99-107.
    (https://cmst.eu/wp-content/uploads/files/10.12921_cmst.2021.0000011_KUROWSKI.pdf)
    
    The code was adapted from the following repository:
    https://github.com/mareksubocz/QRBM
    
    attributes:
        params (QMLParams): Parameters for quantum machine learning operations.
    methods (public):   
        fit(X): fits the quantum restricted boltzmann machine classifier to
        the given features.
        predict(X): predicts the labels for the given features using the
        quantum restricted boltzmann machine classifier.
        compute_layer_activations(X, bias, is_output=False): computes the
        activations of a layer (hidden or visible) using the given parameters.
    methods (private):
        _sample_qubo(v, layer, weights, opposite_layer): samples a binary
        quadratic model (QUBO) using the given parameters.
        _sigmoid(x): computes the sigmoid of the input value.
        
    """
    
    def __init__(self, params: QMLParams, sampler: SimulatedAnnealingSampler):
        """
        method: __init__
        arguments: params (QMLParams): Parameters for quantum machine learning
        return: none
        description: Initializes the DWaveQRBMEncoder object with the given
        parameters.
        """
        
        self.params = params
        self.n_visible = self.params.n_visible
        self.n_hidden = self.params.n_hidden
        self.cs = self.params.chain_strength
        self.lr = self.params.lr
        self.num_reads = self.params.shots
        self.sampler = sampler

        # initialize the weights and biases for the quantum restricted
        # boltzmann machine classifier
        #
        self.weights = (np.random.rand(self.n_visible, self.n_hidden) * 2 - 1) * 1
        self.visible_bias = (np.random.rand(self.n_visible) * 2 - 1) * 1
        self.hidden_bias = (np.random.rand(self.n_hidden) * 2 - 1) * 1
        self.momentum = 0 
                
        
    def fit(self, X):
        """
        method: fit
        arguments: X: input features
        return: none
        description: encode the quantum restricted boltzmann machine 
        to the given features
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: fitting QRBM classifier with X shape=%s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(X.shape)))
                    
        # initialize the momentum variables for the weights and biases
        # these are used to update the weights and biases during training
        # 
        momentum_w = np.zeros((len(self.visible_bias), len(self.hidden_bias)))
        momentum_v = np.zeros(len(self.visible_bias))
        momentum_h = np.zeros(len(self.hidden_bias))
        
        # set the number of epochs for training
        #
        epochs = self.params.epochs
        
        # for each epoch, randomly select a sample from the input features
        # and update the weights and biases using the contrastive divergence
        # algorithm. Here, epochs number is equal to batch size, meaning if we
        # set the epochs as 100, the training will be done for 100 
        # randoms samples
        #
        for epoch in range(epochs):
            if dbgl == ndt.FULL:
                print("%s (line: %s) %s: training epoch=%d" % 
                      (__FILE__, ndt.__LINE__, ndt.__NAME__, epoch))
            
            # select a random sample from the input features
            #
            rnd_idx = random.randint(0, len(X)-1)
            
            # set the visible states to the selected sample
            #
            v = X[rnd_idx]
            old_v = v
            
            # get the hidden states using the visible states and the weights
            # and biases. The hidden states are sampled using the QUBO
            # formulation of the quantum restricted boltzmann machine
            #
            h = self._sample_qubo(v=old_v, layer=self.visible_bias,
                                  weights=self.weights, opposite_layer=self.hidden_bias)
            
            # compute the positive gradient using the visible and hidden
            # states. 
            #
            pos_grad = np.outer(v, h)
            
            # sample the v' using the hidden states and the weights
            # and biases. The v' is the reconstructed visible states
            # using the hidden states and the weights and biases. The
            # v' is sampled using the QUBO formulation of the quantum
            # restricted boltzmann machine
            #
            v_prim = self._sample_qubo(h, self.hidden_bias,
                                       self.weights.T,
                                       self.visible_bias)
            
            # sample the h' using the v' and the weights
            # and biases. The h' is the reconstructed hidden states
            # using the v' and the weights and biases. 
            #
            h_prim = self._sample_qubo(v_prim, self.visible_bias,
                                       self.weights, self.hidden_bias)
            
            # calculate the negative gradient using the v' and
            # h'. The h' is the reconstructed hidden states
            # using the v' and the weights and biases.
            #
            neg_grad = np.outer(v_prim, h_prim)
            
            # Let the update to the weight matrix W be the positive gradient minus the negative gradient, times learning rate
            # this is for momentum (default value 0 doesn't change anything)
            #
            momentum_w = (self.momentum * momentum_w) + (self.lr * (pos_grad - neg_grad))
            self.weights += momentum_w
            
            # Update the biases a and b analogously: a=epsilon (v-v'), #
            # b=epsilon (h-h') 
            # where epsilon is the learning rate
            # v is the visible states
            # h is the hidden states
            # v' is the reconstructed visible states
            # h' is the reconstructed hidden states
            #
            momentum_v = self.momentum * momentum_v + self.lr * np.mean(np.array(v) - np.array(v_prim), axis=0)
            momentum_h = self.momentum * momentum_h + self.lr * np.mean(np.array(h) - np.array(h_prim), axis=0)
            
            # update the biases using the momentum variables
            #
            self.visible_bias += momentum_v
            self.hidden_bias += momentum_h
        
            
    def _sample_qubo(self, v, layer, weights, opposite_layer):
        """
        method: _sample_qubo
        arguments:
         v: visible states
         layer: layer biases
         weights: weights matrix
         opposite_layer: opposite layer biases
        return:
         list: sampled binary states
        description:
         samples a binary quadratic model (QUBO) using the given parameters
        """
        if dbgl == ndt.FULL:
            print("%s (line: %s) %s: sampling QUBO with v=%s, layer=%s, weights=%s, opposite_layer=%s" % 
                  (__FILE__, ndt.__LINE__, ndt.__NAME__, str(v), str(layer), str(weights), str(opposite_layer)))
            
        # set the binary quadratic model (QUBO) type to BINARY
        #
        bqm = BinaryQuadraticModel(const.QUBO_VAR_TYPE)
        
        # construct the QUBO model by adding linear terms
        #
        variables = [f"{str(j)}" for j in range(len(opposite_layer))]
        
        # add linear terms to the QUBO model using the visible states and
        # the weights matrix. The linear terms are the biases of the
        # opposite layer. The biases are the negative of the biases
        # of the opposite layer. 
        #
        for j, opp_bias in enumerate(opposite_layer):
            bqm.add_linear(variables[j], -opp_bias)
        
        # add linear terms to the QUBO model using the visible states and
        # the weights matrix. 
        #
        for i, bias in enumerate(layer):
            if not v[i]:
                continue
                
            for j in range(len(opposite_layer)):
                bqm.add_linear(variables[j], -weights[i][j])
        
        # reading num_reads responses from the sampler
        #
        sampleset = self.sampler.sample(bqm, chain_strength=self.cs, num_reads=self.num_reads, embedding_parameters=dict(timeout=10))

        # take the first sample as it's energy is the lowest
        # and the first sample is the most probable sample
        #
        best_sample = sampleset.first.sample
        solution1_list = [(k, v) for k, v in best_sample.items()]
        solution1_list.sort(key=lambda tup: int(tup[0]))  
        solution1_list_final = [v for (k, v) in solution1_list]

        return solution1_list_final
    
    def compute_layer_activations(self, X, bias, is_output=False):
        """
        method: compute_layer_activations
        
        arguments:
        X: input features or hidden states
        bias: bias terms for the layer
        is_output: whether to binarize the outputs (default: False)
        
        return:
        list: computed layer activations
        
        description:
        computes sigmoid activations for a layer (hidden or visible)
        with optional binarization for output layer
        """
        activations = []
        
        # for each item in the input features, compute the layer
        # activations using the sigmoid function. The sigmoid function
        # is applied to the dot product of the item and the weights
        # and the bias. 
        #
        for item in X:
            layer_activation = self._sigmoid(np.dot(item, self.weights) + bias)
            
            # if the layer is the output layer, binarize the outputs
            # using the threshold of 0.5. The binarization is done
            # by setting the output to 1 if the activation is greater
            # than or equal to 0.5, otherwise set the output to 0.
            # This is done to convert the continuous output of the
            # sigmoid function to a binary output.
            #
            if is_output:
                layer_activation = (layer_activation >= 0.5).astype(int)
            
            # append the layer activation to the list of activations
            #
            activations.append(layer_activation)
        
        # exit gracefully: return the list of activations
        #
        return activations
    
    def _sigmoid(self, x):
        """
        method: _sigmoid
        arguments: x: input value
        return: float: sigmoid of the input value
        description: computes the sigmoid of the input value
        """
        # exit gracefully: return the sigmoid of the input value
        # 
        return 1 / (1 + np.exp(-x))
    
 
# end of DWaveQRBMEncoder     
            
            

            

            
            
