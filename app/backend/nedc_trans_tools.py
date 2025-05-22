#!/usr/bin/env python

# file: $(NEDC_NFC)/class/python/nedc_ml_tools/nedc_trans_tools.py
#
# revision history: 
#
# 20250322 (JP): refactored
# 20250304 (JP): reviewed
# 20250106 (SP): initial version
#
# This file contains an implementation of the transformer architecture. The
# transformer architecture was introduced in the paper "Attention Is All You
# Need" (Vaswani et al., 2017). The transformer architecture is a
# sequence-to-sequence model that uses self-attention mechanisms to capture the
# context of the input sequence. The transformer architecture consists of an
# encoder and a decoder. The encoder processes the input sequence and produces
# a sequence of hidden states. The decoder processes the hidden states and uses
# a self-attention mechanism to generate the output sequence. The transformer
# is known for its parallelism and scalability.
# 
# The reference for this work can be found here:
# 
#  Vaswani et al. "Attention Is All You Need." NeurIPS 2017.
#  https://arxiv.org/abs/1706.03762
#  
#------------------------------------------------------------------------------

# import required system modules
#
import math
import os
from typing import Tuple, Dict

# import various machine learning tools
#
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

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
# default epsillon value for layer normalization class
#
DEFAULT_LN_EPS = 10**-6

# default mask value for the transformer architecture, choose 
# a large negative infinite value, so that when the softmax 
# is applied, the probability of the tokes will be zero
#
DEFAULT_MASK_VALUE = -1e9

# default value for the attention mask, which is False
#
DEFAULT_ATTENTION_MASK = 0

# default skip connection value for the encoder, which is 2
#
DEFAULT_NUM_OF_SKIP_CONNECTIONS_ENCODER = 2

# default skip connection value for the decoder, which is 3
#
DEFAULT_NUM_OF_SKIP_CONNECTIONS_DECODER = 3

# default device
#
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# set the seed for PyTorch's random number generators
#
torch.manual_seed(ndt.RANDSEED)

# set the seed for all GPUs in PyTorch (if available)
#
torch.cuda.manual_seed_all(ndt.RANDSEED)

# set the seed for cuDNN's random number generators
#
torch.backends.cudnn.deterministic = True

# set the benchmark mode for cuDNN to False to disable auto-tuners
# 
torch.backends.cudnn.benchmark = False

# declare a global debug object so we can use it in functions
#
dbgl_g = ndt.Dbgl()
vrbl_g = ndt.Vrbl()

#------------------------------------------------------------------------------
#
# classes listed here
#
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: InputEmbeddings
#------------------------------------------------------------------------------

class InputEmbeddings(nn.Module):
    """
    class: InputEmbeddings

    description:
     The InputEmbeddings layer converts input tokens into continuous vector
     representations.
    
     This class implements the input embeddings layer from the Transformer
     architecture as described in "Attention Is All You Need"
     (Vaswani et al., 2017). It converts input tokens into learned vector
     representations and adds positional encoding to retain sequence
     order information.
    
     The embedding dimension is multiplied by √d_model as per the paper 
     (section 3.4) to scale the embeddings to the appropriate size before
     adding positional encodings.
    """

    #--------------------------------------------------------------------------
    #
    # constructors are listed here
    #
    #--------------------------------------------------------------------------

    def __init__(self, d_model: int, vocab_size: int):
        """
        method: constructor

        arguments:
         d_model (int): The dimensionality of the embedding vectors
         vocab_size (int): Size of the vocabulary

        return:
         none

        description:
         This method initializes the class.
        """

        # set the class name
        #
        InputEmbeddings.__CLASS_NAME__ = self.__class__.__name__

        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating input embedding" % 
                  " with d_model = %d, vocab_size = %d" %
                  (__FILE__, ndt.__LINE__, InputEmbeddings.__CLASS_NAME__, 
                   ndt.__NAME__, d_model, vocab_size))
        
        # set the class variables
        #
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # getting embeddings for the input tokens
        # 
        self.embedding = nn.Embedding(vocab_size, d_model)
    #
    # end of method

    def __repr__(self) -> str:
        """
        method: __repr__

        arguments: none

        return: str

        description: This method returns a string 
                     representation of the class
        """

        # exit gracefully
        #
        return f"{self.__class__.__name__} (d_model = {self.d_model}, vocab_size = {self.vocab_size})"
       
    #
    # end of method

    def forward(self, x: Tensor) -> Tensor:
        """
        method: forward
        
        arguments:
         x (Tensor): Input tokens
        
        return:
         Tensor: Continuous vector representations of input tokens
        
        description:
         This method converts input tokens into continuous vector
         representations and later it will be used in the positional
         encoding to retain sequence order information.
        """

        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   InputEmbeddings.__CLASS_NAME__, ndt.__NAME__,
                   "input embedding - input shape", x.shape))
            
        # multiply the embedding weights by √d_model
        # output shape: (batch_size, seq_len, d_model)
        #
        x = self.embedding(x) * math.sqrt(self.d_model)
        
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   InputEmbeddings.__CLASS_NAME__, ndt.__NAME__,
                   "input embedding - output shape", x.shape))
        
        # exit gracefully
        #
        return x
    #
    # end of method
#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: PositionalEmbedding
#------------------------------------------------------------------------------

class PositionalEmbedding(nn.Module):
    """
    class: PositionalEmbedding

    description:
     This class adds positional encodings to the input embeddings. It
     uses sinusoidal position encoding from "Attention Is All You Need" paper:
      PE(pos,2i) = sin(pos/10000^(2i/d_model))
      PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    
     where:
      - pos is the position in sequence (0 to max_seq_length)
      - i is the dimension index (0 to d_model/2)
      - d_model is the embedding dimension
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """
        method: constructor

        arguments:
         d_model (int): The dimensionality of the embedding vectors
         seq_len (int): The maximum length of the input sequence
         dropout (float): The dropout rate

        return:
         none

        description:
         none
        """
        
        # set the class name
        #
        PositionalEmbedding.__CLASS_NAME__ = self.__class__.__name__

        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s" %
                  " with d_model = %d, seq_len = %d, dropout = %f" %
                  (__FILE__, ndt.__LINE__,
                   PositionalEmbedding.__CLASS_NAME__, ndt.__NAME__,
                   "creating positional embedding", d_model, seq_len, dropout))
        
        # set the class variables
        #
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(p = dropout)
        
        # create a tensor with zeros to store the positional encodings
        # shape: (seq_len, d_model)
        #
        pe = torch.zeros(seq_len, d_model)
        
        # create a position tensor with values from 0 to seq_len
        # output shape: (seq_len, 1)
        #
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 

        # position embedding formula from the paper
        #
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        # only the even position will be applied by sine
        #
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # only the odd position will be applied by cos
        #
        pe[:, 1::2] = torch.cos(position * div_term)
       
        # add a batch dimension
        # output shape: (1, seq_len, d_model)
        #
        pe = pe.unsqueeze(0) 

        # registering it to save in the file, 
        # but not as a learnable parameter
        # 
        self.register_buffer('pe', pe)

    #
    # end of method
        
    def __repr__(self) -> str:
        """
        method: __repr__

        arguments: none

        return: str

        description: This method returns a string representation of the class
        """

        # exit gracefully
        #
        return f"{self.__class__.__name__}(d_model = {self.d_model}," \
               f"seq_len = {self.seq_len}, dropout = {self.dropout.p})"

    #
    # end of method

    def forward(self, x: Tensor) -> Tensor:
        """
        method: forward

        arguments:
         x (Tensor): Input embeddings

        return:
         Tensor: Input embeddings with positional encodings added

        description:
         This method adds positional encodings to the input embeddings.
        """

        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   PositionalEmbedding.__CLASS_NAME__, ndt.__NAME__,
                   "positional embedding input shape", x.shape))
        
        # add the positional encodings to the input embeddings
        # required_grad=False means that the gradients will not be calculated
        #  as it's not a learnable parameter
        # output shape: (batch_size, seq_len, d_model)
        #
        x = x + (pe[:, :x.shape[1], :]).requires_grad(False)
        
        # apply the dropout and return the output
        # output shape: (batch_size, seq_len, d_model)
        #
        x = self.dropout(x)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   PositionalEmbedding.__CLASS_NAME__, ndt.__NAME__,
                   "positional embedding shape", x.shape))
        
        # exit gracefully
        #
        return x

    #
    # end of method
#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: LayerNormalization
#------------------------------------------------------------------------------

class LayerNormalization(nn.Module):
    """
    class: LayerNormalization

    description:
     Layer normalization layer that normalizes the input tensor
     across the last dimension.

     Formula from "Attention Is All You Need":
      LayerNorm(x) = α * (x - μ) / (σ + ε) + β

     where:
      - x: input features
      - μ: mean of the features
      - σ: standard deviation of the features
      - ε: small constant for numerical stability
      - α: learnable scale parameter
      - β: learnable bias parameter
     """

    def __init__(self, eps: float = DEFAULT_LN_EPS) -> None:
        """
        method: constructor

        arguments:
         eps (float): Small constant for numerical stability

        return:
         none

        description:
         none
        """

        # set the class name
        #
        LayerNormalization.__CLASS_NAME__ = self.__class__.__name__

        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating layer normalization" 
                  " with eps=%f" %
                  (__FILE__, ndt.__LINE__,
                   LayerNormalization.__CLASS_NAME__, ndt.__NAME__, eps))
        
        # set the learning rate
        #
        self.eps = eps
        
        # set the multiplicative param
        #
        self.alpha = nn.Parameter(torch.ones(1)) 

        # set the additive param
        # 
        self.beta = nn.Parameter(torch.ones(1)) 
        
    #
    # end of method
        
    def __repr__(self) -> str:
        """
        method: __repr__

        arguments: none

        return: str

        description:
         This method returns a string representation of the class.
        """

        # exit gracefully
        #
        return f"{self.__class__.__name__}(eps = {self.eps})"

    #
    # end of method

    def forward(self, x: Tensor) -> Tensor:
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   LayerNormalization.__CLASS_NAME__, ndt.__NAME__,
                   "layer normalization input shape", x.shape))
        
        # calculate the mean and standard deviation of the input tensor:
        #  dim = -1 means the last dimension meaning everything after
        #  the batch dimension
        #
        #  keepdims = True means the output tensor will have the same
        #  number of dimensions
        # 
        mean = x.mean(dim = -1, keepdims = True) 
        std = x.std(dim = -1, keepdims = True)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   LayerNormalization.__CLASS_NAME__, ndt.__NAME__,
                   "layer normalization mean shape", mean.shape))
            print("%s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   LayerNormalization.__CLASS_NAME__, ndt.__NAME__,
                   "layer normalization std shape", std.shape))
        
        # apply the layer normalization formula:
        #  output shape: (batch_size, seq_len, d_model)
        #
        x = self.alpha * (x - mean) / (std + self.eps) + self.beta
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   LayerNormalization.__CLASS_NAME__, ndt.__NAME__,
                   "layer normalization output shape", x.shape))
        
        # exit gracefully
        #
        return x
    #
    # end of method
#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: FeedForwardBlock
#------------------------------------------------------------------------------

class FeedForwardBlock(nn.Module):
    """
    class: FeedForwardBlock

    description:
     Feedforward block that consists of two linear transformations with a ReLU
     activation in between.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        method: constructor

        arguments:
         d_model (int): The dimensionality of the embedding vectors
         d_ff (int): The dimensionality of the feedforward layer
         dropout (float): The dropout rate

        return:
          none

        description:
         In the "Attention Is All You Need" paper, the feedforward block
         consists of two linear transformations, d_model -> d_ff and d_ff ->
         d_model, where d_model = 512 and d_ff = 2048. The activation
         function was ReLU.
        """

        # set the class name
        #
        FeedForwardBlock.__CLASS_NAME__ = self.__class__.__name__

        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating feedforward block" 
                  "with d_model = %d, d_ff = %d, dropout = %f" %
                 (__FILE__, ndt.__LINE__,
                  FeedForwardBlock.__CLASS_NAME__, ndt.__NAME__,
                  d_model, d_ff, dropout))
        
        # set the class variables
        #
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        
        # create the first linear layer's weights and biases
        # according to the paper, it's equivalent to w1 and b1
        #
        self.linear_1 = nn.Linear(self.d_model, self.d_ff) 
        
        # set the dropout value
        #
        self.dropout = nn.Dropout(p=self.dropout)
        
        # create the second linear layer's weights and biases
        # according to the paper, it's equivalent to w2 and b2
        #
        self.linear_2 = nn.Linear(self.d_ff, self.d_model) 
        
    #
    # end of method

    def __repr__(self) -> str:
        """
        method: __repr__

        arguments: none

        return: str

        description:
         This method returns a string representation of the class.
        """

        # exit gracefully
        #
        return f"{self.__class__.__name__}(d_model = {self.d_model}," \
               f"d_ff = {self.d_ff}, dropout = {self.dropout.p})"

    #
    # end of method

    def forward(self, x: Tensor) -> Tensor:
        """
        method: forward

        arguments:
         x (Tensor): Input tensor

        return:
         Tensor: Output tensor

        description:
         This method applies two linear transformations with a ReLU
         activation in between. The input shape is
         [batch_size, seq_len, d_model].
        """

        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   FeedForwardBlock.__CLASS_NAME__, ndt.__NAME__,
                   "feedforward block input shape", x.shape))
        
        # apply the first linear transformation, the output shape will be
        # (batch_size, seq_len, d_ff)
        #
        x = self.linear_1(x)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   FeedForwardBlock.__CLASS_NAME__, ndt.__NAME__,
                   "feedforward block linear 1 shape", x.shape))
            
        # apply the ReLU activation function
        #
        x = torch.relu(x)
        
        # apply the dropout
        #
        x = self.dropout(x)
        
        # apply the second linear transformation, the output shape will be
        # (batch_size, seq_len, d_model)
        #
        x = self.linear_2(x)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   FeedForwardBlock.__CLASS_NAME__, ndt.__NAME__,
                   "feedforward block linear 2 shape", x.shape))
        
        # exit gracefully
        #
        return x

    #
    # end of method
#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: MultiHeadAttentionBlock
#------------------------------------------------------------------------------

class MultiHeadAttentionBlock(nn.Module):
    """
    class: MultiHeadAttentionBlock

    description:
     Multi-head attention block that consists of multiple scaled dot-product
     attention heads.
    
     Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
     MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
     where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    
     where,
      Q: Query matrix
      K: Key matrix
      V: Value matrix
      d_k: Scaling factor (dimension of key vectors)
      W_i^Q, W_i^K, W_i^V: Learned projection matrices
      W^O: Output projection matrix
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        """
        method: constructor

        arguments:
         d_model (int): The dimensionality of the embedding vectors
         num_heads (int): The number of attention heads
         dropout (float): The dropout rate

        return:
         none

        description:
         none
        """

        # set the class name
        #
        MultiHeadAttentionBlock.__CLASS_NAME__ = self.__class__.__name__

        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s " \
                  " with d_model = %d, num_heads = %d, dropout = %f" %
                  (__FILE__, ndt.__LINE__,
                   MultiHeadAttentionBlock.__CLASS_NAME__, ndt.__NAME__,
                   "creating multi-head attention block",
                   d_model, num_heads, dropout))
        
        # set the class variables
        #
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(p=dropout)
        
        # check if d_model is divisible by num_heads
        #
        assert d_model % num_heads == 0, "d_model is not divisible by h"
        
        # set the d_k value which is embedding size of each head
        #
        self.d_k = d_model // num_heads
        
        # create the query, key, and value linear transformations
        # weights and biases
        #
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # create the output linear transformation weights and biases
        #
        self.w_o = nn.Linear(d_model, d_model)
        
    #
    # end of method
        
    def __repr__(self) -> str:
        """
        method: __repr__

        arguments: none

        return: str

        description:
         This method returns a string representation of the class.
        """

        # exit gracefully
        #
        return f"{self.__class__.__name__}(d_model = {self.d_model},"\
               f"num_heads = {self.num_heads}, d_k = {self.d_k})"

    #
    # end of method

    @staticmethod
    def attention(query: Tensor, key: Tensor, value: Tensor,
                  mask: Tensor = None,
                  dropout: nn.Dropout = None) -> Tuple[Tensor, float]:
        """
        method: attention

        arguments:
         query (Tensor): Query matrix
         key (Tensor): Key matrix
         value (Tensor): Value matrix
         mask (Tensor): Mask tensor
         dropout (nn.Dropout): Dropout layer

        return:
         Tuple[Tensor, float]: Attention output and attention scores

        description:
         This method computes the scaled dot-product attention mechanism.
        """
        
        # getting the last dimension of the query matrix,
        # which is embedding dimension of each head     
        # for example, if the embedding size is 512 
        # and the number of heads is 8 then the d_k will 
        # be 512 / 8 = 64
        #
        d_k = query.shape[-1]
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: query shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   MultiHeadAttentionBlock.__CLASS_NAME__, ndt.__NAME__,
                   query.shape))
            print("%s (line: %s) %s::%s: key shape = %s" %
                  (__FILE__, ndt.__LINE__,
                   MultiHeadAttentionBlock.__CLASS_NAME__, ndt.__NAME__,
                   key.shape))
            print("%s (line: %s) %s::%s: value shape = %s" %
                  (__FILE__, ndt.__LINE__,
                   MultiHeadAttentionBlock.__CLASS_NAME__, ndt.__NAME__,
                   value.shape))
            print("%s (line: %s) %s::%s: d_k = %d" %
                  (__FILE__, ndt.__LINE__,
                   MultiHeadAttentionBlock.__CLASS_NAME__, ndt.__NAME__,
                   d_k))

        # calculating the attention scores
        # (batch, num_heads, seq_len, d_k) -> transpose*() -> (batch, h, d_k,
        #  seq_len)
        #
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: attention scores shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   MultiHeadAttentionBlock.__CLASS_NAME__, ndt.__NAME__,
                   attention_scores.shape))
        
        # applying the mask if available, by default it's None
        # during training the Encoder,  we need the mask for the padding 
        # tokens, so that the model will not pay attention to them
        #
        # during training the Decoder, we need the mask for the future tokens 
        # so that the model will not pay attention to them
        #
        if mask:
            
            # display informational message
            #
            if dbgl_g > ndt.BRIEF:
                print("%s (line: %s) %s::%s: applying mask" %
                      (__FILE__, ndt.__LINE__,
                       MultiHeadAttentionBlock.__CLASS_NAME__, ndt.__NAME__))
            
            # set the attention scores
            #
            attention_scores.masked_fill_(mask == DEFAULT_ATTENTION_MASK,
                                          DEFAULT_MASK_VALUE)
        
        # dim = -1 means the last dimension meaning everything after the batch
        # output shape: (batch, num_heads, seq_len, d_k)
        # 
        attention_scores = attention_scores.softmax(dim=-1) 
        
        # applying the dropout
        #
        if dropout:
            attention_scores = dropout(attention_scores)
        
        # exit gracefully:
        #  output shape: (batch, num_heads, seq_len, d_k)
        #
        return (attention_scores @ value), attention_scores

    #
    # end of method

    def forward(self, query: Tensor, key: Tensor, value: Tensor,
                mask: Tensor = None) -> Tensor:
        """
        method: forward

        arguments:
         query (Tensor): Query matrix
         key (Tensor): Key matrix
         value (Tensor): Value matrix
         mask (Tensor): Mask tensor

        return:
         Tensor: Multi-head attention output

        description:
         This method computes the multi-head attention mechanism.
        """
        
        # output shape: (batch, seq_len, d_model)
        #
        query = self.w_q(query) 
        key = self.w_q(key) 
        value = self.w_q(value) 
        
        # splitting the input into smaller matrix
        # view() is used to reshape the tensor
        # transpose() is used to change the order of the dimensions and
        # it is important because each head needs to see the smaller parts
        # (d_k) of the full sentences (seq_len)
        # (batch, seq_len, d_model) -> view() -> (batch, seq_len, num_heads,
        # d_k) -> transpose() -> (batch, num_heads, seq_len, d_k)
        #
        query = query.view(query.shape[0], query.shape[1],
                           self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],
                       self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1],
                           self.num_heads, self.d_k).transpose(1, 2)

        # applying the attention mechanism
        # output shape: (batch, num_heads, seq_len, d_k)
        #
        x, attention_scores = self.attention(query=query, key=key,
                                             value=value,
                                             dropout=self.dropout)

        # combining each head into the complete matrix
        # (batch, num_heads, d_k, seq_len) -> transpose() -> (batch, d_k,
        # num_heads, seq_len) -> view ()-> (batch, seq_len, d_model)
        # using contiguous() for storing the elements sequentially, otherwise
        # getting errors during view() operation
        # 
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1,
                                               self.d_k * self.num_heads)

        # applying the output linear transformation
        # output shape: (batch, seq_len, d_model)
        #
        x = self.w_o(x)
        
        # display informational message
        # 
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s (%s)" %
                (__FILE__, ndt.__LINE__,
                 MultiHeadAttentionBlock.__CLASS_NAME__, ndt.__NAME__,
                 "multi-head attention block shape", x.shape))
        
        # exit gracefully   
        #
        return x 

    #
    # end of method
#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: ResidualConnection
#------------------------------------------------------------------------------

class ResidualConnection(nn.Module):
    """
    class: ResidualConnection

    description:
     This class implements residual (skip) connection with layer
     normalization and dropout.
    
     Formula from "Attention Is All You Need":
      LayerNorm(x + Dropout(Sublayer(x)))
    
     where:
      - x: Input to the sublayer
      - Sublayer: Any transformer sublayer (attention or feed-forward)
      - Dropout: Regularization
      - LayerNorm: Layer normalization class
    
     arguments:
      dropout (float): Dropout probability
    """

    def __init__(self, dropout: float) -> None:
        """
        method: constructor

        arguments:
         dropout (float): dropout probability threshold

        return:
         none

        description:
         none
        """

        # set the class name
        #
        MultiHeadAttentionBlock.__CLASS_NAME__ = self.__class__.__name__

        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating residual connection" 
                  " with dropout=%f" %
                  (__FILE__, ndt.__LINE__,
                   ResidualConnection.__CLASS_NAME__, ndt.__NAME__,
                   dropout))        

        # set the dropout value
        #
        self.dropout = nn.Dropout(dropout)
        
        # layer normalization
        #
        self.norm = LayerNormalization()
        
    #
    # end of method

    def __repr__(self) -> str:
        """
        method: __repr__

        arguments: none

        return: str

        description:
         This method returns a string representation of the class.
        """

        # exit gracefully
        #
        return f"{self.__class__.__name__}(dropout = {self.dropout.p})"

    #
    # end of method

    def forward(self, x: Tensor, sublayer: nn.Module) -> Tensor:
        """
        method: forward

        arguments:
         x (Tensor): Input tensor
         sublayer (nn.Module): Sublayer (layer normalization)

        return:
         Tensor: Output tensor

        description:
         This method implements the residual connection with layer
         normalization and dropout.
        """  

        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s (%s)" %
                  (__FILE__, ndt.__LINE__,
                   ResidualConnection.__CLASS_NAME__, ndt.__NAME__,
                   "residual connection input shape", x.shape))
            
        # output shape: (batch, seq_len, d_model)
        # 
        x_prime = self.norm(x)
        
        # call the sublayer (attention or feedforward)'s forward method:
        #  output shape: (batch, seq_len, d_model)
        #
        x_prime = sublayer(x)
        
        # apply the dropout:
        #  output shape: (batch, seq_len, d_model)
        #
        x_prime = self.dropout(x)
        
        # adding the skip connection:
        #  output shape: (batch, seq_len, d_model)
        #
        x = x + x_prime
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: residual connection shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   ResidualConnection.__CLASS_NAME__, ndt.__NAME__,
                   x.shape))
        
        # exit gracefully
        #
        return x

    #
    # end of method
#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: EncoderBlock
#------------------------------------------------------------------------------

class EncoderBlock(nn.Module):
    """ 
    class: EncoderBlock

    description:
     Encoder block that consists of a multi-head attention block and a
     feedforward block.
    """
    
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        """
        method: constructor

        arguments:
         self_attention_block: MultiHeadAttentionBlock
         feed_forward_block: FeedForwardBlock
         dropout (float): dropout rate

        return:
         none

        description:
         none
        """

        # set the class name
        #
        EncoderBlock.__CLASS_NAME__ = self.__class__.__name__

        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating encoder block" 
                  " with self_attention_block = %s," 
                  " feed_forward_block = %s, dropout = %f" % 
                  (__FILE__, ndt.__LINE__,
                   EncoderBlock.__CLASS_NAME__, ndt.__NAME__, 
                   self_attention_block, feed_forward_block, dropout))
        
        # set the class variables
        #
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block        
        self.dropout = dropout
        
        # create two skip connections for the encoder block
        # 
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout = self.dropout)
             for _ in range(DEFAULT_NUM_OF_SKIP_CONNECTIONS_ENCODER)])
        
        # display informational message
        # 
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: encoder block" 
                  " residual connections: %s" %
                  (__FILE__, ndt.__LINE__,
                   EncoderBlock.__CLASS_NAME__, ndt.__NAME__,
                   self.residual_connections))
        
    #
    # end of method

    def __repr__(self) -> str:
        """
        method: __repr__

        arguments: none

        return: str

        description:
         This method returns a string representation of the class.
        """

        # exit gracefully
        #
        return f"{self.__class__.__name__}(self_attention_block = {self.self_attention_block}, feed_forward_block = {self.feed_forward_block}))"
    #
    # end of method

    def forward(self, x: Tensor, src_mask: Tensor) -> Tensor:
        """
        method: forward

        arguments:
         x (Tensor): Input tensor
         src_mask (Tensor): Mask tensor

        return:
         Tensor: Output tensor

        description:
         This method computes the encoder block.
        """
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: encoder block input shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   EncoderBlock.__CLASS_NAME__, ndt.__NAME__, x.shape))
            
            print("%s (line: %s) %s::%s: encoder block skip connections: %s" %
                  (__FILE__, ndt.__LINE__,
                   EncoderBlock.__CLASS_NAME__, ndt.__NAME__,  
                   len(self.residual_connections)))
               
        # apply the self-attention mechanism on the first skip connection:
        #  a lambda keyword is used to define an anonymous function.
        #  It is used to pass the self-attention function as a parameter:
        #
        # the full process:
        #  1. the residual connection block saves the input x
        #  2. layer normalization is applied on x creating x_prime
        #  3. x_prime passes through the self-attention sublayer
        #  4. the self-attention block:
        #     - projects x_prime into Q, K, V matrices
        #     - computes attention scores (Q × K^T)
        #     - applies the mask to prevent invalid attention
        #     - applies softmax to get attention weights
        #     - multiplies with V to get the output
        #  5. the residual connection adds the original input x 
        #     to the attention output
        #
        # output shape: (batch, seq_len, d_model)
        # 
        x = self.residual_connections[0](
            x = x, sublayer = lambda x: self.self_attention_block(
                query = x, value = x, key = x, mask = src_mask))
        
        # display informational message
        # 
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: after first skip connection," 
                  " encoder block self-attention shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   EncoderBlock.__CLASS_NAME__, ndt.__NAME__, x.shape))
        
        # apply the feedforward block on the second skip connection:
        #
        # the full process:
        #  1. the residual connection block saves the input x
        #  2. layer normalization is applied on x making x_prime
        #  3. passes x_prime through the feed-forward network:
        #     - projects x_prime to a larger dimension via first 
        #       linear layer    (d_model -> d_ff)
        #     - applies dropout as regularization
        #     - applies ReLU activation function 
        #     - projects back to original dimension via second 
        #       linear layer (d_ff -> d_model)
        #  4. the residual connection adds the original input 
        #     x to the FFN output
        #
        # output shape: (batch, seq_len, d_model)
        #
        x = self.residual_connections[1](x = x,
                                         sublayer = self.feed_forward_block)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: after second skip connection," 
                  " encoder block feedforward shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   EncoderBlock.__CLASS_NAME__, ndt.__NAME__, x.shape))
            
        # exit gracefully
        #
        return x
    #
    # end of method

#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: Encoder
#------------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    class: Encoder

    description:
     This class implements an encoder that consists of multiple encoder blocks.
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        """
        method: constructor

        arguments:
         layers (nn.ModuleList): List of encoder blocks

        return:
         none

        description:
         none
        """
        
        # set the class name
        #
        Encoder.__CLASS_NAME__ = self.__class__.__name__

        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        # 
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating encoder with layers=%s" %
                  (__FILE__, ndt.__LINE__,
                   Encoder.__CLASS_NAME__, ndt.__NAME__, layers))
        
        # list of encoder blocks
        #
        self.layers = layers

        # initializing layer normalization as the last layer
        #
        self.norm = LayerNormalization()
        
    #
    # end of method
        
    def __repr__(self) -> str:
        """

        method: __repr__

        arguments: none

        return: str

        description:
         This method returns a string representation of the class.
        """

        # exit gracefully
        #
        return f"{self.__class__.__name__}(layers = {self.layers})"

    #
    # end of method

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        method: forward

        arguments:
         x (Tensor): input tensor
         mask (Tensor): mask tensor

        return:
         Tensor: Output tensor

        description:  
         This method computes the encoder.
        """
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: encoder input shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   Encoder.__CLASS_NAME__, ndt.__NAME__, x.shape))
        
        # for each encoder block, apply the forward method
        #
        for layer in self.layers:

            # output shape: (batch, seq_len, d_model)
            #
            x = layer(x, mask)
        
        # apply the layer normalization
        # output shape: (batch, seq_len, d_model)
        #
        x = self.norm(x)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: encoder output shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   Encoder.__CLASS_NAME__, ndt.__NAME__, x.shape))
        
        # exit gracefully
        #  
        return x
    #
    # end of method

#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: DecoderBlock
#------------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    class: DecoderBlock

    description:
     This class implements a decoder block that consists of a multi-head
     self-attention block, a multi-head cross-attention block, and a
     feedforward block. 
    """

    def __init__(self, self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        """
        method: constructor

        arguments:
         self_attention_block: MultiHeadAttentionBlock
         cross_attention_block: MultiHeadAttentionBlock
         feed_forward_block: FeedforwardBlock
         dropout (float): dropout rate

        return:
         none

        description:
         none
        """

        # set the class name
        #
        DecoderBlock.__CLASS_NAME__ = self.__class__.__name__
        
        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating decoder block" 
                  " with self_attention_block = %s," 
                  " cross_attention_block = %s," 
                  " feed_forward_block = %s, dropout = %f" %
                  (__FILE__, ndt.__LINE__,
                   DecoderBlock.__CLASS_NAME__, ndt.__NAME__,
                   self_attention_block, cross_attention_block,
                   feed_forward_block, dropout))
        
        # set the class variables
        #
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        # creating three skip connections for the decoder block
        # 
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(dropout=dropout)
            for _ in range(DEFAULT_NUM_OF_SKIP_CONNECTIONS_ENCODER)])
        
    #
    # end of method

    def __repr__(self) -> str:
        """

        method: __repr__

        arguments: none

        return: str

        description:
         This method returns a string representation of the class.
        """

        # exit gracefully
        #
        return f'''{self.__class__.__name__}("self_attention_block = {self.self_attention_block}," "cross_attention_block = {self.cross_attention_block},", "feed_forward_block = {self.feed_forward_block}")'''

    #
    # end of method

    def forward(self, x: Tensor,
                encoder_output: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor) -> Tensor:
        """
        method: forward

        arguments:
         x (Tensor): Input tensor
         encoder_output (Tensor): Encoder output tensor
         src_mask (Tensor): Source mask tensor
         tgt_mask (Tensor): Target mask tensor

        return:
         Tensor: Output tensor

        description:
         This method computes the decoder block.
        """
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: decoder block input shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   DecoderBlock.__CLASS_NAME__, ndt.__NAME__, x.shape))
            
            print("%s (line: %s) %s::%s: decoder block skip connections: %s" %
                  (__FILE__, ndt.__LINE__,
                   DecoderBlock.__CLASS_NAME__, ndt.__NAME__,
                   len(self.residual_connections)))
        
        # apply the self-attention mechanism on the first skip connection:
        #  a lambda keyword is used to define an anonymous function.
        #  It is used to pass the self-attention function as a parameter.
        #
        # the full process :
        #  1. the residual connection block saves the input x
        #  2. layer normalization is applied on x making x_prime
        #  3. x_prime passes through the self-attention sublayer
        #  4. the self-attention block:
        #     - projects x_prime into Q, K, V matrices
        #     - computes attention scores (Q × K^T)
        #     - applies the mask to prevent invalid attention
        #     - applies softmax to get attention weights
        #     - multiplies with V to get the output
        #  5. the residual connection adds the original input x 
        #     to the attention output
        # output shape: (batch, seq_len, d_model)
        # 
        x = self.residual_connections[0](
            x = x, sublayer = lambda x: self.self_attention_block(
                query = x, key = x, value = x, mask = tgt_mask))
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: after first skip connection," 
                  " decoder block self-attention shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   DecoderBlock.__CLASS_NAME__, ndt.__NAME__, x.shape))
        
        # applying the cross-attention mechanism on the second skip connection:
        #  the process is exactly the same as the self-attention mechanism,
        #  but the query comes from the previous decoder block and the key
        #  and value comes from the encoder's last layer output.
        #  It is useful because it allows the decoder to focus on different
        #  parts of the input sequence based on the different output positions
        #
        # output shape: (batch, seq_len, d_model)
        # 
        x = self.residual_connections[1](
            x = x, sublayer=lambda x: self.cross_attention_block(
                query = x, key = encoder_output,
                value = encoder_output, mask = src_mask))
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: after second skip connection," 
                  " decoder block cross-attention shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   DecoderBlock.__CLASS_NAME__, ndt.__NAME__, x.shape))
         
        # applying the feedforward block on the third skip connection:
        #  1. the residual connection block saves the input x
        #  2. layer normalization is applied on x making x_prime
        #  3. passes x_prime through the feed-forward network:
        #     - projects x_prime to a larger dimension via first 
        #       linear layer (d_model -> d_ff)
        #     - applies dropout as regularization
        #     - applies ReLU activation function 
        #     - projects back to original dimension via second 
        #       linear layer (d_ff -> d_model)
        #  4. the residual connection adds the original input 
        #     x to the FFN output
        #
        # output shape: (batch, seq_len, d_model)
        # 
        x = self.residual_connections[2](x = x,
                                         sublayer=self.feed_forward_block)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: after third skip connection," 
                  " decoder block feedforward shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   DecoderBlock.__CLASS_NAME__, ndt.__NAME__, x.shape))
        
        # exit gracefully
        #
        return x
    #
    # end of method
    
#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: Decoder
#------------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    class: Decoder

    description:
     Decoder that consists of multiple decoder blocks.
    """
    
    def __init__(self, layers: nn.ModuleList) -> None:
        """
        method: constructor

        arguments:
         layers (nn.ModuleList): List of decoder blocks

        return:
         none

        description:
         none
        """

        # set the class name
        #
        Decoder.__CLASS_NAME__ = self.__class__.__name__
        
        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating decoder with layers=%s" %
                  (__FILE__, ndt.__LINE__,
                   Decoder.__CLASS_NAME__, ndt.__NAME__, layers))
        
        # set the class variables
        #
        self.layers = layers

        # initializing layer normalization as the last layer
        #
        self.norm = LayerNormalization()
        
    #
    # end of method
    
    def __repr__(self) -> str:
        """
        method: __repr__

        arguments: none

        return: str

        description:
         This method returns a string representation of the class.
        """

        # exit gracefully
        #
        return f"{self.__class__.__name__}(layers = {self.layers})"

    #
    # end of method

    def forward(self, x: Tensor, encoder_output: Tensor,
                src_mask: Tensor, tgt_mask: Tensor) -> Tensor:
        """

        method: forward

        arguments:
         x (Tensor): Input tensor
         encoder_output (Tensor): Encoder output tensor
         src_mask (Tensor): Source mask tensor
         tgt_mask (Tensor): Target mask tensor

        return:
         Tensor: Output tensor

        description:
         This method computes the decoder.
        """
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: decoder input shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   Decoder.__CLASS_NAME__, ndt.__NAME__, x.shape))
        
        # for each decoder block, apply the forward method
        #
        for layer in self.layers:

            # the src_mask is used for the padding tokens
            # the tgt_mask is used for the future tokens
            # output shape: (batch, seq_len, d_model)
            #
            x = layer(x = x, encoder_output=encoder_output,
                      src_mask = src_mask, tgt_mask = tgt_mask)
            
        # apply the layer normalization:
        #  output shape: (batch, seq_len, d_model)
        #
        x = self.norm(x)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: decoder output shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   Decoder.__CLASS_NAME__, ndt.__NAME__, x.shape))
        
        # exit gracefully
        # 
        return x

    #
    # end of method
#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: ProjectionLayer
#------------------------------------------------------------------------------

class ProjectionLayer(nn.Module):
    """
    class: ProjectionLayer

    description:
     This class implements a projection layer that projects the
     decoder output to the vocabulary size.
    """
    
    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        method: constructor

        arguments:
         d_model (int): The dimensionality of the embedding vectors
         vocab_size (int): Size of the vocabulary

        return:
         none

        description:
         none
        """

        # set the class name
        #
        ProjectionLayer.__CLASS_NAME__ = self.__class__.__name__

        # create the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating projection layer with"  
                  " d_model="%d, "vocab_size="%d %
                  (__FILE__, ndt.__LINE__,
                   ProjectionLayer.__CLASS_NAME__, ndt.__NAME__, 
                   d_model, vocab_size))
        
        # create the linear layer's weights and biases:
        #  output shape: (d_model, vocab_size)
        #
        self.proj = nn.Linear(d_model, vocab_size)
        
    #
    # end of method

    def forward(self, x: Tensor) -> Tensor:
        """
        method: forward

        arguments:
         x (Tensor): Input tensor

        return:
         Tensor: Output tensor

        description:
         This method projects the decoder output to the vocabulary size.
        """

        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: projection layer input shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   ProjectionLayer.__CLASS_NAME__, ndt.__NAME__, x.shape))
        
        # applying the linear transformation and the log softmax:
        #  (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        #
        x = torch.log_softmax(self.proj(x), dim = -1)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: projection layer output shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   ProjectionLayer.__CLASS_NAME__, ndt.__NAME__, x.shape))
        
        # exit gracefully
        #
        return x
    #
    # end of method
    
#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: BaseTransformerClassifier
#------------------------------------------------------------------------------

class BaseTransformerClassifier(nn.Module):
    """
    class: BaseTransformerClassifier

    description:
     This is a base class for transformer-based classifiers.
    """

    def get_cross_entropy_loss_function(self):
        """
        method: get_loss_function

        arguments:
         none

        return: 
         nn.CrossEntropyLoss: cross-entropy loss function

        description:    
         This method returns the cross-entropy loss function.
        """

        # set the class name
        #
        BaseTransformerClassifier.__CLASS_NAME__ = self.__class__.__name__
        
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating CrossEntropyLoss" %
                  (__FILE__, ndt.__LINE__,
                   BaseTransformerClassifier.__CLASS_NAME__, ndt.__NAME__))
        
        # exit gracefully:
        #  return the cross-entropy loss function
        #
        return nn.CrossEntropyLoss()
    #
    # end of method
    
    def get_adam_optimizer(self, lr: float):
        """
        method: get_adam_optimizer

        arguments:
         lr (float): Learning rate

        return:
         torch.optim.Adam: Adam optimizer

        description:
         This method returns the Adam optimizer.
        """
        
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: %s = %f" %
                  (__FILE__, ndt.__LINE__,
                   BaseTransformerClassifier.__CLASS_NAME__, ndt.__NAME__,
                   "creating Adam optimizer with lr", lr))
        
        # exit gracefully: 
        #  return the Adam optimizer
        #
        return torch.optim.Adam(self.parameters(), lr = lr)
    #
    # end of method
    
    def to_tensor(self, value: np.ndarray) -> Tensor:
        """
        method: to_tensor

        arguments:
         value (np.ndarray): Numpy array

        return: 
         Tensor: Tensor

        description:    
         This method converts a numpy array to a tensor.
        """

        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: converting numpy array to tensor" %
                  (__FILE__, ndt.__LINE__,
                   BaseTransformerClassifier.__CLASS_NAME__, ndt.__NAME__))
        
        # convert the numpy array to a tensor
        #
        value = torch.LongTensor(value)
        
        # move the tensor to the default device
        #
        value = self.to_device(value)
        
        # exit gracefully
        #
        return value    

    #
    # end of method
    
    def to_device(self, tensor: Tensor) -> Tensor:
        """
        method: to_device

        arguments:
            tensor (Tensor): Tensor

        return:
            Tensor: Tensor

        description:
            This method moves the tensor to the default device.
        """

        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: moving tensor to device: %s" %
                  (__FILE__, ndt.__LINE__,
                   BaseTransformerClassifier.__CLASS_NAME__, ndt.__NAME__,
                   DEFAULT_DEVICE))
        
        # exit gracefully:
        #  return the tensor moved to the default device
        #
        return tensor.to(DEFAULT_DEVICE)
    
    #
    # end of method
#
# end of class

#------------------------------------------------------------------------------
# NEDC Trans Tools Class: NEDCTransformer
#------------------------------------------------------------------------------

class NEDCTransformer(BaseTransformerClassifier):
    """
    class: NEDCTransformer

    description:
     This class is a transformer-based (encoder only) classifier that was
     added to facilitate classification experiments like IMLD.
    """

    def __init__(self, input_dim: int, num_classes: int,
                 d_model: int, nhead: int, num_layers: int,
                 dim_feedforward: int, dropout: float):
        """

        method: constructor

        arguments:
         input_dim (int): the dimensionality of the input features
         num_classes (int): the number of classes
         d_model (int): the dimensionality of the embedding vectors
         nhead (int): the number of attention heads
         num_layers (int): the number of encoder blocks
         dim_feedforward (int): the dimensionality of the feedforward layer
         dropout (float): dropout rate

        return:
         none

        description:
         none
        """

        # set the class name
        #
        NEDCTransformer.__CLASS_NAME__ = self.__class__.__name__

        # call the parent class (nn.Module) constructor
        #
        super().__init__()
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: initializing transformer classifier" %
                  (__FILE__, ndt.__LINE__,
                   NEDCTransformer.__CLASS_NAME__, ndt.__NAME__))
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: model parameters - input_dim: %d," 
                  "num_classes: %d, d_model: %d, nhead: %d" %
                  (__FILE__, ndt.__LINE__,
                   NEDCTransformer.__CLASS_NAME__, ndt.__NAME__,
                   input_dim, num_classes, d_model, nhead))
        
        # creating the linear layer for the input embedding
        # output shape: (input_dim, d_model)
        #
        self.input_embedding = nn.Linear(input_dim, d_model)
       
        # no mask is needed for IMLD datasets
        # 
        self.mask = None 
        
        # creating the encoder blocks
        #
        encoder_blocks = []
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: creating %d encoder blocks" %
                  (__FILE__, ndt.__LINE__,
                   NEDCTransformer.__CLASS_NAME__, ndt.__NAME__, num_layers))
        
        # for each encoder block, create the self-attention and feedforward
        # blocks
        #
        for i in range(num_layers):
            
            # display informational message
            #
            if dbgl_g > ndt.BRIEF:
                print("%s (line: %s) %s::%s: creating encoder block %d" %
                      (__FILE__, ndt.__LINE__,
                       NEDCTransformer.__CLASS_NAME__, ndt.__NAME__, i+1))
            
            # creating the self-attention and feedforward blocks
            #
            self_attention = MultiHeadAttentionBlock(d_model = d_model,
                                                     num_heads = nhead,
                                                     dropout = dropout)
            
            # creating the feedforward block
            #
            feed_forward = FeedForwardBlock(d_model = d_model,
                                            d_ff = dim_feedforward,
                                            dropout = dropout)
            
            # creating the encoder block
            #
            encoder_block = EncoderBlock(self_attention_block = self_attention,
                                         feed_forward_block = feed_forward,
                                         dropout = dropout)
            
            # adding the encoder block to the list
            #
            encoder_blocks.append(encoder_block)
        
        # creating the encoder
        #
        self.encoder = Encoder(layers = nn.ModuleList(encoder_blocks))

        # creating the classifier
        #
        self.classifier = nn.Linear(d_model, num_classes)
    
    #
    # end of method
        
    def forward(self, x: Tensor) -> Tensor:
        """
        method: forward

        arguments:
         x (Tensor): Input tensor

        return:
         Tensor: Output tensor

        description:
         This method computes the forward pass of the transformer classifier
         for IMLD datasets.
        """
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: forward pass - input shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   NEDCTransformer.__CLASS_NAME__, ndt.__NAME__, str(x.shape)))
        
        # converting the input to a tensor as forward() expects a tensor
        #
        x = torch.FloatTensor(np.array(x))
        
        # moving the tensor to the default device
        #
        x = self.to_device(x)
                
        # expanded to have a seq_dim of 1 (batch_size, 1, features)
        # to make it work with the transformer architecture
        #
        x = x.unsqueeze(1)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: after unsqueeze - shape: %s" %
                  (__FILE__, ndt.__LINE__,
                   NEDCTransformer.__CLASS_NAME__, ndt.__NAME__, str(x.shape)))
        
        # apply the input embedding
        # output shape: (batch_size, 1, d_model)
        #
        x = self.input_embedding(x)
        
        # apply the encoder:
        #  output shape: (batch_size, 1, d_model)
        #
        x = self.encoder(x, mask = self.mask)
        
        # squeeze the seq_dim dimension
        # output shape: (batch_size, d_model)
        #
        x = x[:, 0, :] 
        x = self.classifier(x)
        
        # display informational message
        #
        if dbgl_g > ndt.BRIEF:
            print("%s (line: %s) %s::%s: output shape: %s" %
                (__FILE__, ndt.__LINE__,
                 NEDCTransformer.__CLASS_NAME__, ndt.__NAME__, str(x.shape)))
        
        # exit gracefully
        #
        return x
    #
    # end of method
#
# end of class

#
# end of file
