from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = len(y)
    num_classes = np.shape(W)[1]
    Wt = np.transpose(W)
    

    
    for i in range(num_train):
        cumm_score = 0
        class_score=0
        softmax = np.zeros(10)
        
        for j in range(num_classes):
            cumm_score = cumm_score+math.exp(np.dot(X[i],Wt[j]))
            
            softmax[j] = math.exp(np.dot(X[i],Wt[j]))
            
            
            if y[i]==j:
                class_score=math.exp(np.dot(X[i],Wt[j]))
                
                
        softmax = softmax/cumm_score
        softmax[y[i]] = softmax[y[i]]-1
        
        softmax = np.expand_dims(softmax,axis=-1)
        Xtemp = np.expand_dims(X[i],axis=-1)
        dWTemp = softmax*np.transpose(Xtemp)
        
        
        
        dW += np.transpose(dWTemp)
        
        loss += -math.log(class_score/cumm_score)
        
    
    loss /= num_train
    dW /= num_train
    
  
    
    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W 
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = len(y)
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #one hot encoding of Y
    z = np.max(y) + 1
    one_hot = np.eye(z)[y]
    
    scores = np.dot(X,W)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    scores=np.exp(scores)
    denom = np.expand_dims(np.sum(scores,axis=-1),axis=-1)
    
    normailized_scores = scores/denom
    loss = np.sum(-np.log(np.sum(normailized_scores*one_hot,axis=-1)))
    
    normailized_scores =   normailized_scores-one_hot  
    
    dW = np.dot(np.transpose(X),normailized_scores) 
    
    
    # Averege loss
    loss /= num_train
    dW /= num_train
    
    # Regularization
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
