import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv

def simplex_iteration(A, b, c, m: int, n: int):

    """ 
    
    Computes the optimal solution to the linear programming problem:
    
        Max z(x)=c^Tx
        S.T. Ax=b
             x>=0
             
        x ∈ R^n
        c ∈ R^n
        A ∈ R^(m x n)
        b ∈ R^m
        
    Arguments:
    
        A: matrix of constraint equations coefficients {R^(m x n)}
        b: vector of initial constraint values {R^m}
        c: objective function coefficients {R^m}
        n: dimension of x, must be >= 1
        m: dimension of b
    
    Returns:
    
        x: solution to Ax=b
        rc: reduced costs of x and slack variables
        zOpt: Optimal value of objective function z(x)
        
    Intermediary:
    
        B: m x m  basis matrix
        NB: n x m non-basis matrix
    
    """
    
    # Initialization
    i = 0                       # initial iteration
    z = 0                       # initial objective value
    x = np.zeros(n+m)           # initial design values
    xB = np.zeros(m)
    cB = np.zeros(m)
    xN = np.zeros(n)
    cN = np.zeros(n)
    rc = np.zeros(n+m)
    Basis:int=np.zeros(n+m)
    b = np.zeros(n)