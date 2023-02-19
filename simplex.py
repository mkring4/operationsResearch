import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
import math


def initialize(n, m, A):
        
    # Initialize the basic matrix with the columns of A corresponding to the slack variable indices
    # and the non-basic matrix with the columns of A corresponding to the remaining design variable indices.
    
    notJ = np.arange(0, n-m, 1, dtype=np.int8)
    J  = np.zeros(m,np.int8)
    B = np.zeros((m,m))
    N = np.zeros((m,m))
    for i in range(0, m):
        J[i]        = n - m + i         # slack variable indices enter the basis
        B[:,i]      = A[:, J[i]]        # columns of A corresponding to the slack variable indices enter the basic matrix

    N = np.zeros((m,n-m))
    xB  = inv(B)@b                      # Basic variables
    xN  = np.zeros((m,1))               # Non-basic variables

    return notJ, J, B, N, xB, xN
 
def descent_direction(notJ, J, xB, xN, c, B, A):

    """
    
    Select a non-basic variable x_p, such that rc corresponding to index p is less than or equal to 0.
    x_p will enter the basis.

    If all reduced costs are greater than or equal to 0, an optimal solution has been found.
    
    """

    # cost vectors
    cB = c[J]
    cN = c[notJ]
    xOpt = np.zeros((n,1))

    for i, c_ in enumerate(cN):
        rc = c_ - np.transpose(cB)@inv(B)@A[:,notJ[i]]

        if rc < 0:
            pInd = i
            return pInd
        
        else:
            if i == len(cN) - 1:
                print("-----------------------------------")
                print("The optimal solution has been found!")
                xOpt[J] = xB
                xOpt[notJ] = xN
                print(f"xOpt = {xOpt}")
                zOpt = np.transpose(c)@xOpt
                print(f"zOpt = {zOpt}")
                quit()

def step(notJ, J, A, B, xB, n, p):
    
    """

    Calculate the distance to each constraint, i.e. alfa_i, such that:

    (xB)_i + alfa_i*(dB)_i = 0 <=> alfa_i = - (xB)_i / (dB)_i

    Note: If (dB)_i ≥ 0, then alfa_i = +∞

    alfa_q = min (alfa_i), i ∈ J^k 

    if all entries of dB ≥ 0, then the problem is unbounded and not interesting.

    x_q will leave the basis.

    """
    
    flag=False
    
    alfa = np.zeros(m)
    for i, xB_i in enumerate(xB):
        A_p = A[:,p]
        dB = -inv(B)@A_p

        if dB[i] >= 0:
            alfa[i] = math.inf
        else:
            flag=True
            alfa[i] = - xB_i/dB[i]
  
    if flag == False:
        print("The problem is unbounded.")
        quit()
    else:
        alfa_q = min(alfa)
        qInd = np.argmin(alfa)
        return qInd

def main(A, b, c, m: int, n: int):

    """ 
    
    Computes the optimal solution to the linear programming problem (in STD form (slack variables have the highest indices)):

        Max z(x)=cTx
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
        m: dimension of b
        n: dimension of x, must be >= 1
    
    Returns:
    
        xOpt:   solution to Ax=b
        zOpt:   Optimal value of objective function z(x)
   
    Intermediary:
   
        rc:     reduced costs of x and slack variables
        B:      m x m    basic matrix
        N:      m x n-m  non-basic matrix
        J:      index of basic matris
        notJ:   index of non-basic matrix
        p:      index of non-basic variable in notJ that is supposed to enter the basic matrix for the next iteration
        pInd:   index in notJ such that notJ[pInd] = p
        q:      index of basic variable in J that is supposed to enter the basic matrix for the next iteration
        qInd:   index in J such that J[qInd] = q

    """

    # Max iterations of the algorithm
    maxIter = 20

    # Initialize algo
    notJ, J, B, N, xB, xN = initialize(n=n, m=m, A=A)
    
    for k in range(0,maxIter):

        # Calculate descent direction & check optimality
        pInd = descent_direction(notJ=notJ, J=J, xB=xB, xN=xN, c=c, B=B, A=A)
        p = notJ[pInd]

        # Calculate step length & check unboundedness of problem
        qInd = step(notJ=notJ, J=J, A=A, B=B, xB=xB, n=n, p=p)
        q = J[qInd]

        # Update basis
        J[qInd], notJ[pInd] = p, q.copy()
        B = A[:,J]
        N = A[:,notJ]
        xB = inv(B)@b

if __name__ == "__main__":
    
    print("")
    print("This simplex algorithm can be used for solving LP-problems on standard form.")
    print("")
    print(" min     cTx")
    print(" s.t.    Ax=b")
    print("         x≥0")
    print("")
    m       = int(input('n - input # of rows in A-matrix: '))
    n       = int(input('m - input # of columns in A-matrix: '))
    
    print("Enter the entries of A in a single line (separated by space): ")
    entries = list(map(int, input().split()))
    A  = np.array(entries).reshape(m, n)

    print("Enter the entries of b in a single line (separated by space): ")
    b = np.array(list(map(int, input().split()))).reshape(m,1)

    print("Enter the entries of c in a single line (separated by space): ")
    c = np.array(list(map(int, input().split()))).reshape(n,1)
    if len(c) != n:
        print("Warning: length of c is not equal to n. \n Hint: Forgot slack vars?")

    main(m=m, n=n, A=A, b=b, c=c)



