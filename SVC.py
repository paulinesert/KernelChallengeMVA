# Code by J.Mairal, M.Arbel, J-P. Vert from MVA Kernel Methods class : https://mva-kernel-methods.github.io/course-2021-2022/

import numpy as np
import pickle as pkl
from scipy import optimize
from scipy.linalg import cho_factor, cho_solve

class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
       
    
    def fit(self, X, y):
       #### You might define here any variable needed for the rest of the code
        N = len(y)
        hXX = self.kernel(X, X)
        G = np.einsum('ij,i,j->ij',hXX,y,y)
        A = np.vstack((-np.eye(N), np.eye(N)))             
        b = np.hstack((np.zeros(N), self.C * np.ones(N)))  

        # Lagrange dual problem
        def loss(alpha):
            return -alpha.sum() + 0.5 * alpha.dot(alpha.dot(G))  #'''--------------dual loss ------------------ '''

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return -np.ones_like(alpha) + alpha.dot(G) # '''----------------partial derivative of the dual loss wrt alpha-----------------'''


        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha:  np.dot(alpha, y) # '''----------------function defining the equality constraint------------------'''        
        jac_eq = lambda alpha:   y  #'''----------------jacobian wrt alpha of the  equality constraint------------------'''
        fun_ineq = lambda alpha:  b - np.dot(A, alpha) # '''---------------function defining the ineequality constraint-------------------'''     
        jac_ineq = lambda alpha:  -A # '''---------------jacobian wrt alpha of the  inequality constraint-------------------'''
        
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq , 
                        'jac': jac_ineq})

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x

        ## Assign the required attributes
        
        margin_pointsIndices = (self.alpha > self.epsilon)
        boundaryIndices = (self.alpha > self.epsilon) * (self.C- self.alpha > self.epsilon )
        
        self.support = X[boundaryIndices] #'''------------------- A matrix with each row corresponding to a support vector ------------------'''
        
        self.margin_points = X[margin_pointsIndices]
        self.margin_points_AlphaY = y[margin_pointsIndices] * self.alpha[margin_pointsIndices]
        
        self.b = y[boundaryIndices][0] - self.separating_function(np.expand_dims(X[boundaryIndices][0],axis=0)) #''' -----------------offset of the linear classifier------------------ '''
        K_margin_points = self.kernel(self.margin_points, self.margin_points)
        self.norm_f = np.einsum('i,ij,j->', self.margin_points_AlphaY , K_margin_points, self.margin_points_AlphaY)


    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        x1 = self.kernel(self.margin_points, x)
        return np.einsum('ij,i->j',x1,self.margin_points_AlphaY)
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b> 0) - 1