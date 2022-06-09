import numpy as np
from sklearn.covariance import EmpiricalCovariance
from numpy.linalg import det,inv
from numpy import log

class GDA:
    '''
    Implements Gaussian Discriminant Analysis assuming constant prior over data
    '''
    def __init__(self):
        self.mean = []
        self.cov = []
        self.classes = None
        
    def fit(self,X,y):
        self.classes = np.unique(y)
        for c in self.classes:
            #Get the indices for each class
            idx = np.where(y==c)
            #Get class mean and covariance
            mean = X[idx].mean(axis=0)
            cov = EmpiricalCovariance().fit(X[idx]).covariance_
            self.mean.append(mean)
            self.cov.append(cov)
            
            # We will need to take the inverse of cov[c], if this matrix is singular, it cannot be computed
            assert det(self.cov[c]) != 0
            
        self.mean = np.array(self.mean)
        self.cov = np.array(self.cov)
        
    
    def predict(self,X):
        
        m,d = X.shape
        preds = []
        
        '''
        For each datapoint, compute the log posterior over class labels
        Assumes prior is uniform
        '''
        for i in range(m):
            xi = X[i].reshape(d,1)
            maxVal,maxClass = -float("inf"),None
            
            for c in self.classes:
                mu_c = self.mean[c].reshape(d,1)
                #compute log posterior
                val = -0.5*log(2*np.pi*det(self.cov[c]))- 0.5*(xi-mu_c).T@inv(self.cov[c])@(xi-mu_c)
                #Update argmax 
                if(val > maxVal):
                    maxVal = val
                    maxClass = c
            preds.append(maxClass)
        return preds
    
    
class LDA:
    '''
    Implements Gaussian Discriminant Analysis assuming constant prior over data and covariance of every classes are the same
    '''
    def __init__(self):
        self.mean = []
        self.cov = None
        self.classes = None
        
    def fit(self,X,y):
        self.classes = np.unique(y)
        for c in self.classes:
            #Get the indices for each class
            idx = np.where(y==c)

            mean = X[idx].mean(axis=0)
            self.mean.append(mean)
        #get dataset covariance
        self.cov = EmpiricalCovariance().fit(X).covariance_    
        assert det(self.cov) != 0
        self.mean = np.array(self.mean)
        self.cov = np.array(self.cov)
        
    
    def predict(self,X):
        
        m,d = X.shape
        preds = []
        
        '''
        For each datapoint, compute the log posterior over class labels
        Assumes prior is uniform
        '''
        for i in range(m):
            xi = X[i].reshape(d,1)
            maxVal,maxClass = -float("inf"),None
            
            for c in self.classes:
                mu_c = self.mean[c].reshape(d,1)
                #compute log posterior
                val = -0.5*log(2*np.pi*det(self.cov))- 0.5*(xi-mu_c).T@inv(self.cov)@(xi-mu_c)
                #Update argmax 
                if(val > maxVal):
                    maxVal = val
                    maxClass = c
            preds.append(maxClass)
        return preds
                
            
        
            
            
            