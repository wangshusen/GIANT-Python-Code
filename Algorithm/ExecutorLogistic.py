import numpy
from scipy import optimize 

import sys
home_dir = '../'
sys.path.append(home_dir)
import Util.CG as CG
    
class Executor:
    def __init__(self, xMat, yVec, xMat2=None, yVec2=None):
        self.s, self.d = xMat.shape
        self.yVec = yVec.reshape(self.s, 1)
        self.xMat = xMat * self.yVec

        # matrix A is for approximating the Hessian matrix
        if xMat2 is None:
            self.xMat2 = self.xMat
        else:
            s2 = xMat2.shape[0]
            self.xMat2 = xMat2 * yVec2.reshape(s2, 1)

        # initialize w and p
        self.w = numpy.zeros((self.d, 1))
        self.p = numpy.zeros((self.d, 1))
        

    def setParam(self, gamma, gtol, maxiter, isSearch, isExact, etaList):
        self.gamma = gamma
        self.gtol = gtol
        self.maxiter = maxiter
        self.isSearch = isSearch
        if isSearch:
            self.etaList = etaList
            self.numEta = len(self.etaList)
        
    def updateP(self, p):
        self.p = p
        
    def updateW(self):
        self.w -= self.p
        
    
    def objFun(self, wVec):
        '''
        f_j (w) = log (1 + exp(-w dot x_j)) + (gamma/2) * ||w||_2^2
        return the mean of f_j for all local data x_j
        '''
        zVec = numpy.dot(self.xMat, wVec.reshape(self.d, 1))
        lVec = numpy.log(1 + numpy.exp(-zVec))
        loss = numpy.mean(lVec)
        reg = self.gamma / 2 * numpy.sum(wVec ** 2)
        return loss + reg
    
    def objFunSearch(self):
        objValVec = numpy.zeros(self.numEta + 1)
        for l in range(self.numEta):
            objValVec[l] = self.objFun(self.w - self.etaList[l] * self.p)
        objValVec[-1] = self.objFun(self.w)
        return objValVec
    
    def computeGrad(self):
        '''
        Compute the gradient of the objective function using local data
        '''
        zVec = numpy.dot(self.xMat, self.w)
        expZVec = numpy.exp(zVec)
        vec1 = 1 + expZVec
        vec2 = -1 / vec1
        grad = numpy.mean(self.xMat * vec2, axis=0)
        return grad.reshape(self.d, 1) + self.gamma * self.w
    
    def computeNewton(self, gVec):
        zVec = numpy.dot(self.xMat2, self.w)
        expZVec = numpy.exp(zVec)
        expZVec = numpy.sqrt(expZVec) / (1 + expZVec)

        aMat = self.xMat2 * (expZVec / numpy.sqrt(self.s))
        
        #pVec = CG.svrgSolver(aMat, gVec, self.gamma, alpha=0.6, Tol=self.gtol, MaxIter=self.maxiter)
        pVec = CG.cgSolver(aMat, gVec, self.gamma, Tol=self.gtol, MaxIter=self.maxiter)
        #pVec = CG.cgSolver2(numpy.dot(aMat.T, aMat), gVec, self.gamma, Tol=self.gtol, MaxIter=self.maxiter)
        self.gtol *= 0.5 # decrease the convergence error paramter of CG
        
        return pVec
        
        
        