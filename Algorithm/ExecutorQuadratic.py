import numpy

import sys
home_dir = '../'
sys.path.append(home_dir)
import Util.CG as CG
    
class Executor:
    def __init__(self, xMat, yVec, xMat2=None, yVec2=None):
        self.s, self.d = xMat.shape
        self.xMat = xMat
        self.yVec = yVec.reshape(self.s, 1)
        
        # matrix A is for approximating the Hessian matrix
        if xMat2 is None:
            self.aMat = xMat / numpy.sqrt(self.s)
        else:
            s2 = xMat2.shape[0]
            self.aMat = xMat2 / numpy.sqrt(s2)
            
        s2 = self.aMat.shape[0]
        if s2 > self.d * 2:
            self.aaMat = numpy.dot(self.aMat.T, self.aMat)
            self.isAA = True
        else:
            self.isAA = False
        
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
        
        self.isExact = isExact
        if isExact:
            _, si, v = numpy.linalg.svd(self.aMat, full_matrices=False)
            si = 1 / (si**2 + self.gamma)
            hinv = v.T * si.reshape(1, len(si))
            self.hInvMat = numpy.dot(hinv, v)
        
    def updateP(self, p):
        self.p = p
        
    def updateW(self):
        self.w -= self.p
        
    def objFun(self, wVec):
        '''
        Compute 1/s*||X w - y||_2^2 + gamma * ||w||_2^2,
        where X and y are local data
        '''
        res = numpy.dot(self.xMat, wVec.reshape(self.d, 1)) - self.yVec.reshape(self.s, 1)
        loss = numpy.linalg.norm(res) ** 2 / self.s
        reg = numpy.linalg.norm(wVec) ** 2 * self.gamma
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
        res = numpy.dot(self.xMat, self.w) - self.yVec
        grad = numpy.dot(self.xMat.T, res) / self.s
        return grad + self.gamma * self.w
        
    def computeNewton(self, gVec):
        '''
        Compute the Newton direction using local data
        '''
        # Compute Newton's direction
        if self.isExact: # compute Newton's direction by inverting the Hessian
            pVec = numpy.dot(self.hInvMat, gVec)
            gradCalls = self.d
        else: # compute Newton's direction by CG
            p0 = numpy.zeros(self.d)
            pVec = CG.cgSolver2(numpy.dot(aMat.T, aMat), gVec, self.gamma, Tol=self.gtol, MaxIter=self.maxiter)
            self.gtol *= 0.5 # decrease the convergence error paramter of CG
        

        # Line search to decide step size
        self.isSearch = False ### temporary
        if self.isSearch:
            pg = -0.5 * numpy.sum(pVec * gVec)
            objValOld = self.objFun(self.w)
            for l in range(self.numEta):
                eta = self.etaList[l]
                objValNew = self.objFun(self.w - eta * pVec)
                if objValNew < objValOld + pg * eta:
                    break
            pVec = eta * pVec
            
        return pVec
    