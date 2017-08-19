import numpy
import sys
home_dir = '../'
sys.path.append(home_dir)
#from Algorithm.ExecutorQuadratic import Executor
from Algorithm.ExecutorLogistic import Executor

EtaList = 1 / (4 ** numpy.arange(0, 10))

class Solver:
    def __init__(self, m=8):
        self.m = m
        self.executorList = list()
    
    
    def fit(self, xMat, yVec, s=None):
        '''
        Partition X and y to self.m blocks.
        If s is not given, then we set s=n/m and the partition has no overlap.
        '''
        n, d = xMat.shape
        perm = numpy.random.permutation(n)
        xMat = xMat[perm, :]
        yVec = yVec[perm, :]
        s1 = int(numpy.floor(n / self.m))
        n = int(s1 * self.m)
        
        iBegin = 0
        for i in range(self.m):
            idx = range(iBegin, iBegin+s1)
            iBegin += s1
            xBlk = xMat[idx, :]
            yBlk = yVec[idx, :].reshape(s1, 1)
        
            if s is None:
                executor = Executor(xBlk, yBlk)
            else:
                idx2 = numpy.random.choice(n, s, replace=False)
                xBlk2 = xMat[idx2, :]
                yBlk2 = yVec[idx2, :]
                executor = Executor(xBlk, yBlk, xMat2=xBlk2, yVec2=yBlk2)
                
            self.executorList.append(executor)
        
        self.n = n
        self.d = d
        
    
    def train(self, gamma, wopt, maxIter=20, isSearch=False, newtonTol=1e-100, newtonMaxIter=20, isExact=False):
        errorList = list()
        wnorm = numpy.linalg.norm(wopt)
        w = numpy.zeros((self.d, 1))
            
        err = numpy.linalg.norm(w - wopt) / wnorm
        errorList.append(err)
        
        self.etaList = EtaList
        self.numEta = len(self.etaList)
        
        for executor in self.executorList:
            executor.setParam(gamma, newtonTol, newtonMaxIter, isSearch, isExact, self.etaList)
        
        # iteratively update w
        for t in range(maxIter):
            wold = w
            w = self.update(w, gamma, isSearch)
            
            err = numpy.linalg.norm(w - wopt) / wnorm
            errorList.append(err)
            print('Iter ' + str(t) + ': error is ' + str(err))
            
            #diff = numpy.linalg.norm(w - wold)
            #print('The change between two iterations is ' + str(diff))
            #if diff < self.tolConverge:
            #   break
        
        self.w = w
        return errorList
    
    def update(self, w, gamma, isSearch):
        '''
        Perform one iteration of update
        '''
        # compute gradient
        grad = numpy.zeros((self.d, 1))
        for executor in self.executorList:
            grad += executor.computeGrad()
        #grad = grad / self.n + gamma * w
        grad = grad / self.m

        # compute Newton direction
        p = numpy.zeros((self.d, 1))
        for i in range(self.m):
            plocal = self.executorList[i].computeNewton(grad)
            p += plocal
        p /= self.m

        # broadcast p to all the executors
        for executor in self.executorList:
            executor.updateP(p)
            
        if isSearch:
            pg = -0.1 * numpy.sum(p * grad)
            objValVec = numpy.zeros(self.numEta + 1)
            for executor in self.executorList:
                objValVec += executor.objFunSearch()
            
            objValOld = objValVec[-1]
            for l in range(self.numEta):
                objValNew = objValVec[l]
                eta = self.etaList[l]
                if objValNew < objValOld + pg * eta:
                    break
            
            p *= eta
            for executor in self.executorList:
                executor.updateP(p)
                
        for executor in self.executorList:
            executor.updateW()
            
        # driver takes a Newton step
        w -= p
        return w

 