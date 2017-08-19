import numpy
from scipy import optimize 

import sys
home_dir = '../'
sys.path.append(home_dir)
import Util.CG as CG

class Solver:
    def __init__(self, X=None, y=None):
        if (X is not None) and (y is not None):
            self.n, self.d = X.shape
            self.xMat = X * y.reshape(self.n, 1)
        
    def fit(self, xMat, yVec):
        self.n, self.d = xMat.shape
        self.xMat = xMat * yVec.reshape(self.n, 1)
        
    def objFun(self, wVec, *args):
        gamma = args[0]
        zVec = numpy.dot(self.xMat, wVec.reshape(self.d, 1))
        lVec = numpy.log(1 + numpy.exp(-zVec))
        return numpy.mean(lVec) + gamma / 2 * numpy.sum(wVec ** 2)
    
    def grad(self, wVec, *args):
        gamma = args[0]
        zVec = numpy.dot(self.xMat, wVec.reshape(self.d, 1))
        expZVec = numpy.exp(zVec)
        vec1 = 1 + expZVec
        vec2 = -1 / vec1
        grad1 = numpy.mean(self.xMat * vec2, axis=0)
        grad = grad1.reshape(self.d) + gamma * wVec.reshape(self.d)
        return grad
        
    def cg(self, gamma, tol=1e-20, maxiter=5000):
        wVec0 = numpy.zeros(self.d)
        args = (gamma, )
        print(self.objFun(wVec0, *args))
        wVec, _, _, gradCalls, _ = optimize.fmin_cg(self.objFun, wVec0, args=args, fprime=self.grad, gtol=tol, maxiter=maxiter, disp=True, full_output=True)
        print(self.objFun(wVec, *args))
        return wVec
    
    def newton(self, gamma, maxIter=50, tol=1e-15):
        wVec = numpy.zeros((self.d, 1))
        etaList = 1 / (2 ** numpy.arange(0, 10))
        eyeMat = gamma * numpy.eye(self.d)
        args = (gamma, )
        
        for t in range(maxIter):
            zVec = numpy.dot(self.xMat, wVec.reshape(self.d, 1))
            expZVec = numpy.exp(zVec)
            loss = numpy.log(1 + 1 / expZVec)
            vec1 = 1 + expZVec
            vec2 = -1 / vec1
            vec3 = numpy.sqrt(expZVec) / vec1
            
            objVal = numpy.mean(loss) + numpy.sum(wVec ** 2) * gamma / 2
            #print('Iter ' + str(t) + ', objective value = ' + str(objVal))
            
            grad1 = numpy.mean(self.xMat * vec2, axis=0)
            grad = grad1.reshape(self.d, 1) + gamma * wVec
            
            gradNorm = numpy.sqrt(numpy.sum(grad ** 2))
            print('Iter ' + str(t) + ', L2 norm of gradient = ' + str(gradNorm))
            if gradNorm < tol:
                print('The change of obj val is smaller than ' + str(tol))
                break
            
            aMat = self.xMat * vec3
            #pVec = numpy.linalg.lstsq(hMat, grad)[0]
            pVec = CG.cgSolver(aMat / numpy.sqrt(self.n), grad, gamma, Tol=tol, MaxIter=100)
            
            if gradNorm > 1e-10:
                pg = -0.5 * numpy.sum(pVec * grad)
                for eta in etaList:
                    objValNew = self.objFun(wVec - eta*pVec, *args)
                    if objValNew < objVal + eta*pg:
                        break
            else:
                eta = 0.5
            wVec = wVec - eta * pVec
            
            
        hMat = numpy.dot(aMat.T, aMat) / self.n + eyeMat
        sig = numpy.linalg.svd(hMat, compute_uv=False)
        condnum = sig[0] / sig[-1]
        print('Condition number is ' + str(condnum))
        return wVec, condnum
    