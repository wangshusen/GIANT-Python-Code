import numpy
import matplotlib.pyplot as plt
import sys
home_dir = '../'
sys.path.append(home_dir)
from Algorithm.Solver import Solver


class Demo:
    def __init__(self, maxiter, repeat, gamma):
        self.maxiter = maxiter
        self.repeat = repeat
        self.gamma = gamma
        
    def fit(self, xMat, yVec, m=256):
        n, self.d = xMat.shape
        n = int(numpy.floor(n / m)) * m
        self.xMat = xMat[0:n, :]
        self.yVec = yVec[0:n].reshape(n, 1)
        self.n = n
        self.exactRidgeRegression()

    def exactRidgeRegression(self):
        '''
        Solve 1/2n * || X w - y ||_2^2 + gamma/2 * || w ||_2^2.
        '''
        u, si, v = numpy.linalg.svd(self.xMat, full_matrices=False)
        w = numpy.dot(u.T, self.yVec)
        del u
        si2 = si / (si**2 + self.n * self.gamma)
        w = w * si2.reshape(self.d, 1)
        self.wopt = numpy.dot(v.T, w)
        
    def testConvergence(self, m, s=None, isSearch=False, isExact=True):
        errMat = numpy.zeros((self.repeat, self.maxiter+1))
        for r in range(self.repeat):
            print(str(r) + '-th repeat,   m = ' + str(m))
            solver = Solver(m=m)
            solver.fit(self.xMat, self.yVec, s=s)
            err = solver.train(self.gamma, self.wopt, maxIter=self.maxiter, isSearch=isSearch, isExact=isExact)
            del solver
            l = min(len(err), self.maxiter+1)
            errMat[r, 0:l] = err[0:l]
        return errMat
        

def loadData(dataname):
    filename = home_dir + 'resource/' + dataname + '.npz'
    
    npzfile = numpy.load(filename)
    print(npzfile.files)
    X = npzfile['X']
    y = npzfile['y']
    n, d = X.shape
    #X = numpy.concatenate((X, numpy.ones((n, 1))), axis=1)

    print('Size of X is ' + str(n) + '-by-' + str(d))
    print('Size of y is ' + str(y.shape))
    
    return X, y

def experiment(xMat, yVec, maxiter, repeat, gamma, isSearch, isExact):
    demo = Demo(maxiter, repeat, gamma)
    demo.fit(xMat, yVec, m=256)
    
    m = 16
    errMat = demo.testConvergence(m, isSearch=isSearch, isExact=isExact)
    
    return errMat
    

def plotConvergence(outfilename):
    npzfile = numpy.load(outfilename)
    dataname = str(npzfile['dataname'])
    err = npzfile['err']
    
    repeat = err.shape[0]
    
    # plot
    fig = plt.figure(figsize=(9, 8))
    
    for r in range(repeat):
        plt.semilogy(err[r, :], color='greenyellow', linestyle='-', linewidth=0.5, alpha=0.5)
        
    line0, = plt.semilogy(numpy.median(err, axis=0), color='g', linestyle='-', linewidth=3)
    
    plt.legend([line0], ['m=16'], fontsize=20)
    plt.xlabel('iterations (t)', fontsize=30)
    plt.ylabel(r"$|| w - w^\star ||_2 / || w^\star ||_2$", fontsize=28)
    plt.xticks([0, 5, 10, 15, 20, 25, 30], fontsize=28) 
    plt.yticks(fontsize=28) 
    plt.axis([0, 20, 1e-12, 1.1])
    plt.tight_layout()
    imagename = home_dir + 'Output/demo_' + dataname + '.pdf'
    fig.savefig(imagename, format='pdf', dpi=1200)
    plt.show()

if __name__ == '__main__':  
    MaxIter = 20
    Repeat = 2
    Gamma = 1e-6
    IsSearch = True
    IsExact = True

    dataname = 'YearPredictionMSD'
    xMat, yVec = loadData(dataname)
    errMat = experiment(xMat, yVec, MaxIter, Repeat, Gamma, IsSearch, IsExact)
    print(errMat)
    
    outfilename = home_dir + 'Output/result_' + dataname + '.npz'
    numpy.savez(outfilename, err=errMat, dataname=dataname)
    
    plotConvergence(outfilename)
    
    
    
    
    