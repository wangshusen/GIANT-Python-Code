import numpy
import matplotlib.pyplot as plt
import sys
home_dir = '../'
sys.path.append(home_dir)
import Util.Logistic as Logistic
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
        solver = Logistic.Solver(X=self.xMat, y=self.yVec)
        self.wopt, self.condnum = solver.newton(self.gamma)

    def testConvergence(self, m, s=None, isSearch=False, isExact=True, newtoniter=100):
        errMat = numpy.zeros((self.repeat, self.maxiter+1))
        for r in range(self.repeat):
            print(str(r) + '-th repeat,   m = ' + str(m))
            solver = Solver(m=m)
            solver.fit(self.xMat, self.yVec, s=s)
            err = solver.train(self.gamma, self.wopt, maxIter=self.maxiter, isSearch=isSearch, isExact=isExact, newtonMaxIter=newtoniter)
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


def experiment(xMat, yVec, maxiter, repeat, gamma, isSearch, isExact, newtoniter=100):
    demo = Demo(maxiter, repeat, gamma)
    demo.fit(xMat, yVec, m=256)
    
    m = 512
    errMat = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    
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
    
    
    #plt.legend([line0, line1, line2, line3], ['m=4', 'm=16', 'm=64', 'm=256'], fontsize=20)
    plt.xlabel('iterations (t)', fontsize=30)
    #plt.ylabel(r"$|| w - w^\star ||_2$", fontsize=28)
    plt.xticks([0, 5, 10, 15, 20, 25, 30], fontsize=28) 
    plt.yticks(fontsize=28) 
    plt.axis([0, 30, 1e-12, 5])
    plt.tight_layout()
    imagename = home_dir + 'output/demo_logis_' + dataname + '.pdf'
    fig.savefig(imagename, format='pdf', dpi=1200)
    plt.show()

if __name__ == '__main__': 
    MaxIter = 30
    Repeat = 2
    Gamma = 1e-6
    IsSearch = True
    IsExact = False
    NewtonIter = 50
 
    dataname = 'covtype'
    xMat, yVec = loadData(dataname)
    
    print(xMat)
    print(yVec)
    
    errMat = experiment(xMat, yVec, MaxIter, Repeat, Gamma, IsSearch, IsExact, newtoniter=NewtonIter)
    print(errMat)
    
    outfilename = home_dir + 'Output/demo_logis_' + dataname + '.npz'
    numpy.savez(outfilename, err=errMat, dataname=dataname)
    
    plotConvergence(outfilename)
    
    
    
    
    
    