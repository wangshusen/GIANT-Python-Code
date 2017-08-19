import numpy
import sys
home_dir = '../'
sys.path.append(home_dir)
from ExperimentLogistic.demo import Demo


def loadData(dataname):
    filename = home_dir + 'Resource/' + dataname + '.npz'
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
    condnum = demo.condnum
    print('Condition number is ' + str(condnum))
    
    m = 4
    print('m = ' + str(m))
    err1 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 16
    print('m = ' + str(m))
    err2 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 64
    print('m = ' + str(m))
    err3 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    m = 256
    print('m = ' + str(m))
    err4 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact, newtoniter=newtoniter)
    
    return err1, err2, err3, err4, condnum

    
def main(NewtonIter, Gamma, ResultName): 
    #dataname = 'logis_U8'
    dataname = 'covtype'
    path = home_dir + 'Output/logis/'
    MaxIter = 30
    Repeat = 10
    IsSearch = False
    IsExact = False
    
    xMat, yVec = loadData(dataname)
    print(xMat)
    print(yVec)
    
    err1, err2, err3, err4, condnum = experiment(xMat, yVec, MaxIter, Repeat, Gamma, IsSearch, IsExact, newtoniter=NewtonIter)
    outfilename = path + dataname + ResultName + '.npz'
    numpy.savez(outfilename, err1=err1, err2=err2, err3=err3, err4=err4, dataname=dataname, maxiter=MaxIter, newtoniter=NewtonIter, condnum=condnum)
    
    

if __name__ == '__main__':
    
    NewtonIter = 20
    Gamma = 1e-2
    GammaName = '_gamma-2'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)
    
    NewtonIter = 65
    Gamma = 1e-3
    GammaName = '_gamma-3'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)
    
    NewtonIter = 199
    Gamma = 1e-4
    GammaName = '_gamma-4'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)
    
    NewtonIter = 623
    Gamma = 1e-5
    GammaName = '_gamma-5'
    ResultName = '_iter' + str(NewtonIter) + GammaName
    main(NewtonIter, Gamma, ResultName)
    
    
    
    
    
