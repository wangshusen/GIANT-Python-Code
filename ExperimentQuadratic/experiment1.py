import numpy
import matplotlib.pyplot as plt
import sys
home_dir = '../'
sys.path.append(home_dir)
from ExperimentQuadratic.demo import Demo

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
    
    m = 4
    print('m = ' + str(m))
    err1 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact)
    m = 16
    print('m = ' + str(m))
    err2 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact)
    m = 64
    print('m = ' + str(m))
    err3 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact)
    m = 256
    print('m = ' + str(m))
    err4 = demo.testConvergence(m, isSearch=isSearch, isExact=isExact)
    
    return err1, err2, err3, err4


def plotConvergence(outfilename):
    npzfile = numpy.load(outfilename)
    dataname = str(npzfile['dataname'])
    err1 = npzfile['err1']
    err2 = npzfile['err2']
    err3 = npzfile['err3']
    err4 = npzfile['err4']
    
    repeat = err1.shape[0]
    
    # plot
    fig = plt.figure(figsize=(9, 8))
    
    for r in range(repeat):
        plt.semilogy(err1[r, :], color='greenyellow', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err2[r, :], color='salmon', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err3[r, :], color='skyblue', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err4[r, :], color='grey', linestyle='-', linewidth=0.5, alpha=0.5)  
    
    line0, = plt.semilogy(numpy.median(err1, axis=0), color='g', linestyle='-', linewidth=3)
    line1, = plt.semilogy(numpy.median(err2, axis=0), color='r', linestyle='-', linewidth=3.5)
    line2, = plt.semilogy(numpy.median(err3, axis=0), color='b', linestyle='-', linewidth=3.5)
    line3, = plt.semilogy(numpy.median(err4, axis=0), color='k', linestyle='-', linewidth=3.5)
    
    fontsize = 32
    #fontsize = 28
    #plt.legend([line0, line1, line2, line3], ['m=4', 'm=16', 'm=64', 'm=256'], fontsize=20)
    plt.xlabel('iterations (t)', fontsize=fontsize+2)
    #plt.ylabel(r"$|| w - w^\star ||_2$", fontsize=fontsize)
    #plt.title(r"$\gamma = 10^{-2}$", fontsize=fontsize+2)
    plt.xticks([0, 5, 10, 15, 20, 25, 30], fontsize=fontsize) 
    plt.yticks(fontsize=fontsize) 
    plt.axis([0, 20, 1e-12, 5])
    plt.tight_layout()
    imagename = home_dir + 'Output/quad/quadratic_' + dataname + '.pdf'
    print(imagename)
    fig.savefig(imagename, format='pdf', dpi=1200)
    plt.show()

if __name__ == '__main__':  
    MaxIter = 20
    Repeat = 20
    Gamma = 0
    IsSearch = True
    IsExact = True

    #dataname = 'YearPredictionMSD' # raw libsvm data
    #xMat, yVec = loadData(dataname)

    dataname = 'rfm_YearPredictionMSD' # raw libsvm data
    xMat, yVec = loadData(dataname)
    
    #dataname = 'N8' # synthetic data
    #xMat, yVec = loadData(dataname)
    
    err1, err2, err3, err4 = experiment(xMat, yVec, MaxIter, Repeat, Gamma, IsSearch, IsExact)
    
    outfilename = home_dir + 'Output/quad/quadratic_' + dataname + '.npz'
    numpy.savez(outfilename, err1=err1, err2=err2, err3=err3, err4=err4, dataname=dataname)
    
    plotConvergence(outfilename)
    
    
    
    
    