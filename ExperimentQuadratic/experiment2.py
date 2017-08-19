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
    demo.fit(xMat, yVec, m=100) # set m=100 because we use 1, 10, or 100 executors
    
    s = 5000
    m = 1
    err1 = demo.testConvergence(m, s=s, isSearch=isSearch, isExact=isExact)
    m = 10
    err2 = demo.testConvergence(m, s=s, isSearch=isSearch, isExact=isExact)
    m = 100
    err3 = demo.testConvergence(m, s=s, isSearch=isSearch, isExact=isExact)
    
    s = 20000
    m = 1
    err4 = demo.testConvergence(m, s=s, isSearch=isSearch, isExact=isExact)
    m = 10
    err5 = demo.testConvergence(m, s=s, isSearch=isSearch, isExact=isExact)
    m = 100
    err6 = demo.testConvergence(m, s=s, isSearch=isSearch, isExact=isExact)
    
    return err1, err2, err3, err4, err5, err6


def plotConvergence(outfilename):
    npzfile = numpy.load(outfilename)
    dataname = str(npzfile['dataname'])
    err1 = npzfile['err1']
    err2 = npzfile['err2']
    err3 = npzfile['err3']
    err4 = npzfile['err4']
    err5 = npzfile['err5']
    err6 = npzfile['err6']
    
    repeat = err1.shape[0]
    
    # plot
    fig = plt.figure(figsize=(9, 8))
    
    for r in range(repeat):
        # s = 5000
        plt.semilogy(err1[r, :], color='greenyellow', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err2[r, :], color='maroon', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err3[r, :], color='lightblue', linestyle='-', linewidth=0.5, alpha=0.5)
        # s = 20000
        plt.semilogy(err4[r, :], color='lightgreen', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err5[r, :], color='lightpink', linestyle='-', linewidth=0.5, alpha=0.5)
        plt.semilogy(err6[r, :], color='skyblue', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # s = 5000
    line0, = plt.semilogy(numpy.median(err1, axis=0), color='g', linestyle='-', linewidth=3)
    line1, = plt.semilogy(numpy.median(err2, axis=0), color='r', linestyle='-', linewidth=3.5)
    line2, = plt.semilogy(numpy.median(err3, axis=0), color='b', linestyle='-', linewidth=3.5)
    # s = 20000
    line3, = plt.semilogy(numpy.median(err4, axis=0), color='darkgreen', linestyle='--', linewidth=3.5, marker='o', ms=8)
    line4, = plt.semilogy(numpy.median(err5, axis=0), color='darkred', linestyle='--', linewidth=3.5, marker='x', ms=8)
    line5, = plt.semilogy(numpy.median(err6, axis=0), color='darkblue', linestyle='--', linewidth=3, marker='^', ms=8)
    
    #plt.legend([line0, line3, line1, line4, line2, line5], ['s=5,000, m=1', 's=20,000, m=1', 's=5,000, m=10', 's=20,000, m=10', 's=5,000, m=100', 's=20,000, m=100'], fontsize=20)
    plt.xlabel('iterations (t)', fontsize=30)
    #plt.ylabel(r"$|| w - w^\star ||_2$", fontsize=28)
    plt.xticks([0, 5, 10, 15, 20, 25, 30], fontsize=28) 
    plt.yticks(fontsize=28) 
    plt.axis([0, 20, 1e-12, 2])
    plt.tight_layout()
    imagename = home_dir + 'Output/' + dataname + '.pdf'
    print(imagename)
    fig.savefig(imagename, format='pdf', dpi=1200)
    plt.show()

if __name__ == '__main__':  
    MaxIter = 20
    Repeat = 2
    Gamma = 1e-6
    IsSearch = False
    IsExact = True

    dataname = 'YearPredictionMSD'
    xMat, yVec = loadData(dataname)
    
    #dataname = 'U8'
    #xMat, yVec = loadData(dataname)
    
    err1, err2, err3, err4, err5, err6 = experiment(xMat, yVec, MaxIter, Repeat, Gamma, IsSearch, IsExact)
    
    outfilename = home_dir + 'Output/result_' + dataname + '.npz'
    numpy.savez(outfilename, err1=err1, err2=err2, err3=err3, err4=err4, err5=err5, err6=err6, dataname=dataname)
    
    plotConvergence(outfilename)
    
    
    
    
    