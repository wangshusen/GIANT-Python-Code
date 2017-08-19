import numpy
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
home_dir = '../'

def txt2npz(dataname, isTransformY=False, isRemoveEmpty=False):
    filename = home_dir + 'Resource/' + dataname
    X, y = load_svmlight_file(filename)
    X = numpy.array(X.todense())
    n, d = X.shape
    print('Size of X is ' + str(n) + '-by-' + str(d))
    print('Size of y is ' + str(y.shape))
    X = MinMaxScaler().fit_transform(X)
    
    if isTransformY:
        y = (y*2) - 3
        
    if isRemoveEmpty:
        sumX = numpy.sum(numpy.abs(X), axis=1)
        idx = (sumX > 1e-6)
        X = X[idx, :]
        y = y[idx]
        
    
    outfilename = home_dir + 'Resource/' + dataname + '.npz'
    numpy.savez(outfilename, dataname=dataname, X=X, y=y)


if __name__ == '__main__':  
    dataname = 'YearPredictionMSD'
    txt2npz(dataname)
    
    dataname = 'covtype'
    txt2npz(dataname, isTransformY=True)
    
    dataname = 'w8a'
    txt2npz(dataname, isRemoveEmpty=True)
    
    
    
