import numpy
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler
home_dir = '../'


def randFeature(matX, s, sigma=None):
    n, d = matX.shape
    if sigma is None:
        idx = numpy.random.choice(n, 500, replace=False)
        sigma = estimate_param(matX[idx, :], matX[idx, :])
    #k = int(numpy.ceil(s * 0.8)) # can be tuned
    matW = numpy.random.standard_normal((d, s)) / sigma
    vecV = numpy.random.rand(1, s) * 2 * numpy.pi
    matL = numpy.dot(matX, matW) + vecV
    del matW
    matL = numpy.cos(matL) * numpy.sqrt(2/s)
    return matL
    #matUL, vecSL, matVL = numpy.linalg.svd(matL, full_matrices=False)
    #vecSL = vecSL[0:k]
    #return matUL[:, 0:k], vecSL

def estimate_param(x1_mat, x2_mat):
    n1 = x1_mat.shape[0]
    n2 = x2_mat.shape[0]
    w_mat = numpy.dot(x1_mat, x2_mat.T)
    row_norm1 = numpy.sum(numpy.square(x1_mat), 1) / 2
    row_norm2 = numpy.sum(numpy.square(x2_mat), 1) / 2
    w_mat = w_mat - row_norm1.reshape(n1, 1)
    w_mat = w_mat - row_norm2.reshape(1, n2)
    return numpy.sqrt(numpy.mean(-w_mat))


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

def main(dataname, s):
    outfilename = home_dir + 'Resource/rfm_' + dataname + '.npz'
    
    # load data
    X, y = loadData(dataname)
    n, d = X.shape
    print('Size of X is ' + str(n) + '-by-' + str(d))
    print('Size of y is ' + str(y.shape))
    y = y.reshape(n, 1)
    
    L = randFeature(X, s)
    numpy.savez(outfilename, X=L, y=y, dataname=dataname)
    

if __name__ == '__main__': 
    s = 1000 # number of random features
    
    dataname = 'YearPredictionMSD'
    main(dataname, s)

    dataname = 'covtype'
    main(dataname, s)
    
    dataname = 'w8a'
    main(dataname, s)
    
