import numpy

def generateCovMatrix(d):
    covMatrix = numpy.zeros((d, d));
    for i in range(d):
        for j in range(d):
            covMatrix[i, j] = 0.5 ** (abs(i-j))
    return covMatrix


def mvtrnd(mu, Cov, v, n):
    '''
    Output:
    Produce M samples of d-dimensional multivariate t distribution
    Input:
    mu = mean (d dimensional numpy array or scalar)
    Cov = covariance matrix (d-by-d matrix)
    v = degrees of freedom
    n = # of samples to produce
    '''
    d = len(Cov)
    g = numpy.tile(numpy.random.gamma(v/2., 2./v, n), (d,1)).T
    Z = numpy.random.multivariate_normal(numpy.zeros(d), Cov, n)
    return mu + Z / numpy.sqrt(g)


def generateX(n, d, datatype, v=4):
    # ================= Generate the Bases ================= #
    mu = numpy.ones(d)
    covMatrix = generateCovMatrix(d);
    if datatype[0] == 'N':
        matX = mvtrnd(mu, covMatrix, v, n)
    elif datatype[0] == 'U':
        matX = numpy.random.multivariate_normal(mu, covMatrix, n)
    matU = numpy.linalg.qr(matX, mode='reduced')[0]
    
    # ============ Generate the Singular Values ============ #
    if datatype[1] == '1':
        vecS = numpy.logspace(0, -1, d)
        vecS = numpy.logspace(0, -4, d)
    elif datatype[1] == '6':
        vecS = numpy.logspace(0, -6, d)
    elif datatype[1] == '8':
        vecS = numpy.logspace(0, -8, d)
        
    # ================ Generate the X matrix ================ #
    matV = numpy.random.randn(d, d)
    matV = numpy.linalg.qr(matV, mode='reduced')[0]
    matX = matU * vecS.reshape(1, d)
    matX = numpy.dot(matX, matV)
    
    return matX


def generateXW(n, d, datatype):
    d1 = int(numpy.ceil(d / 5))
    vecW1 = numpy.ones((d1, 1))
    vecW2 = numpy.ones((d-2*d1, 1))
    vecW = numpy.concatenate((vecW1, 0.1 * vecW2, vecW1))

    matX = generateX(n, d, datatype)
    return matX, vecW

def generateDataLogis(n, d, datatype):
    d1 = int(numpy.ceil(d / 5))
    vecW1 = numpy.ones((d1, 1))
    vecW2 = numpy.ones((d-2*d1, 1))
    vecW = numpy.concatenate((vecW1, 0.1 * vecW2, vecW1))

    matX = generateX(n, d, datatype)
    vecF = numpy.dot(matX, vecW)
    
    vecExp = numpy.exp(vecF)
    vecProb = vecExp / (vecExp + 1)
    vecY = numpy.random.binomial(1, vecProb) * 2.0 - 1.0
    print(numpy.sum(vecY))
    print(vecY)
    
    return matX, vecY, vecW

def main():
    # specify parameter
    n = 200000 # number of samples
    d = 1000 # number of features
    datatype = 'N8' # N denotes "non-uniform", 8 denotes the condition number of X
    
    matX, vecY, vecW = generateDataLogis(n, d, datatype)

    #outfile = datatype + '_n=' + str(n) + '_d=' + str(d) + '.npz'
    outfile = 'logis_' + datatype + '.npz'
    print(matX.shape)
    
    numpy.savez(outfile, X=matX, w=vecW, y=vecY)
    
    npzfile = numpy.load(outfile)
    print(npzfile.files)

if __name__ == '__main__':  
    main()
