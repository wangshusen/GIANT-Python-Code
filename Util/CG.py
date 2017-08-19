import numpy

def cgSolver(A, b, lam, Tol=1e-16, MaxIter=1000):
    '''
    Solve (A^T * A + lam * I) * w = b.
    '''
    d = A.shape[1]
    b = b.reshape(d, 1)
    tol = Tol * numpy.linalg.norm(b)
    w = numpy.zeros((d, 1))
    r = b - lam * w - numpy.dot(A.T, numpy.dot(A, w))
    p = r
    rsold = numpy.sum(r ** 2)

    for i in range(MaxIter):
        Ap = lam * p + numpy.dot(A.T, numpy.dot(A, p))
        alpha = rsold / numpy.dot(p.T, Ap)
        w += alpha * p
        r -= alpha * Ap
        rsnew = numpy.sum(r ** 2)
        if numpy.sqrt(rsnew) < tol:
            print('Converged! res = ' + str(rsnew))
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew    

    #print('res = ' + str(rsnew) + ',   iter = ' + str(i))
    #if i == MaxIter-1:
        #print('Warn: CG does not converge! Res = '  + str(rsnew))
    return w



def cgSolver2(AA, b, lam, Tol=1e-16, MaxIter=1000):
    '''
    Solve (A^T * A + lam * I) * w = b.
    '''
    d = AA.shape[1]
    b = b.reshape(d, 1)
    tol = Tol * numpy.linalg.norm(b)
    w = numpy.zeros((d, 1))
    r = b - lam * w - numpy.dot(AA, w)
    p = r
    rsold = numpy.sum(r ** 2)

    for i in range(MaxIter):
        Ap = lam * p + numpy.dot(AA, p)
        alpha = rsold / numpy.dot(p.T, Ap)
        w += alpha * p
        r -= alpha * Ap
        rsnew = numpy.sum(r ** 2)
        if numpy.sqrt(rsnew) < tol:
            print('Converged! res = ' + str(rsnew))
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew    

    #if i == MaxIter-1:
    #    print('Warn: CG does not converge! Res = '  + str(rsnew))
    return w


def svrgSolver(A, b, lam, alpha=0.01, Tol=1e-16, MaxIter=1000, BatchSize=100):
    '''
    Solve (A^T * A + lam * I) * w = b.
    '''
    s, d = A.shape
    b = b.reshape(d, 1)
    
    # parameter
    Scaling = s / BatchSize
    NumInnerLoop = int(numpy.ceil(Scaling))
    
    # initialize
    w = numpy.zeros((d, 1))
    
    for q in range(MaxIter):
        wtilde = numpy.copy(w)

        # compute full gradient at wtilde
        aw = numpy.dot(A, wtilde)
        gradFull = numpy.dot(A.T, aw) - b + lam * wtilde

        # mini-batch stochastic gradient
        for j in range(NumInnerLoop):
            idx = numpy.random.choice(s, BatchSize)
            Arand = A[idx, :]
            
            # the stochastic gradient at w
            aw = numpy.dot(Arand, w)
            grad1 = numpy.dot(Arand.T, aw) * Scaling - b + lam * w
            
            # the stochastic gradient at wtilde
            aw = numpy.dot(Arand, wtilde)
            grad2 = numpy.dot(Arand.T, aw) * Scaling - b + lam * wtilde
            
            gradRand = grad1 - grad2 + gradFull
            w -= alpha * gradRand
            
    return w
    