"""
    Propensity Functions for running gillespie algorithm.

    function(x,**kwargs):
        return
    
    x : np.ndarray
    return value is transposed    
"""
import numpy as np
import numba

def loopModifyingPropensity(x,**kwargs):
    """return reaction propensities for given state x

    Parameters
        kwargs['km1']
        kwargs['kdecay']
        kwargs['K']
        kwargs['alpha']
        kwargs['knuc']
        kwargs['cen']
        
    Reaction rate depends on Contact probability
        references
        F. Erdel, K. Müller-Ott and K. Rippe, Ann. N. Y. Acad. Sci., 2013, 1305, 29–43.
        K. Rippe, Trends Biochem. Sci., 2001, 26, 733–740.
    
    """
    kdecay = kwargs.get("kdecay",0.006)
    K = kwargs.get("K",1)
    km1 = kwargs.get("km1",K*kdecay)
    alpha = kwargs.get("alpha",1)
    knuc = kwargs.get("knuc",1)
    cenIndex = kwargs.get("cen",0)
    N = np.size(x)
    x.flags.writeable = False
    
    ## modification from bound modifier
    @numba.njit
    def _modification(length,site,alpha):
        ## Contact probability
        nb = 154
        lengthPerBp = 0.13
        kuhnLength = lengthPerBp * nb
        d = 0.16
        
        def contactProb(x):
            return 0.53 * kuhnLength**(-3) * (x*lengthPerBp/kuhnLength)**(-3/2) * np.exp((d-2)/((x*lengthPerBp/kuhnLength)**2 + d))
        
        c0 = contactProb(1*nb)
        Km = 15 * 10**(-6)
        
        k = np.zeros((length,1))
        
        for j in range(length):
            dis = np.abs(site - j)
            keff = (contactProb(dis*nb)/c0)*(Km + c0)/(Km + contactProb(dis*nb))
            if dis > 0:
                if alpha:
                    k[j] = keff*alpha
                else: 
                    k[j] = keff
                    
        return k
    
    ## propensity logic
    def _action(x,kdecay,K,km1,alpha,knuc,cenIndex,N):
        X = np.copy(x).astype(np.float32).reshape(1,N)
        Y = np.copy(x).astype(np.float32).reshape(1,N)
        propBool = np.zeros((N,1))
        k = np.zeros((N,1))
        decayBool = np.transpose(Y).reshape(N,1)    # does not convert id?
        assert not id(X)==id(decayBool)
        if knuc == 1:
            decayBool[cenIndex,0] = 0     

        for i in range(N):
            if X[0,i]==0:
                propBool[i,0] = 1
            else:
                if i==cenIndex:
                    k += km1*_modification(N,i,0)
                if alpha and (i != cenIndex):
                    k += km1*alpha*_modification(N,i,alpha)
        
        assert k.shape == (N,1) == propBool.shape
        """
        print(x,X)
        print(k)
        print(propBool)
        print(decayBool)
        """
        return np.concatenate([k*propBool,kdecay*decayBool],axis=0) 
    
    ## propensity matrix
    a = _action(x,kdecay,K,km1,alpha,knuc,cenIndex,N)
    a.flags.writeable = False
    assert a.shape == (2*N,1)
    return a

def linearModifyingPropensity(x,**kwargs):
    """return reaction propensities for given state x

    Parameters
        kwargs['km1']
        kwargs['kdecay']
        kwargs['K']
        kwargs['alpha']
        kwargs['knuc']
        kwargs['cen']
    """
    kdecay = kwargs.get("kdecay",0.006)
    K = kwargs.get("K",1)
    km1 = kwargs.get("km1",K*kdecay)
    alpha = kwargs.get("alpha",1)
    knuc = kwargs.get("knuc",1)
    cenIndex = kwargs.get("cen",0)
    x.flags.writeable = False
    
    ## propensity logic
    N = np.size(x)
    X = np.copy(x).astype(np.float32).reshape(1,N)
    Y = np.copy(x).astype(np.float32).reshape(1,N)
    propBool = np.zeros((N,1))
    k = np.zeros((N,1))
    decayBool = np.transpose(Y)
    if knuc == 1:
        decayBool[cenIndex,0] = 0
    
    if not alpha: alpha=1
    for i in range(N):
        if X[0,i]==0:
            propBool[i,0] = 1
        else:
            if i==0:
                k[i+1,0] += alpha*km1
            elif i==N-1:
                k[-2,0] += alpha*km1
            elif i==cenIndex:
                k[i-1,0] += km1
                k[i+1,0] += km1
            else:
                k[i-1,0] += alpha*km1
                k[i+1,0] += alpha*km1
    
    ## propensity matrix
    a = np.concatenate([k*propBool,kdecay*decayBool],axis=0)
    a.flags.writeable = False
    assert a.shape==(2*N,1)
    
    return a