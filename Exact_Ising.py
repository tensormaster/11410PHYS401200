import numpy as np
import itertools

J = 1
kB = 1
Tc = 2/np.log(1+np.sqrt(2))

def gammas(T, n):
    beta = 1/(kB*T)
    K  = beta*J
    ls = np.arange(1,2*n)
    gamma_0  = 2*K + np.log(np.tanh(K))   # l = 0
    gamma_ls = np.arccosh(np.cosh(2*K)/np.tanh(2*K)-np.cos(ls*np.pi/n))  # l = 1, 2, 3, ..., 2n-1
    gammas   = np.append(gamma_0, gamma_ls)
    return gammas 

def Ei_kmax(n,T,num_collect, kmax=None):
    beta = 1/(kB*T)
    K = beta*J 
    ln_lambda_s = np.empty(0)
    
    for parity in ('even', 'odd'):
        if parity == 'even':
            gamma_array = gammas(T,n)[1::2]
        if parity == 'odd':
            gamma_array = gammas(T,n)[0::2]

        """lambda_0 and lambda_1 (all plus-sign for gamma)"""
        ln_lambda_s = np.append(ln_lambda_s, (np.log(2)+np.log(np.sinh(2*K)))*(n/2) + 1/2*np.sum(gamma_array))
        
        """lambda_i with 2k minus-sign in the summation of gamma"""
        if kmax == None:
            kmax = min([n,num_collect])
        
        gamma_array_sorted = np.sort(gamma_array)
        for k in np.arange(2, kmax+1, 2):
            for ij in itertools.combinations(np.arange(min([n,num_collect])), k):
                gamma_array_new = np.copy(gamma_array_sorted)
                gamma_array_new[list(ij)] *= -1
                ln_lambda_s = np.append(ln_lambda_s, (np.log(2)+np.log(np.sinh(2*K)))*(n/2) + 1/2*np.sum(gamma_array_new))

    ln_lambda_s = np.sort(ln_lambda_s)[::-1]

    return -1*ln_lambda_s[:num_collect]


if __name__ == "__main__":
    print("Exact solution of eigenvalues of the transfer matrix for the 2D Ising model")
    for L in range(1, 4+1):
        num_collect = 2**L
        E = Ei_kmax(L, Tc, num_collect)
        print("L={}".format(L))
        print(E)
