import numpy as np
import itertools
import pprint
    
class State:
    
    def __init__(self,n,q):
        self.n = n
        self.q = q
        self.T = np.zeros([n,q,q])
        self.basis = None # staticにしたい
        self.feature = None
        
    def hello(self):
        print('hello')
    
    def shape(self):
        return self.T.shape
    
    def show(self):
        print(self.T)
        
    def mk_basis(self):
        q = self.q
        n = self.n
        bss = np.zeros([(q*q)**n,n,q,q])
        p = itertools.product(range(0,q*q), repeat=n)
        i = 0
        for b in p:
            for j in range(0,n):
                for k in range(0,q*q):
                    #print(b,i,j,k)
                    if b[j]==k: 
                        #print(b,i,j,k)
                        bss[i][j][int(k/q)][k%q] = 1
            i = i+1
        self.basis = bss
    
    #def mk_feature():
    #    self.feature = 1#feature
    
    # theta = np.random.uniform(size=2*2*2).reshape([2,2,2])
    # s.basis[0]*(theta) #要素積