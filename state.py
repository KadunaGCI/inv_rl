import numpy as np
import itertools
import pprint
import pickle
import sys
    
class State:
    def __init__(self,n=None,q=None,T=None):
        if T is None:
            self.n = n
            self.q = q
            self.T = np.zeros([n,q,q])
        else:
            self.n, self.q, _ = T.shape
            self.T = T
        self.basis = None # staticにしたい
        self.feature = None
    
    def shape(self):
        return self.T.shape
    
    def show(self):
        print(self.T)

    def save(self,path):
        with open(path + '.pickle', 'wb') as f:
            pickle.dump(self, f)
        
    # wanna cash as static
    def mk_basis(self):
        q = self.q
        n = self.n
        bss = np.zeros([(q*q)**n,n,q,q])
        p = itertools.product(range(0,q*q), repeat=n)
        i = 0
        for b in p:
            for j in range(0,n):
                for k in range(0,q*q):
                    if b[j]==k: 
                        bss[i][j][int(k/q)][k%q] = 1
            i = i+1
        self.basis = bss
    
    #def mk_feature():
    #    self.feature = 1#feature
    
    # theta = np.random.uniform(size=2*2*2).reshape([2,2,2])
    # s.basis[0]*(theta) #要素積

if __name__ == '__main__':
    if (len(sys.argv)==1):
        s = State(5,5)
        s.mk_basis()
        #s.save('./')
    else:
        with open('tmp.pickle', 'rb') as f:
            print('LOADING')
            s = pickle.load(f)
    s.show()
    print('-----------')
    print(s.basis)
    